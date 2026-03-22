# Copyright 2025 Pathway Technology, Inc.

import dataclasses
import math
import time
import einops

from fla.ops.linear_attn import chunk_linear_attn
from fla.modules.rotary import RotaryEmbedding
from fla.modules.layernorm import LayerNorm
from fla.modules.activations import sqrelu
from fla.ops.utils import matmul

import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn, _assert
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 128
    dropout: float = 0.1
    n_head: int = 16
    mlp_internal_dim_multiplier: int = 16
    vocab_size: int = 64
    rotary_chunk_size: int = 256
    # Target: ~500M parameters

def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))

# Insight: The terms of rotational positional encoding can be expressed in 
# terms of a complex Vandermonde matrix
# Specifically, each term is e^(j(b+ax)t), which equals (e^jt)^(b) * (e^jt)^(ax)
def rope_matrix(complex_coeffs, t_offset, t_len):
    initial = torch.pow(complex_coeffs, t_offset)
    return torch.vander(complex_coeffs, N=t_len, increasing=True) * initial.reshape((-1, 1))

class LinearSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, Q, V, state, return_new_state=False):
        # This forward method now supports both parallel training and stateful generation
        B, T, nh, N = Q.shape
        D = V.shape[-1]
        _assert(V.shape[0] == B, "V does not match Q in the B dimension")
        _assert(V.shape[1] == T, "V does not match Q in the T dimension")
        
        new_state = None
        
        if state is None:
            output = torch.zeros((B, T, nh, D), device=Q.device, dtype=COMPUTE_DTYPE)
            if return_new_state:
                #new_state = torch.einsum('BTnN,BTD->BnND', Q, V.to(dtype=COMPUTE_DTYPE))
                # Flatten Q from [B T nh N] to [B T (nh N)], and transpose to [B (nh N) T]
                Q_transpose = Q.flatten(2).mT
                # Unflatten the product from [B (nh N) D] to [B nh N D]
                new_state = (Q_transpose @ V.to(dtype=COMPUTE_DTYPE)).unflatten(1, (nh, N))

        else:
            _assert(state.shape[0] == B, "state does not match Q in the B dimension")
            _assert(state.shape[1] == nh, "state does not match Q in the nh dimension")
            _assert(state.shape[2] == N, "state does not match Q in the N dimension")
            _assert(state.shape[3] == D, "state does not match V in the D dimension")
            output = torch.einsum('BTnN,BnND->BTnD', Q, state)
            # Permute Q from [B T nh N] to [B nh T N]
            #Q_permute = Q.transpose(1, 2)
            # Transpose the product from [B nh T D] to [B T nh D]
            #output = (Q_permute @ state).transpose(1, 2)
            if return_new_state:
                #new_state = state + torch.einsum('BTnN,BTD->BnND', Q, V.to(dtype=COMPUTE_DTYPE))
                # Flatten Q from [B T nh N] to [B T (nh N)], and transpose to [B (nh N) T]
                Q_transpose = Q.flatten(2).mT
                # Unflatten the product from [B (nh N) D] to [B nh N D]
                new_state = state + (Q_transpose @ V.to(dtype=COMPUTE_DTYPE)).unflatten(1, (nh, N))
        

        if T > 1:
            scores = torch.einsum('BTnN,BUnN->BnTU', Q, Q).tril(diagonal = -1)
            # Transpose the left Q from [B T nh N] to [B nh T N]
            #Q_permute = Q.transpose(1, 2)
            #scores = (Q_permute @ Q_permute.mT).tril(diagonal = -1)
            output += torch.einsum('BnTU,BUD->BTnD', scores, V.to(dtype=COMPUTE_DTYPE))
            # Transpose scores from [B nh T U] to [B T nh U], then flatten to [B (T nh) U]
            #scores_flatten = scores.transpose(1, 2).flatten(1, 2)
            # Unflatten the product from [B (T nh) D] to [B T nh D]
            #output += (scores_flatten @ V.to(dtype=COMPUTE_DTYPE)).unflatten(1, (T, nh))
            del scores
        
        return output, new_state
        '''
        # Implementation using flash-linear-attention
        # Keeps faulting for some reason
        return chunk_linear_attn(
            Q,
            Q,
            V.to(dtype=COMPUTE_DTYPE),
            initial_state=state,
            output_final_state=return_new_state
        )
        '''
        

class TransposeFromMatrix(nn.Module):
    @torch.no_grad()
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, X):
        # Reshape X from [(nh N) D] to [nh N D]
        new_last_dim = X.shape[0] // self.n_heads
        return torch.unflatten(X, 0, (self.n_heads, new_last_dim))

    def right_inverse(self, X):
        # Flatten from [nh N D] to [(nh N) D]
        return torch.flatten(X, 0, 1)

class SparseActivation(nn.Module):
    def __init__(self, n_heads, n_inputs, n_outputs):
        super().__init__()
        # A bit of weight trickery to let Muon optimize this
        self.weight = nn.Parameter(torch.zeros((n_heads, n_outputs, n_inputs), dtype=torch.bfloat16))
        param_transform = TransposeFromMatrix(n_heads)
        parametrize.register_parametrization(self, "weight", param_transform)
        # Doing vmap unexpectedly saves a large amount of vram
        self.vmap_linear = torch.vmap(
            F.linear,
            (-2, 0),
            -2
        )
    
    def forward(self, x):
        x_shape = x.shape
        x_sparse = self.vmap_linear(
            x.expand(*x.shape[:-2], self.weight.shape[0], -1),
            self.weight.to(dtype=x.dtype)
        )

        # Using ReLU^2 instead of ReLU for activations
        # Using this kernel here also saves a surprising amount of memory
        return sqrelu(x_sparse)

class BDHLayer(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.decoder = nn.Linear(nh * N, D, bias=False, dtype=torch.bfloat16)
        self.encoder = SparseActivation(nh, D, N)

        #import math
        #rope_angular_freqs = math.tau / 16 * torch.Tensor([65536 ** -(i / N) for i in range(self.config.rotary_chunk_size)]).to(dtype=torch.float)
        #rope_coeffs = torch.polar(torch.Tensor([1.0]), rope_angular_freqs)
        #self.register_buffer('rope_coeffs', rope_coeffs, persistent=False)
        self.rope_embed = RotaryEmbedding(
            dim = self.config.rotary_chunk_size,
            base = 2 ** 16,
            scale_base = 512
        )
        self.attn = LinearSelfAttention()

        # Investigate if RMSNorm could work here
        # Separate middle and end norm layers to avoid numerical type confusion
        self.middle_norm = LayerNorm(D, elementwise_affine=False, bias=False, dtype=torch.bfloat16)
        self.end_norm = LayerNorm(D, elementwise_affine=False, bias=False, dtype=torch.float32)
        self.dropout = nn.Dropout(config.dropout)
        self.encoder_v = SparseActivation(nh, D, N)

    @torch.no_grad()
    def init_weights(self):
        n_embd = self.config.n_embd
        init_sd = (n_embd)**-0.5 
        nn.init.normal_(self.decoder.weight, mean=0.0, std=init_sd)
        nn.init.normal_(self.encoder.weight, mean=0.0, std=init_sd)
        nn.init.normal_(self.encoder_v.weight, mean=0.0, std=init_sd)

    def apply_rope(self, Q, t_offset):
        coeffs = rope_matrix(self.rope_coeffs, t_offset, Q.shape[1]).transpose(1, 0).to(device='cuda') # Dimensions T, N
        Q_rot = torch.stack((-Q[..., 1::2], Q[..., ::2]), dim=-1).view(*Q.size()) # Dimensions B, T, nh, N
        #left_part = torch.einsum('BTnN,TN->BTnN', Q, torch.real(coeffs))
        left_part = Q * torch.real(coeffs).unsqueeze(-2).expand_as(Q)
        #right_part = torch.einsum('BTnN,TN->BTnN', Q_rot, torch.imag(coeffs))
        right_part = Q_rot * torch.imag(coeffs).unsqueeze(-2).expand_as(Q)
        return (left_part + right_part).to(Q.dtype)
    
    def forward_layer_front(self, x, state, t_offset, return_final_state):
        C = self.config
        nh = C.n_head
        D = C.n_embd
        N = D * C.mlp_internal_dim_multiplier // nh
        B, T, _ = x.size()
        # Expand x from [... D] to [... 1 D]
        #x_sparse = self.encoder(einops.rearrange(x.to(dtype=COMPUTE_DTYPE), '... D -> ... 1 D')) # B, T, nh, N
        x_sparse = self.encoder(x.to(dtype=COMPUTE_DTYPE).unsqueeze(-2))

        # This RoPE implementation doesn't like D > 256, so give it a 
        # reshaped view that has lesser dimension and more heads
        x_sparse_reshaped = einops.rearrange(x_sparse.flatten(-2, -1), '... (x CS) -> ... x CS', CS = C.rotary_chunk_size)
        #x_sparse_reshaped = x_sparse.view(B, T, -1, C.rotary_chunk_size)

        # With some luck, this should prod torch compile into removing 
        # the duplicate computation
        x_sparse_rot_unreshaped = self.rope_embed(
            x_sparse_reshaped, 
            x_sparse_reshaped, 
            seqlen_offset = t_offset
        )[0]

        # Then form it back to the actual shape
        x_sparse_rot = einops.rearrange(x_sparse_rot_unreshaped.flatten(-2, -1), '... (nh N) -> ... nh N', nh = nh)
        #x_sparse_rot = self.apply_rope(x_sparse_reshaped, t_offset).reshape(B, T, nh, -1)
        #.rearrange(, '... (nh f) CS -> ... nh (f CS)', nh = nh, CS = C.rotary_chunk_size)

        # Pass the time offset to the LinearSelfAttention layer
        yKV, next_state = self.attn(
            Q=x_sparse_rot,
            V=x,
            state=state,
            return_new_state=return_final_state
        )
        yKV = self.middle_norm(yKV) # B, T, nh, D
        return yKV, next_state
    
    def forward_layer_mid(self, x, yKV):
        C = self.config
        D = C.n_embd
        #x_reshaped = einops.rearrange(x.to(dtype=COMPUTE_DTYPE), '... D -> ... 1 D')
        x_reshaped = x.to(dtype=COMPUTE_DTYPE).unsqueeze(-2)
        x_sparse = self.encoder(x_reshaped) # B, T, nh, N
        
        y_sparse = self.encoder_v(yKV)
        y_sparse = self.dropout(y_sparse)
        #xy_sparse = einops.rearrange(x_sparse * y_sparse, '... nh N -> ... (nh N)')  # B, T, (nh * N)
        xy_sparse = torch.flatten(x_sparse * y_sparse, -2)
        del yKV
        del x_sparse
        del y_sparse
        yMLP = self.decoder(xy_sparse) # B, T, D
        del xy_sparse
        return yMLP
    
    def forward_layer_back(self, x, yMLP):
        y = self.middle_norm(yMLP)
        del yMLP
        next_x = norm(y + x) # x and y are both zero-mean, so we can just RMSnorm
        del y
        return next_x
    
    def forward(self, x, past_state=None, t_offset=0, return_final_state=False):
        # Avoid checkpointing some veeeery memory-intensive tensors
        yKV, layer_state = torch.utils.checkpoint.checkpoint(
            self.forward_layer_front,
            x, 
            past_state, 
            t_offset, 
            return_final_state,
            use_reentrant=False,
        )
        yMLP = torch.utils.checkpoint.checkpoint(
            self.forward_layer_mid,
            x,
            yKV,
            use_reentrant=False
        )
        
        del yKV
        #import gc
        #gc.collect(2)
        #torch.accelerator.memory.empty_cache()
        next_x = torch.utils.checkpoint.checkpoint(
            self.forward_layer_back,
            x,
            yMLP,
            use_reentrant=False,
        )
        del yMLP
        del x
        return next_x, layer_state


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        D = config.n_embd
        self.bdh_layer = BDHLayer(self.config)
        self.embed = nn.Embedding(config.vocab_size, D, dtype=torch.float32)
        self.lm_head = nn.Linear(D, config.vocab_size, bias=False, dtype=torch.float32)
        self.end_norm = LayerNorm(D, elementwise_affine=False, bias=False, dtype=torch.float32)

        self.init_weights()
        #self.apply(self._init_weights)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        self.bdh_layer.init_weights()
    
    # FORWARD METHOD IS NOW STATEFUL
    def forward(self, idx, targets=None, past_states=None, t_offset=0, return_final_state=False):
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx)
        x = self.end_norm(x)  # B, T, D

        if past_states is None:
            past_states = [None] * C.n_layer
            #past_states = [torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16, device=x.device).expand(B, nh, N, D)] * C.n_layer
        
        #with torch.autograd.graph.save_on_cpu(pin_memory=True), parametrize.cached():
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
        #torch.compiler.set_stance("force_eager")
        '''
        def scan_fn(x, st):
            next_x, next_st = self.bdh_layer(x, st, t_offset, return_final_state)
            return (next_x, torch.zeros(1) if next_st is None else next_st)

        x, present_states_tensor = torch._higher_order_ops.scan(
            scan_fn,
            x,
            torch.stack(past_states)
        )
        if return_final_state:
            present_states = [*present_states_tensor.unbind(0)]
        else:
            present_states = None
        '''
        present_states = []
        for i in range(C.n_layer):
            next_x, layer_state = self.bdh_layer(
                x,
                past_states[i],
                t_offset,
                return_final_state
            )
            present_states.append(layer_state)
            del layer_state
            del x
            x = next_x
        
            
        logits = self.lm_head(x)
        loss = None
        if targets is not None and T > 1: # Calculate loss only during training
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        final_states = None
        if return_final_state:
            final_states = present_states
        
        return logits, loss, final_states
    
    # GENERATE METHOD IS NOW STATEFUL AND EFFICIENT
    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        start_time = time.perf_counter()
        last_checkpoint = start_time
        states = None
        # The idx tensor will grow, but we only pass the newest token to the model
        for i in range(max_new_tokens):
            current_seq_len = idx.size(1)
            
            # On the first pass, process the whole prompt. On subsequent passes, only the last token.
            idx_cond = idx if i == 0 else idx[:, -1:]
            
            # The time offset is the length of the sequence already processed.
            t_offset = 0 if i == 0 else current_seq_len - 1
            
            logits, _, states = self(idx_cond, past_states=states, t_offset=t_offset, return_final_state=True)
            
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if i % 100 == 0 and i > 0:
                now = time.perf_counter()
                elapsed = now - last_checkpoint
                total_elapsed = now - start_time
                print(f"Generation, token {i}, last 100 tokens took {elapsed:.2f}s (total {total_elapsed:.2f}s)")
                last_checkpoint = now
        return idx


def load_checkpoint(model, weights_optimizer, embed_head_optimizer, checkpoint_path):
    """Load model and optimizer from checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    weights_optimizer.load_state_dict(checkpoint['weights_optimizer'])
    embed_head_optimizer.load_state_dict(checkpoint['embed_head_optimizer'])
    return checkpoint['step']
