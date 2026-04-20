# Copyright 2025 Pathway Technology, Inc.

import dataclasses
import math
import time
import einops

from fla.ops.linear_attn import chunk_linear_attn, fused_recurrent_linear_attn
from fla.ops.gla import chunk_gla, fused_recurrent_gla
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
from nanochat.gpt import Linear

@dataclasses.dataclass
class BDHConfig:
    sequence_len: int = 2048
    n_layer: int = 8
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 2048
    add_gating: bool = True
    add_backout_lambda: bool = True
    add_resids: bool = True
    vocab_size: int = 64
    rotary_chunk_size: int = 256
    gate_divider: int = 1024
    # Target: ~500M parameters

def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),), eps=1e-5)

class BDHState:
    """
    Wrapper for BDH state. 
    """

    def __init__(self, batch_size, num_heads, head_dim, embedding_dim, num_layers, device):
        # state has shape [B, nh, N, D]
        self.batch_size = batch_size
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        self.embedding_dim = embedding_dim
        # Pre-allocate cache tensors: (B, nh, N, D)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.state = [torch.zeros(batch_size, num_heads, head_dim, embedding_dim, device=device, dtype=torch.float32) for _ in range(num_layers)]

    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        return self.cache_seqlens[0].item()

    def get_state(self, layer_idx):
        """Return state for a specific layer."""
        return self.state[layer_idx]

    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        """
        Copy state from another cache into this one.
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        """
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim and self.embedding_dim == other.embedding_dim
        assert len(self.state) == len(other.state)
        for i, state_layer in enumerate(self.state):
            other.state[i] = state_layer

class LinearSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, Q, V, gate, state):
        B, T, nh, N = Q.shape
        # Apparently this is what it takes to make the linear attention function not crash: shape it correctly
        _assert(V.shape[0] == B, "V does not match Q in the B dimension")
        _assert(V.shape[1] == T, "V does not match Q in the T dimension")
        _assert(V.shape[2] == nh, "V does not match Q in the nh dimension")
        _assert(Q.dtype == COMPUTE_DTYPE, "")
        # It seems that flash-linear-attention did not anticipate processing matrices this large,
        # judging by how illegal memory accesses happen if certain limits are exceeded
        # Split along the batch dimension
        BATCH_SPLIT_SIZE = max(2 ** 45 // (N * nh)**2 // T, 1)
        Q_parts = Q.split(BATCH_SPLIT_SIZE)
        V_parts = V.split(BATCH_SPLIT_SIZE)
        out_parts = [None for i in range(len(Q_parts))]
        if state is not None:
            state_parts = state.split(BATCH_SPLIT_SIZE)
        else:
            state_parts = tuple([None for i in range(len(Q_parts))])
        next_state_parts = [None for i in range(len(Q_parts))]
        if gate is not None:
            gate_parts = gate.split(BATCH_SPLIT_SIZE)
            for i in range(len(Q_parts)):
                if T >= 32:
                    out_part, next_state_part = chunk_gla(
                        Q_parts[i], 
                        Q_parts[i], 
                        V_parts[i],
                        -gate_parts[i],
                        initial_state = state_parts[i],
                        output_final_state = state is not None
                    )
                else:
                    out_part, next_state_part = fused_recurrent_gla(
                        Q_parts[i], 
                        Q_parts[i], 
                        V_parts[i],
                        -gate_parts[i],
                        initial_state = state_parts[i],
                        output_final_state = state is not None
                    )
                out_parts[i] = out_part
                if state is not None:
                    next_state_parts[i] = next_state_part
        else:
            for i in range(len(Q_parts)):
                if T >= 32:
                    out_part, next_state_part = chunk_linear_attn(
                        Q_parts[i], 
                        Q_parts[i], 
                        V_parts[i],
                        initial_state = state_parts[i],
                        output_final_state = state is not None
                    )
                else:
                    out_part, next_state_part = fused_recurrent_linear_attn(
                        Q_parts[i], 
                        Q_parts[i], 
                        V_parts[i],
                        initial_state = state_parts[i],
                        output_final_state = state is not None
                    )
                out_parts[i] = out_part
                if state is not None:
                    next_state_parts[i] = next_state_part
        
        torch.cuda.synchronize()
        '''
        out_parts = []
        if state is not None:
            state_parts = state.split(BATCH_SPLIT_SIZE)
            next_state_parts = []
        else:
            state_parts = tuple([None for i in range(len(Q_parts))])
            next_state_parts = None
        if gate is not None:
            gate_parts = gate.split(BATCH_SPLIT_SIZE)
            for Q_part, V_part, gate_part, state_part in zip(Q_parts, V_parts, gate_parts, state_parts):
                out_part, next_state_part = chunk_gla(
                    Q_part, 
                    Q_part, 
                    V_part,
                    -gate_part,
                    initial_state = state_part,
                    output_final_state = state is not None
                )
                out_parts.append(out_part)
                if state is not None:
                    next_state_parts.append(next_state_part)
        else:
            for Q_part, V_part, state_part in zip(Q_parts, V_parts, state_parts):
                out_part, next_state_part = chunk_linear_attn(
                    Q_part, 
                    Q_part, 
                    V_part,
                    initial_state = state_part,
                    output_final_state = state is not None
                )
                out_parts.append(out_part)
                if state is not None:
                    next_state_parts.append(next_state_part)
        '''

        if state is not None:
            state.copy_(torch.cat(next_state_parts))
        return torch.cat(out_parts)
        #if next_state is not None:
        #    state.copy_(next_state)
        #return out

class BDHLayer(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.decoder = Linear(nh * N, D, bias=False)
        self.encoder = Linear(D, nh * N, bias=False)
        if config.add_gating:
            self.encoder_gate = Linear(D, nh * N, bias=False)
        else:
            self.encoder_gate = None

        #import math
        #rope_angular_freqs = math.tau / 16 * torch.Tensor([65536 ** -(i / N) for i in range(self.config.rotary_chunk_size)]).to(dtype=torch.float)
        #rope_coeffs = torch.polar(torch.Tensor([1.0]), rope_angular_freqs)
        #self.register_buffer('rope_coeffs', rope_coeffs, persistent=False)
        self.rope_embed = torch.nn.utils.init.skip_init(
            RotaryEmbedding,
            dim = self.config.rotary_chunk_size,
            base = 2 ** 18, 
            scale_base = 512,
            pos_idx_in_fp32 = True, 
        )
        self.attn = LinearSelfAttention()

        # Investigate if RMSNorm could work here
        # Separate middle and end norm layers to avoid numerical type confusion
        self.middle_norm = LayerNorm(D, elementwise_affine=False, bias=False, dtype=COMPUTE_DTYPE)
        self.end_norm = LayerNorm(D, elementwise_affine=False, bias=False, dtype=torch.float32)
        self.dropout = nn.Dropout(config.dropout)

        self.encoder_v = nn.ModuleList([Linear(D, N, bias=False) for i in range(nh)])
        # The encoder_v buffers dict is empty
        #self.encoder_v = nn.Parameter(torch.zeros(nh * N, D))
        #with torch.device('meta'):
        #    self._encoder_v_wrap = nn.utils.skip_init(Linear, D, N, bias=False).to(device='meta')
        #    self._encoder_v_wrap.requires_grad = False
        #
        #def linear_wrapper(weights, data):
        #    return torch.func.functional_call(self._encoder_v_wrap, ({'weight': weights}, {}), data, strict=True)
        #
        #self.encoder_v_forward = torch.vmap(linear_wrapper, (0, -2), -2)


    def apply_rope(self, Q, t_offset):
        coeffs = rope_matrix(self.rope_coeffs, t_offset, Q.shape[1]).transpose(1, 0) # Dimensions T, N
        Q_rot = torch.stack((-Q[..., 1::2], Q[..., ::2]), dim=-1).view(*Q.size()) # Dimensions B, T, nh, N
        #left_part = torch.einsum('BTnN,TN->BTnN', Q, torch.real(coeffs))
        left_part = Q * torch.real(coeffs).unsqueeze(-2).expand_as(Q)
        #right_part = torch.einsum('BTnN,TN->BTnN', Q_rot, torch.imag(coeffs))
        right_part = Q_rot * torch.imag(coeffs).unsqueeze(-2).expand_as(Q)
        return (left_part + right_part).to(Q.dtype)
    
    def forward(self, x, state=None, t_offset=0):
        C = self.config
        nh = C.n_head
        D = C.n_embd
        N = D * C.mlp_internal_dim_multiplier // nh
        B, T, _ = x.size()
        _assert(x.dtype == torch.float32, "")
        # Expand x from [... D] to [... 1 D]
        #x_sparse = self.encoder(einops.rearrange(x.to(dtype=COMPUTE_DTYPE), '... D -> ... 1 D')) # B, T, nh, N
        x_unsqueeze = x.unsqueeze(-2).to(dtype=COMPUTE_DTYPE)
        x_sparse = sqrelu(self.encoder(x_unsqueeze).reshape(B, T, nh, N))

        # This RoPE implementation doesn't like D > 256, so give it a 
        # reshaped view that has lesser dimension and more heads
        x_sparse_reshaped = einops.rearrange(x_sparse.flatten(-2, -1), '... (x CS) -> ... x CS', CS = C.rotary_chunk_size)

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
        if self.encoder_gate is not None:
            x_gate = sqrelu(self.encoder_gate(x_unsqueeze).reshape(B, T, nh, N)) / C.gate_divider
        else:
            x_gate = None
        # Pass the time offset to the LinearSelfAttention layer
        yKV = self.attn(
            Q=x_sparse_rot,
            V=x_unsqueeze.expand(-1, -1, nh, -1),
            gate=x_gate,
            state=state
        )
        yKV = self.middle_norm(yKV) # B, T, nh, D
        _assert(yKV.dtype == COMPUTE_DTYPE, "")

        #y_sparse = sqrelu(self.encoder_v_forward(self.encoder_v.reshape(nh, N, D), yKV))
        # Works around some crash involving vmap and 8-bit scaled_mm, and it's not much of a loss
        # anyways since the vmap isn't optimized
        y_sparse = torch.stack([sqrelu(self.encoder_v[i](yKV[:,:,i,:])) for i in range(nh)], -2)
        y_sparse = self.dropout(y_sparse)

        #xy_sparse = einops.rearrange(x_sparse * y_sparse, '... nh N -> ... (nh N)')  # B, T, (nh * N)
        xy_sparse = torch.flatten(x_sparse * y_sparse, -2)

        yMLP = self.decoder(xy_sparse) # B, T, D
        _assert(yMLP.dtype == COMPUTE_DTYPE, "")
        y = self.end_norm(yMLP)
        next_x = norm(y + x) # x and y are both zero-mean, so we can just RMSnorm
        _assert(next_x.dtype == torch.float32, "")
        return next_x

class BDH(nn.Module):
    def __init__(self, config: BDHConfig, pad_vocab_size_to=64):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        D = config.n_embd
        self.bdh_layer = BDHLayer(self.config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.embed = nn.Embedding(padded_vocab_size, D, dtype=torch.float32)
        self.lm_head = Linear(D, padded_vocab_size, bias=False, dtype=torch.float32)
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        # The BDH paper also mentions merging predictions from other layers, which this is a simple form of
        if config.add_backout_lambda:
            self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        else:
            self.backout_lambda = None

        if config.add_resids:
            self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
            self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        else:
            self.resid_lambdas = None
            self.x0_lambdas = None

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        nh = self.config.n_head
        n_embd = self.config.n_embd
        N = self.config.mlp_internal_dim_multiplier * n_embd // nh

        init_decoder_sd = N**(-0.5)
        nn.init.normal_(self.bdh_layer.decoder.weight, mean=0.0, std=init_decoder_sd)

        init_sd = (n_embd)**-0.5 
        nn.init.normal_(self.bdh_layer.encoder.weight, mean=0.0, std=init_sd)
        if self.bdh_layer.encoder_gate is not None:
            nn.init.normal_(self.bdh_layer.encoder_gate.weight, mean=0.0, std=init_sd)
        for encoder_layer in self.bdh_layer.encoder_v:
            nn.init.normal_(encoder_layer.weight, mean=0.0, std=init_sd)
        #nn.init.normal_(self.bdh_layer.encoder_v, mean=0.0, std=init_sd)
        self.bdh_layer.rope_embed = RotaryEmbedding(
            dim = self.config.rotary_chunk_size,
            base = 2 ** 18, 
            scale_base = 512,
            pos_idx_in_fp32 = True, 
            device = self.get_device(),
        )

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        # Decaying x0 init: earlier layers get more input embedding blending
        n_layer = self.config.n_layer
        if self.backout_lambda is not None:
            self.backout_lambda = nn.Parameter(0.2 * torch.ones(1).to(device=self.get_device()))
        if self.resid_lambdas is not None:
            for i in range(n_layer):
                self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
                self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        if COMPUTE_DTYPE != torch.float16:
            self.embed.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return self.embed.weight.device
    
    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        #phantom_matrix_params = [self.bdh_layer._encoder_v_wrap.weight]
        #matrix_params = list(filter(lambda x: x is not self.bdh_layer._encoder_v_wrap.weight, self.bdh_layer.parameters()))
        matrix_params = list(self.bdh_layer.parameters())
        embedding_params = list(self.embed.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas] if self.resid_lambdas is not None else []
        x0_params = [self.x0_lambdas] if self.x0_lambdas is not None else []
        smear_params = [self.backout_lambda] if self.backout_lambda is not None else []
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 256 dim model)
        dmodel_lr_scale = (model_dim / 256) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/256) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        # Though I doubt 
        nparams_exclude = (self.embed.weight.numel() + self.lm_head.weight.numel() +
                          (self.resid_lambdas.numel() + self.x0_lambdas.numel() if self.resid_lambdas is not None else 0) +
                          (self.backout_lambda.numel() if self.backout_lambda is not None else 0))
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len

        # Assuming that t is small enough that using the matmul view is economical
        # Which is the operating region which training happens in
        # No sliding window either
        attn_flops = 12 * h * q * t * self.config.n_layer
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.embed.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        #bdh_matrices = sum(p.numel() for p in self.bdh_layer.parameters() if p is not self.bdh_layer._encoder_v_wrap.weight)
        bdh_matrices = sum(p.numel() for p in self.bdh_layer.parameters())
        scalars = (self.resid_lambdas.numel() + self.x0_lambdas.numel() if self.resid_lambdas is not None else 0) + (self.backout_lambda.numel() if self.backout_lambda is not None else 0)
        total = wte + lm_head + bdh_matrices + scalars
        # For completeness, because they will be counted even though no actual values back them
        #phantom_bdh_matrices = sum(p.numel() for p in self.bdh_layer._encoder_v_wrap.parameters())
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'lm_head': lm_head,
            'bdh_matrices': bdh_matrices,
            'scalars': scalars,
            'total': total,
        }
    
    # FORWARD METHOD IS NOW STATEFUL
    def forward(self, idx, targets=None, state=None, loss_reduction='mean'):
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).to(dtype=torch.float32)
        x = F.layer_norm(x, (D,), eps=1e-5)
        #x = self.end_norm(x.clone())  # B, T, D
        
        #with torch.autograd.graph.save_on_cpu(pin_memory=True), parametrize.cached():
        #torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
        #torch.compiler.set_stance("force_eager")

        #assert self.backout_lambda.device == self.get_device()

        backout_layer = C.n_layer // 2  # cache at halfway point
        x_backout = None
        x0 = x.clone()
        for i in range(C.n_layer):
            if self.resid_lambdas is not None:
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            next_x = self.bdh_layer(
                x,
                state.get_state(i) if state is not None else None,
                state.get_pos() if state is not None else 0,
            )
            x = next_x
            if self.backout_lambda is not None and i == backout_layer:
                x_backout = x.clone()
        
        if state is not None:
            state.advance(T)
        
        # Subtract mid-layer residual to remove low-level features before logit projection
        # (Pulled from nanochat)
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
            
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits_slice = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        
        logits = softcap * torch.tanh(logits / softcap) # squash the logits
        
        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            if loss_reduction == 'mean' and not (targets != -1).any():
                return logits.sum() * 0.0
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            return logits
    
    # GENERATE METHOD IS NOW STATEFUL AND EFFICIENT
    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        start_time = time.perf_counter()
        last_checkpoint = start_time
        C = self.config
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        device = self.get_device()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        state = BDHState(1, nh, N, D, C.n_layer, device)
        # The idx tensor will grow, but we only pass the newest token to the model
        for i in range(max_tokens):
            current_seq_len = idx.size(1)
            
            # On the first pass, process the whole prompt. On subsequent passes, only the last token.
            idx_cond = idx if i == 0 else idx[:, -1:]
            
            # The time offset is the length of the sequence already processed.
            t_offset = 0 if i == 0 else current_seq_len - 1
            
            logits, state = self(idx_cond, state=state, t_offset=t_offset)
            
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
