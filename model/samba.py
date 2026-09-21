"""Samba architecture for Basic Block prediction in DOOM.

Combines Mamba (Selective State Space Model) with Sliding Window Attention (SWA)
and SwiGLU MLP into a unified hybrid architecture for ultra-fast, low-drift
hardware emulation.

Reference:
Ren et al., "Samba: Simple Hybrid State Space Models for Efficient Unlimited
Context Language Modeling" (Microsoft Research, 2024).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaSSM(nn.Module):
    """Selective State Space Model (SSM) core.

    Implements input-dependent selection mechanism:
      Δ = softplus(Linear(x) + Δ_bias)
      B = Linear(x)
      C = Linear(x)
      A_bar = exp(Δ * A)
      B_bar = Δ * B
      h_t = A_bar * h_{t-1} + B_bar * x_t
      y_t = C * h_t + D * x_t
    """

    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank

        # In-projection to expand dimension
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=2, groups=d_model
        )

        # Selection mechanism projections (Δ, B, C)
        self.x_proj = nn.Linear(d_model, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        # S4D real initialization for A: A_d = -exp(log_A_d)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

        # Out-projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, prev_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequence forward pass.

        Args:
            x: (B, L, D) input tensor.
            prev_state: Optional (B, D, N) initial hidden state.

        Returns:
            out: (B, L, D) output tensor.
            final_state: (B, D, N) last hidden state for recurrent chaining.
        """
        B, L, D = x.shape
        # Expand and split into x and gate
        xz = self.in_proj(x)  # (B, L, 2D)
        x_in, z = xz.chunk(2, dim=-1)

        # 1D causal convolution
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Compute input-dependent Δ, B, C
        ssm_params = self.x_proj(x_conv)  # (B, L, dt_rank + 2*d_state)
        dt_raw = ssm_params[:, :, : self.dt_rank]
        B_ssm = ssm_params[:, :, self.dt_rank : self.dt_rank + self.d_state]
        C_ssm = ssm_params[:, :, self.dt_rank + self.d_state :]

        # Δ = softplus(Linear(dt_raw))
        dt = F.softplus(self.dt_proj(dt_raw))  # (B, L, D)

        # Discretize A: A_bar = exp(Δ * -exp(A_log))
        A = -torch.exp(self.A_log)  # (D, N)
        # dt: (B, L, D, 1), A: (1, 1, D, N) -> dt * A: (B, L, D, N)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # dB = dt * B: (B, L, D, N)
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)

        # Recurrent scan over sequence length L
        h = (
            prev_state
            if prev_state is not None
            else torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        )
        ys: list[torch.Tensor] = []
        for t in range(L):
            # h_t = dA_t * h_{t-1} + dB_t * x_t
            h = dA[:, t] * h + dB[:, t] * x_conv[:, t].unsqueeze(-1)
            # y_t = C_t * h_t
            y_t = (h * C_ssm[:, t].unsqueeze(1)).sum(dim=-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, L, D)
        # Skip connection D * x
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        # Multiplicative gating with z
        out = y * F.silu(z)
        return self.out_proj(out), h

    def step(
        self, x_t: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """O(1) single-step recurrent inference.

        Args:
            x_t: (B, D) input vector at time t.
            state: (B, D, N) previous recurrent hidden state.

        Returns:
            out_t: (B, D) output vector.
            new_state: (B, D, N) updated state.
        """
        xz = self.in_proj(x_t)
        x_in, z = xz.chunk(2, dim=-1)
        x_act = F.silu(x_in)

        ssm_params = self.x_proj(x_act)
        dt_raw = ssm_params[:, : self.dt_rank]
        B_ssm = ssm_params[:, self.dt_rank : self.dt_rank + self.d_state]
        C_ssm = ssm_params[:, self.dt_rank + self.d_state :]

        dt = F.softplus(self.dt_proj(dt_raw))  # (B, D)
        A = -torch.exp(self.A_log)  # (D, N)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))  # (B, D, N)
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(1)  # (B, D, N)

        new_state = dA * state + dB * x_act.unsqueeze(-1)
        y_t = (new_state * C_ssm.unsqueeze(1)).sum(dim=-1) + x_act * self.D
        out_t = y_t * F.silu(z)
        return self.out_proj(out_t), new_state


class SlidingWindowAttention(nn.Module):
    """Multi-Head Self-Attention with Causal Sliding Window Mask."""

    def __init__(self, d_model: int, num_heads: int, window_size: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, L, D = x.shape
        q = (
            self.q_proj(x)
            .view(B, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(B, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(B, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)[:, :, -self.window_size :]
            v = torch.cat([v_prev, v], dim=2)[:, :, -self.window_size :]
        new_kv = (k, v)

        # Sliding window causal attention mask
        total_k_len = k.shape[2]
        # Query at pos i can attend to keys in [i - window_size, i]
        mask = torch.ones(
            (L, total_k_len), device=x.device, dtype=torch.bool
        ).tril(diagonal=total_k_len - L)
        if total_k_len > self.window_size:
            # Mask out keys older than window_size
            mask = mask.triu(diagonal=total_k_len - L - self.window_size + 1)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False
        )
        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(B, L, D)
        )
        return self.out_proj(attn_out), new_kv


class SwiGLUMLP(nn.Module):
    """Gated SwiGLU Feed-Forward Network."""

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SambaBlock(nn.Module):
    """Single Samba Hybrid Block: Mamba SSM + SWA + SwiGLU MLP."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_state: int = 16,
        window_size: int = 64,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mamba = MambaSSM(d_model, d_state=d_state)

        self.ln2 = nn.LayerNorm(d_model)
        self.swa = SlidingWindowAttention(
            d_model, num_heads, window_size=window_size
        )

        self.ln3 = nn.LayerNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, d_model * ffn_mult)

    def forward(
        self,
        x: torch.Tensor,
        ssm_state: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]
    ]:
        # Mamba sub-layer
        h_mamba, new_ssm = self.mamba(self.ln1(x), prev_state=ssm_state)
        x = x + h_mamba

        # SWA sub-layer
        h_swa, new_kv = self.swa(self.ln2(x), kv_cache=kv_cache)
        x = x + h_swa

        # MLP sub-layer
        x = x + self.mlp(self.ln3(x))
        return x, new_ssm, new_kv

    def step(
        self,
        x_t: torch.Tensor,
        ssm_state: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]
    ]:
        """Single-step recurrent execution."""
        h_mamba, new_ssm = self.mamba.step(self.ln1(x_t), ssm_state)
        x_t = x_t + h_mamba

        # SWA step
        x_seq = x_t.unsqueeze(1)
        h_swa, new_kv = self.swa(self.ln2(x_seq), kv_cache=kv_cache)
        x_t = x_t + h_swa.squeeze(1)

        x_t = x_t + self.mlp(self.ln3(x_t))
        return x_t, new_ssm, new_kv


def sign_extend(val: int, bits: int) -> int:
    sign_bit = 1 << (bits - 1)
    return (val & (sign_bit - 1)) - (val & sign_bit)


def decode_branch_target(inst: int, pc: int, regs: list[int]) -> int:
    op = inst & 0x7F
    if op == 0x63:  # Branch (BEQ, BNE, BLT, BGE, BLTU, BGEU)
        imm = sign_extend(
            ((inst >> 31) << 12)
            | (((inst >> 7) & 1) << 11)
            | (((inst >> 25) & 0x3F) << 5)
            | (((inst >> 8) & 0xF) << 1),
            13,
        )
        return (pc + imm) & 0xFFFFFFFF
    elif op == 0x6F:  # JAL
        imm = sign_extend(
            ((inst >> 31) << 20)
            | (((inst >> 12) & 0xFF) << 12)
            | (((inst >> 20) & 1) << 11)
            | (((inst >> 21) & 0x3FF) << 1),
            21,
        )
        return (pc + imm) & 0xFFFFFFFF
    elif op == 0x67:  # JALR
        imm = sign_extend(inst >> 20, 12)
        rs1 = (inst >> 15) & 0x1F
        return (regs[rs1] + imm) & ~1 & 0xFFFFFFFF
    return (pc + 4) & 0xFFFFFFFF


class BlockPrediction(NamedTuple):
    """Output of DoomSambaPredictor."""
    next_pc_offset: torch.Tensor       # (B, 1) or (B, L, 1) predicted delta to PC
    branch_taken_logits: torch.Tensor # (B, 1) or (B, L, 1) branch taken vs fallthrough
    reg_mask_logits: torch.Tensor     # (B, 32) or (B, L, 32) which regs written
    reg_vals: torch.Tensor            # (B, 32) or (B, L, 32) predicted new values
    mem_has_write: torch.Tensor       # (B, 2) or (B, L, 2) write flags for 2 slots
    mem_addrs: torch.Tensor           # (B, 2) or (B, L, 2) write addresses
    mem_vals: torch.Tensor            # (B, 2) or (B, L, 2) write values
    mem_sizes: torch.Tensor           # (B, 2, 3) or (B, L, 2, 3) sizes (1, 2, 4)


class DoomSambaPredictor(nn.Module):
    """Full Samba-based Basic Block Predictor for DOOM."""

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_state: int = 16,
        window_size: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # Input projections
        # PC embedding: frequency / Fourier features + linear
        self.pc_proj = nn.Linear(32, d_model // 4)
        # Register file projection: 32 registers * 32 bits -> d_model
        self.regs_proj = nn.Linear(32, d_model - d_model // 4)

        # Samba Stack
        self.blocks = nn.ModuleList(
            [
                SambaBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_state=d_state,
                    window_size=window_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)

        # Multi-task output heads
        # 1. Next PC delta head (offset from start_pc)
        self.head_pc = nn.Linear(d_model, 1)

        # 2. Branch taken vs fallthrough classification (BCE)
        self.head_branch = nn.Linear(d_model, 1)

        # 3. Register write mask: 32 binary logits (BCE)
        self.head_reg_mask = nn.Linear(d_model, 32)

        # 4. Register write values: 32 values
        self.head_reg_vals = nn.Linear(d_model, 32)

        # 5. Memory write slots (up to 2 slots)
        self.head_mem_flag = nn.Linear(d_model, 2)       # Has write
        self.head_mem_addr = nn.Linear(d_model, 2)       # Addr
        self.head_mem_val = nn.Linear(d_model, 2)        # Value
        self.head_mem_size = nn.Linear(d_model, 2 * 3)   # Size logits: 1, 2, 4

    def _embed_input(
        self, start_pc: torch.Tensor, regs_in: torch.Tensor
    ) -> torch.Tensor:
        """Encode start_pc and 32 registers into d_model embedding."""
        # Convert PC to 32 bit features: (B, ..., 32)
        pc_bits = (
            (start_pc.unsqueeze(-1) >> torch.arange(32, device=start_pc.device)) & 1
        ).float()
        pc_emb = self.pc_proj(pc_bits)

        # Normalize registers (scale to [-1, 1] range)
        regs_norm = (regs_in.float() - 2147483648.0) / 2147483648.0
        regs_emb = self.regs_proj(regs_norm)

        return torch.cat([pc_emb, regs_emb], dim=-1)

    def forward(
        self,
        start_pc: torch.Tensor,
        regs_in: torch.Tensor,
        states: list[torch.Tensor] | None = None,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[
        BlockPrediction,
        list[torch.Tensor],
        list[tuple[torch.Tensor, torch.Tensor]],
    ]:
        """Sequence forward pass."""
        x = self._embed_input(start_pc, regs_in)

        new_states: list[torch.Tensor] = []
        new_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []

        for i, blk in enumerate(self.blocks):
            s = states[i] if states is not None else None
            kv = kv_caches[i] if kv_caches is not None else None
            x, s_out, kv_out = blk(x, ssm_state=s, kv_cache=kv)
            new_states.append(s_out)
            new_kvs.append(kv_out)

        h = self.ln_f(x)

        pred = BlockPrediction(
            next_pc_offset=self.head_pc(h),
            branch_taken_logits=self.head_branch(h),
            reg_mask_logits=self.head_reg_mask(h),
            reg_vals=self.head_reg_vals(h),
            mem_has_write=self.head_mem_flag(h),
            mem_addrs=self.head_mem_addr(h),
            mem_vals=self.head_mem_val(h),
            mem_sizes=self.head_mem_size(h).view(*h.shape[:-1], 2, 3),
        )
        return pred, new_states, new_kvs

    @torch.no_grad()
    def step_block(
        self,
        start_pc: int,
        regs_in: list[int],
        states: list[torch.Tensor],
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        end_pc: int | None = None,
        end_inst: int | None = None,
    ) -> tuple[
        int,
        dict[int, int],
        list[list[int]],
        list[torch.Tensor],
        list[tuple[torch.Tensor, torch.Tensor]],
    ]:
        """Single-step recurrent closed-loop execution.

        Returns:
            predicted_next_pc: int
            reg_writes: {reg_idx: new_val}
            mem_writes: [[addr, val, size], ...]
            new_states: updated SSM states
            new_kvs: updated SWA KV caches
        """
        dev = next(self.parameters()).device
        pc_t = torch.tensor([[start_pc]], device=dev, dtype=torch.int64)
        regs_t = torch.tensor([[regs_in]], device=dev, dtype=torch.int64)

        pred, new_states, new_kvs = self.forward(
            pc_t, regs_t, states=states, kv_caches=kv_caches
        )

        # 1. Next PC: branch prediction if end instruction known, else offset fallback
        if end_pc is not None and end_inst is not None:
            branch_prob = torch.sigmoid(pred.branch_taken_logits[0, 0, 0]).item()
            if branch_prob > 0.5:
                pred_next_pc = decode_branch_target(end_inst, end_pc, regs_in)
            else:
                pred_next_pc = (end_pc + 4) & 0xFFFFFFFF
        else:
            pc_offset = int(pred.next_pc_offset[0, 0, 0].item() * 4)
            pred_next_pc = (start_pc + pc_offset) & 0xFFFFFFFF

        # 2. Register writes
        mask_probs = torch.sigmoid(pred.reg_mask_logits[0, 0])
        reg_writes: dict[int, int] = {}
        for r in range(1, 32):  # x0 is hardwired to 0
            if mask_probs[r].item() > 0.5:
                raw_val = pred.reg_vals[0, 0, r].item()
                val_32 = int(raw_val * 2147483648.0 + 2147483648.0) & 0xFFFFFFFF
                reg_writes[r] = val_32

        # 3. Memory writes
        mem_writes: list[list[int]] = []
        mem_flags = torch.sigmoid(pred.mem_has_write[0, 0])
        for slot in range(2):
            if mem_flags[slot].item() > 0.5:
                addr = int(pred.mem_addrs[0, 0, slot].item() * 2147483648.0 + 2147483648.0) & 0xFFFFFFFF
                val = int(pred.mem_vals[0, 0, slot].item() * 2147483648.0 + 2147483648.0) & 0xFFFFFFFF
                sz_idx = int(pred.mem_sizes[0, 0, slot].argmax().item())
                sz = [1, 2, 4][sz_idx]
                mem_writes.append([addr, val, sz])

        return pred_next_pc, reg_writes, mem_writes, new_states, new_kvs
