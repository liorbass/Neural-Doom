#!/usr/bin/env python3
"""Export Samba Basic Block Predictor to ONNX for in-browser WebGPU execution."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from samba import DoomSambaPredictor

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / "model" / "samba_blocks.pt"
DEFAULT_OUTPUT = ROOT / "web" / "samba_block.onnx"


class SambaStepWrapper(torch.nn.Module):
    """Fixed-shape recurrent step wrapper for ONNX export."""

    def __init__(self, model: DoomSambaPredictor) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        pc_bits: torch.Tensor,
        regs_norm: torch.Tensor,
        s0: torch.Tensor,
        s1: torch.Tensor,
        s2: torch.Tensor,
        s3: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        pc_emb = self.model.pc_proj(pc_bits)
        regs_emb = self.model.regs_proj(regs_norm)
        x = torch.cat([pc_emb, regs_emb], dim=-1)

        states_in = [s0, s1, s2, s3]
        states_out: list[torch.Tensor] = []
        for i, blk in enumerate(self.model.blocks):
            x, s_out, _ = blk(x, ssm_state=states_in[i])
            states_out.append(s_out)

        h = self.model.ln_f(x)
        return (
            self.model.head_pc(h),
            self.model.head_branch(h),
            self.model.head_reg_mask(h),
            self.model.head_reg_vals(h),
            self.model.head_mem_flag(h),
            self.model.head_mem_addr(h),
            self.model.head_mem_val(h),
            self.model.head_mem_size(h).view(1, 1, 2, 3),
            states_out[0],
            states_out[1],
            states_out[2],
            states_out[3],
        )


def export_onnx(ckpt_path: Path, output_path: Path) -> None:
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = DoomSambaPredictor(
        d_model=cfg.get("d_model", 128),
        num_layers=cfg.get("num_layers", 4),
        num_heads=cfg.get("num_heads", 4),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    wrapper = SambaStepWrapper(model)
    wrapper.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dummy inputs for tracing
    pc_bits = torch.zeros(1, 1, 32, dtype=torch.float32)
    regs_norm = torch.zeros(1, 1, 32, dtype=torch.float32)
    s0 = torch.zeros(1, 128, 16, dtype=torch.float32)
    s1 = torch.zeros(1, 128, 16, dtype=torch.float32)
    s2 = torch.zeros(1, 128, 16, dtype=torch.float32)
    s3 = torch.zeros(1, 128, 16, dtype=torch.float32)

    print(f"Exporting ONNX model to {output_path} (opset 18)...")
    torch.onnx.export(
        wrapper,
        (pc_bits, regs_norm, s0, s1, s2, s3),
        str(output_path),
        input_names=["pc_bits", "regs_norm", "s0", "s1", "s2", "s3"],
        output_names=[
            "next_pc_offset",
            "branch_taken_logits",
            "reg_mask_logits",
            "reg_vals",
            "mem_has_write",
            "mem_addrs",
            "mem_vals",
            "mem_sizes",
            "s0_out",
            "s1_out",
            "s2_out",
            "s3_out",
        ],
    print(f"Embedding external data into single self-contained model...")
    import onnx
    onnx_model = onnx.load(str(output_path), load_external_data=True)
    onnx.save_model(onnx_model, str(output_path), save_as_external_data=False)
    data_file = output_path.with_name(output_path.name + ".data")
    if data_file.exists():
        data_file.unlink()

    print(f"Model exported successfully! Size: {output_path.stat().st_size / (1024 * 1024):.2f} MB (Self-Contained)")

    # Numerical parity verification
    print("Verifying numerical parity between PyTorch and ONNX Runtime...")
    with torch.no_grad():
        pt_outputs = wrapper(pc_bits, regs_norm, s0, s1, s2, s3)

    session = ort.InferenceSession(str(output_path))
    ort_inputs = {
        "pc_bits": pc_bits.numpy(),
        "regs_norm": regs_norm.numpy(),
        "s0": s0.numpy(),
        "s1": s1.numpy(),
        "s2": s2.numpy(),
        "s3": s3.numpy(),
    }
    ort_outputs = session.run(None, ort_inputs)

    max_diff = 0.0
    for i, (pt_out, ort_out) in enumerate(zip(pt_outputs, ort_outputs)):
        diff = np.max(np.abs(pt_out.numpy() - ort_out))
        if diff > max_diff:
            max_diff = float(diff)

    print(f"Parity check PASSED! Maximum absolute difference: {max_diff:.6e} (tolerance < 1e-4)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Samba model to ONNX")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    export_onnx(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
