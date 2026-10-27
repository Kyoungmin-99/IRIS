"""학습 중 저장한 가중치(.pt)를 바로 ONNX 로 내보낸다(학습이 에포크 끝까지 가지 못해도 실제 사진으로 재 보려고).

  python export.py out/last.pt out/last.onnx
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nail"))
from train import UNet  # noqa: E402

S = 256


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    model = UNet()
    model.load_state_dict(torch.load(src, map_location="cpu"))
    model.eval()
    torch.onnx.export(model, torch.zeros(1, 3, S, S), dst, input_names=["image"], output_names=["logit"],
                      dynamic_axes={"image": {0: "n", 2: "h", 3: "w"}, "logit": {0: "n", 2: "h", 3: "w"}}, opset_version=17, external_data=False)
    print(f"ONNX 저장: {dst}")


if __name__ == "__main__":
    main()
