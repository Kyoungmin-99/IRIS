"""학습 중 저장한 가중치(.pt)를 바로 ONNX 로 내보낸다(학습이 에포크 끝까지 가지 못해도 실제 사진으로 재 보려고).

  python export.py out/last.pt out/last.onnx [--calib 0.9]
--calib p: 출력 logit 에서 log(p/(1-p)) 를 빼서, 학습 모델의 확률 p 가 앱에서 0.5 로 읽히게 한다(마지막 1×1 합성곱 bias 에 반영).
  앱 코드(extractDesign 의 smooth(0.3, 0.6, P) 등)는 그대로 두고 오검출이 적은 문턱으로 옮기는 용도.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nail"))
from train import UNet  # noqa: E402

S = 256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--calib", type=float, default=0.5)
    a = ap.parse_args()
    src, dst = a.src, a.dst
    model = UNet()
    model.load_state_dict(torch.load(src, map_location="cpu"))
    model.eval()
    if a.calib != 0.5:
        with torch.no_grad():
            model.head.bias -= math.log(a.calib / (1 - a.calib))
    torch.onnx.export(model, torch.zeros(1, 3, S, S), dst, input_names=["image"], output_names=["logit"],
                      dynamic_axes={"image": {0: "n", 2: "h", 3: "w"}, "logit": {0: "n", 2: "h", 3: "w"}}, opset_version=17, external_data=False)
    print(f"ONNX 저장: {dst}")


if __name__ == "__main__":
    main()
