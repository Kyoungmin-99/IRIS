"""실제 사진 시험: 선생님 정답(teacher.py, split=eval)으로 잉크 모델의 픽셀 점수를 잰다.

  python eval_real.py --lab /tmp/claude-0/teach/lab ep4=tattoo-ink.onnx new=out/tattoo-ink.onnx
- IoU: 타투 사진에서 가중치 ≥ 0.5 인 픽셀만(모름 제외), 잉크 > 0.5 기준
- 상자 밖 오검출: 타투 상자 밖 맨살·배경에서 모델이 잉크라고 한 픽셀 비율(%)
- 맨살 팔 오검출: 타투 없는 팔 사진 중 잉크가 0.2% 넘게 나온 사진 비율
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from labels import drop_dense


def prob(sess: ort.InferenceSession, im: Image.Image) -> np.ndarray:
    W, H = im.size
    pw, ph = (W + 15) // 16 * 16, (H + 15) // 16 * 16  # 5단계 모델은 16의 배수
    x = np.zeros((1, 3, ph, pw), np.float32)
    x[0, :, :H, :W] = np.asarray(im, np.float32).transpose(2, 0, 1) / 255
    y = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0, :H, :W]
    return 1 / (1 + np.exp(-y))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", required=True)
    ap.add_argument("models", nargs="+", help="이름=경로.onnx")
    a = ap.parse_args()
    lab = Path(a.lab)
    items = []
    for jf in sorted(lab.glob("*.json")):
        m = json.loads(jf.read_text())
        if "error" not in m and m.get("split") == "eval":
            items.append((jf.stem, m))
    res = {}
    for spec in a.models:
        name, path = spec.split("=", 1)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inter = union = 0.0
        ious, out_fp, arm_fa = [], [], []
        for k, m in items:
            im = Image.open(lab / f"{k}.jpg").convert("RGB")
            p = prob(sess, im) > 0.5
            t = np.asarray(Image.open(lab / f"{k}_ink.png")) > 128
            w = drop_dense(m, t, np.asarray(Image.open(lab / f"{k}_w.png")) > 128)
            inbox = np.zeros_like(t)
            for r in m.get("regions", []):
                x0, y0, x1, y1 = map(int, m["boxes"][r["box"]])
                inbox[y0:y1 + 1, x0:x1 + 1] = True
            if m["kind"] == "arm":
                arm_fa.append(float((p & w).mean()) > 0.002)
            elif (t & w).any():  # 정답 잉크가 모두 '모름'(가중 0)인 사진은 채점할 것이 없다(0점으로 세면 사진 평균이 거짓으로 낮아짐)
                i, u = float((p & t & w).sum()), float(((p | t) & w).sum())
                inter, union = inter + i, union + u
                ious.append(i / max(u, 1))
                o = w & ~inbox
                out_fp.append(float((p & o).sum()) / max(1, float(o.sum())))
        r = {
            "tattoo_n": len(ious), "IoU(전체)": round(inter / max(union, 1), 3), "IoU(사진 평균)": round(float(np.mean(ious)), 3) if ious else None,
            "상자 밖 오검출%": round(100 * float(np.mean(out_fp)), 3) if out_fp else None,
            "arm_n": len(arm_fa), "맨살 팔 오검출": round(float(np.mean(arm_fa)), 3) if arm_fa else None,
        }
        res[name] = r
        print(name, json.dumps(r, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
