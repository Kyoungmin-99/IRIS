"""선생님 정답(teacher.py 출력)을 학습용 256 조각으로 자른다(train.py 의 images/masks/weights 형식).

  python crops.py --lab /tmp/claude-0/teach/lab --out /tmp/claude-0/tatreal
  python crops.py --comp /tmp/claude-0/tatcomp --out /tmp/claude-0/tatcompc   (composite.py 합성 사진: 가중치 전부 1)
  python crops.py --neg list.txt --out /tmp/claude-0/tatneg   (타투 없는 맨살 사진 목록: 잉크 없음 조각, 피부색이 40% 넘는 조각만)
시험 전용(split=eval) 사진은 쓰지 않는다. 타투 사진은 잉크 둘레 조각을 더 많이, 맨살 팔·손 사진은 무작위 조각.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from labels import drop_dense

S = 256


def skin_frac(im: Image.Image) -> float:
    from synth import skin_mask
    return float((skin_mask(np.asarray(im)) > 0.5).mean())


def sources(a: argparse.Namespace):
    """(이름, 그림, 잉크 정답, 가중치) — 선생님 정답(split=train 만) 또는 합성 사진"""
    if a.lab:
        lab = Path(a.lab)
        for jf in sorted(lab.glob("*.json")):
            m = json.loads(jf.read_text())
            if "error" in m or m.get("split") != "train":
                continue
            k = jf.stem
            ink, w = Image.open(lab / f"{k}_ink.png"), Image.open(lab / f"{k}_w.png")
            w = Image.fromarray(drop_dense(m, np.asarray(ink) > 128, np.asarray(w) > 128).astype(np.uint8) * 255)
            yield k, Image.open(lab / f"{k}.jpg").convert("RGB"), ink, w
    if a.comp:
        comp = Path(a.comp)
        for line in (comp / "labels.jsonl").read_text().splitlines():
            r = json.loads(line)
            im = Image.open(comp / r["file"]).convert("RGB")
            yield "c" + Path(r["file"]).stem, im, Image.open(comp / r["mask"]), Image.new("L", im.size, 255)
    if a.neg:
        import hashlib
        for line in Path(a.neg).read_text().splitlines():
            if not line.strip():
                continue
            im = Image.open(line.strip()).convert("RGB")
            sc = min(1, 1024 / max(im.size))
            im = im.resize((int(im.width * sc), int(im.height * sc)), Image.BILINEAR)
            yield "n" + hashlib.sha1(line.encode()).hexdigest()[:12], im, Image.new("L", im.size, 0), Image.new("L", im.size, 255)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", default="")
    ap.add_argument("--comp", default="")
    ap.add_argument("--neg", default="", help="타투 없는 사진 목록(한 줄에 경로 하나)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    out = Path(a.out)
    for d in ("images", "masks", "weights"):
        (out / d).mkdir(parents=True, exist_ok=True)
    n = skipped = 0
    for k, im, ink, wgt in sources(a):
        has_ink = np.asarray(ink).max() > 128
        for c in range(6 if has_ink else 3):
            # 크기 흔들기: 앱은 긴 변 768 안팎에서 돈다
            s = random.uniform(0.45, 1.0)
            W, H = max(S, int(im.width * s)), max(S, int(im.height * s))
            ims, inks, ws = im.resize((W, H), Image.BILINEAR), ink.resize((W, H), Image.BILINEAR), wgt.resize((W, H), Image.NEAREST)
            ia = np.asarray(inks)
            ys, xs = np.nonzero(ia > 128) if has_ink and c < 4 else ((), ())
            if len(ys):  # 줄이면 가는 선이 128 아래로 옅어질 수 있다
                j = random.randrange(len(ys))
                cx, cy = xs[j] + random.randint(-S // 4, S // 4), ys[j] + random.randint(-S // 4, S // 4)
            else:
                cx, cy = random.randint(0, W), random.randint(0, H)
            x0, y0 = min(max(0, cx - S // 2), W - S), min(max(0, cy - S // 2), H - S)
            box = (x0, y0, x0 + S, y0 + S)
            wc = ws.crop(box)
            if (np.asarray(wc) > 128).mean() < 0.5:  # 절반 넘게 '모름'이면 버림
                skipped += 1
                continue
            if k.startswith("n") and skin_frac(ims.crop(box)) < 0.4:  # 맨살 조각은 피부가 보이는 곳만
                skipped += 1
                continue
            name = f"{k}_{c}"
            ims.crop(box).save(out / "images" / f"{name}.jpg", quality=92)
            inks.crop(box).save(out / "masks" / f"{name}.png")
            wc.save(out / "weights" / f"{name}.png")
            n += 1
    print(f"조각 {n}개 (버림 {skipped})")


if __name__ == "__main__":
    main()
