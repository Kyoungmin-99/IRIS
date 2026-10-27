"""라벨이 붙은 타투 합성 사진(사진 한 장 전체): 맨살 팔·손 사진에 흰 바탕 도안을 앱 굽기와 같은 투과율 방식으로 입힌다.

  python composite.py --manifests '/tmp/claude-0/pc/dot-candidates/dot-*/manifest.json' --designs /tmp/claude-0/hfdata/tv0 --out /tmp/claude-0/tatcomp --n 200
출력: images/{id}.jpg, masks/{id}.png(잉크 알파), labels.jsonl
  style: black(음영 없는 먹) · blackgrey(회색 음영) · color
  size: mini(사진의 0.5~2%) · small(2~5%) · medium(5~12%) — 도안 상자 넓이 / 사진 넓이
  placement: 배경 사진의 category(forearm_outer 등) 그대로, bbox: [x0, y0, x1, y1] 픽셀
배경은 has_tattoo 가 false 인 사진만. AI 채팅 스타일 분석 학습·IRIS 잉크 모델 학습에 같이 쓴다.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from synth import skin_mask

SIZES = {"mini": (0.005, 0.02), "small": (0.02, 0.05), "medium": (0.05, 0.12)}


def backgrounds(pattern: str) -> list[tuple[Path, str]]:
    out = []
    for mf in sorted(glob.glob(pattern)):
        d = json.loads(Path(mf).read_text())
        for it in d.get("items", []) if isinstance(d, dict) else []:
            p = Path(mf).parent / it["file"]
            if it.get("has_tattoo") is False and p.exists():
                out.append((p, it.get("category", "")))
    return out


def design(path: Path, style: str) -> tuple[np.ndarray, np.ndarray] | None:
    """도안 그림 → 잘라 낸 (알파, 잉크 색). 흰 여백은 잘라 낸다"""
    rgb = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255
    sat = rgb.max(2) - rgb.min(2)
    if style == "color":
        s5 = sat[sat > 0.05]
        if len(s5) < 100 or np.percentile(s5, 90) < 0.25:
            return None  # 색 도안이 아님
    if style != "color":
        rgb = np.repeat((rgb @ np.array([0.299, 0.587, 0.114], np.float32))[..., None], 3, 2)
        sat = np.zeros_like(sat)
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    alpha = np.clip(np.maximum(1 - lum, sat) * 1.15 - 0.05, 0, 1)
    if style == "black":
        alpha = np.clip((alpha - 0.35) / 0.1, 0, 1)  # 회색 음영 없이 먹만
    ys, xs = np.nonzero(alpha > 0.1)
    if len(ys) < 50:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    alpha, rgb = alpha[y0:y1, x0:x1], rgb[y0:y1, x0:x1]
    if (alpha > 0.5).mean() > 0.45 or ((alpha > 0.15) & (alpha < 0.5)).mean() > 0.35:
        return None  # 꽉 찬 원·네모 바탕이나 넓고 밋밋한 회색 면 도안은 스티커·패치처럼 보인다
    color = np.where(alpha[..., None] > 1e-3, np.clip((rgb - (1 - alpha[..., None])) / np.maximum(alpha[..., None], 1e-3), 0, 1), 0)
    return alpha.astype(np.float32), color.astype(np.float32)


def place(bg: np.ndarray, sm: np.ndarray, alpha: np.ndarray, color: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray, list[int], float] | None:
    H, W = sm.shape
    skin = (sm > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(skin)
    if n < 2:
        return None
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    comp = (lab == big).astype(np.uint8)
    if comp.sum() < H * W * 0.05:
        return None
    # 팔 방향(주축)에 도안 세로를 맞추고 ±20° 흔들기
    ys, xs = np.nonzero(comp)
    cov = np.cov(np.stack([xs, ys]).astype(np.float32))
    ev, evec = np.linalg.eigh(cov)
    ang = float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))) - 90 + random.uniform(-20, 20)
    s = np.sqrt(frac * H * W / (alpha.shape[0] * alpha.shape[1]))
    a = cv2.resize(alpha, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    c = cv2.resize(color, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if min(a.shape) < 6:
        return None
    M = cv2.getRotationMatrix2D((a.shape[1] / 2, a.shape[0] / 2), -ang, 1)
    cs, sn = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(a.shape[0] * sn + a.shape[1] * cs), int(a.shape[0] * cs + a.shape[1] * sn)
    M[0, 2] += nw / 2 - a.shape[1] / 2
    M[1, 2] += nh / 2 - a.shape[0] / 2
    a = cv2.warpAffine(a, M, (nw, nh))
    c = cv2.warpAffine(c, M, (nw, nh))
    if nw >= W or nh >= H:
        return None
    # 도안 반지름만큼 피부 안쪽인 곳에 가운데를 둔다
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    r = 0.35 * min(nw, nh)
    cy, cx = np.nonzero(dist > r)
    if not len(cy):
        return None
    j = random.randrange(len(cy))
    x0, y0 = int(np.clip(cx[j] - nw // 2, 0, W - nw)), int(np.clip(cy[j] - nh // 2, 0, H - nh))
    A = np.zeros((H, W), np.float32)
    C = np.zeros((H, W, 3), np.float32)
    A[y0:y0 + nh, x0:x0 + nw] = a
    C[y0:y0 + nh, x0:x0 + nw] = c
    # 잉크는 피부에만, 가장자리 살짝 번짐, 진하기 무작위
    # (피부색 판정만 쓰면 피부색 벽·바닥까지 칠해져 가장 큰 피부 덩어리 안으로 제한)
    inside = cv2.GaussianBlur(comp.astype(np.float32), (0, 0), 1.5)
    A = cv2.GaussianBlur(A, (0, 0), 0.7) * np.clip((sm - 0.3) / 0.4, 0, 1) * inside * random.uniform(0.75, 0.95)
    if A.sum() < 0.85 * a.sum() * 0.75:
        return None  # 도안이 피부 밖으로 많이 나감
    img = bg.astype(np.float32) / 255
    if C.max() < 0.05:
        T = np.broadcast_to(np.array([random.uniform(0.08, 0.2), random.uniform(0.1, 0.22), random.uniform(0.12, 0.27)], np.float32), img.shape)
    else:
        T = np.where(C.max(2, keepdims=True) < 0.05, np.array([0.12, 0.14, 0.17], np.float32), C * 0.9)
    L = img @ np.array([0.299, 0.587, 0.114], np.float32)
    hl = np.clip(L - cv2.GaussianBlur(L, (0, 0), max(2, min(H, W) * 0.01)), 0, 1)[..., None]
    out = img * (1 - A[..., None]) + (img * T + hl * 0.7) * A[..., None]
    ys, xs = np.nonzero(A > 0.1)
    if len(ys) < 30:
        return None
    return (np.clip(out, 0, 1) * 255).astype(np.uint8), A, [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], ang


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", required=True, help="dot-candidates manifest.json 글롭")
    ap.add_argument("--designs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--maxside", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default="", help="배경에서 뺄 사진 목록(시험 전용 맨살 팔 등, 한 줄에 경로 하나)")
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    bgs = backgrounds(a.manifests)
    if a.exclude:
        ex = {l.strip() for l in open(a.exclude) if l.strip()}
        bgs = [b for b in bgs if str(b[0]) not in ex]
    designs = sorted(Path(a.designs).glob("*"))
    print(f"배경 {len(bgs)}장, 도안 {len(designs)}장", flush=True)
    labels = open(out / "labels.jsonl", "a", encoding="utf-8")
    k = len(list((out / "images").glob("*.jpg")))
    if k:  # 이어 만들기: 난수를 다르게(앞에 만든 것과 겹치지 않게)
        random.seed(a.seed + k)
        np.random.seed(a.seed + k)
    tries = 0
    while k < a.n and tries < a.n * 30:
        tries += 1
        bp, cat = random.choice(bgs)
        style = random.choice(["black", "blackgrey", "color"])
        size = random.choice(list(SIZES))
        dp = random.choice(designs)
        d = design(dp, style)
        if d is None:
            continue
        im = Image.open(bp).convert("RGB")
        s = min(1, a.maxside / max(im.size))
        im = im.resize((int(im.width * s), int(im.height * s)), Image.BILINEAR)
        bg = np.asarray(im)
        r = place(bg, skin_mask(bg), *d, random.uniform(*SIZES[size]))
        if r is None:
            continue
        img, A, bbox, ang = r
        name = f"{k:05d}"
        Image.fromarray(img).save(out / "images" / f"{name}.jpg", quality=92)
        Image.fromarray((A * 255).astype(np.uint8)).save(out / "masks" / f"{name}.png")
        labels.write(json.dumps({
            "file": f"images/{name}.jpg", "mask": f"masks/{name}.png", "style": style, "size": size,
            "area_frac": round((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (img.shape[0] * img.shape[1]), 4), "bbox": bbox,
            "placement": cat, "rotation": round(ang, 1), "background_sha256": hashlib.sha256(bp.read_bytes()).hexdigest(), "design": dp.name,
        }, ensure_ascii=False) + "\n")
        labels.flush()
        k += 1
    print(f"합성 {k}장", flush=True)


if __name__ == "__main__":
    main()
