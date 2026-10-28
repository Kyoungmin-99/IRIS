"""라벨이 붙은 타투 합성 사진(사진 한 장 전체): 맨살 팔·손 사진에 흰 바탕 도안을 앱 굽기와 같은 투과율 방식으로 입힌다.

  python composite.py --manifests '/tmp/claude-0/pc/dot-candidates/dot-*/manifest.json' --designs /tmp/claude-0/hfdata/tv0 --out /tmp/claude-0/tatcomp --n 200
출력: images/{id}.jpg, masks/{id}.png(잉크 알파), labels.jsonl
  style: black(음영 없는 먹) · blackgrey(진한 윤곽 + 단계 음영) · color(color_kind: red · point · full)
  --mode pairs: 같은 도안·같은 자리·같은 진하기로 blackgrey / black 짝(pair 칸이 같음 — 차이는 음영 하나뿐)
  --mode pointpairs: 같은 도안·자리로 포인트컬러(먹 위 일부만 색) / 색 뺀 판(흑백, style 은 음영에 따라 blackgrey·black) 짝
  --sizes small,medium: 크기 고르기(AI 채팅: 미니는 효과 없어 뺌)
  --mode color: 색 도안만, --strict: 피부 밖으로 5% 넘게 잘린 것(옷·배경·손가락 위) 버림, --sam: SAM 2.1 로 오린 팔·손 안에만
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
    """도안 그림 → 잘라 낸 (알파, 잉크 색). 흰 여백은 잘라 낸다.
    자르는 범위는 스타일과 상관없이 원본 도안의 잉크 범위(짝끼리 같은 자리에 오게: 전에는 스타일마다 잘라 블랙 판이 조금 밀렸다)"""
    rgb = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255
    sat = rgb.max(2) - rgb.min(2)
    base = np.clip(np.maximum(1 - rgb @ np.array([0.299, 0.587, 0.114], np.float32), sat) * 1.15 - 0.05, 0, 1)
    ys, xs = np.nonzero(base > 0.1)
    if len(ys) < 50:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
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
    alpha, rgb = alpha[y0:y1, x0:x1], rgb[y0:y1, x0:x1]
    if (alpha > 0.1).sum() < 50:
        return None
    if (alpha > 0.5).mean() > 0.45 or ((alpha > 0.15) & (alpha < 0.5)).mean() > 0.35:
        return None  # 꽉 찬 원·네모 바탕이나 넓고 밋밋한 회색 면 도안은 스티커·패치처럼 보인다
    if style == "blackgrey":
        # 음영 있는 블랙앤그레이: 진한 윤곽(알파 0.7 넘음)과 단계 음영(0.2~0.6)이 둘 다 충분해야(전체가 옅은 회색이면 '옅은 블랙'을 가르친다)
        ink = max(1, int((alpha > 0.1).sum()))
        if (alpha > 0.7).sum() / ink < 0.12 or ((alpha > 0.2) & (alpha < 0.6)).sum() / ink < 0.2:
            return None
    color = np.where(alpha[..., None] > 1e-3, np.clip((rgb - (1 - alpha[..., None])) / np.maximum(alpha[..., None], 1e-3), 0, 1), 0)
    return alpha.astype(np.float32), color.astype(np.float32)


_SAM: tuple | None = None


def sam_arm(bg: np.ndarray, sm: np.ndarray, cache: Path, key: str) -> np.ndarray:
    """SAM 2.1 tiny(Apache-2.0)로 팔·손 영역을 오린다(배경 사진마다 한 번, cache 에 저장).
    피부색 판정만으로는 베이지 벽·바닥·나무 책상·손가락 사이 배경까지 피부로 잡혀 도안이 그 위에 걸쳤다"""
    global _SAM
    f = cache / f"{key}.png"
    if f.exists():
        m = np.asarray(Image.open(f)) > 127
        if m.shape == sm.shape:
            return m
    import torch
    from transformers import Sam2Model, Sam2Processor
    torch.set_num_threads(int(__import__("os").environ.get("THREADS", "1")))  # 학습·라벨러와 CPU 를 나눠 쓴다
    if _SAM is None:
        _SAM = (Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny"), Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").eval())
    P, M = _SAM
    skin = (sm > 0.5).astype(np.uint8)
    if skin.sum() < 100:
        return np.zeros_like(skin, bool)
    # 피부색 픽셀 8곳을 눌러 SAM 후보(곳마다 3개) → 안쪽 질감 조각을 SigLIP 으로 보아 '사람 피부'(0.8 넘음) 중 가장 넓은 후보.
    # 피부색 판정은 나무 책상도 피부로 잡아(사진의 85%) 가장 큰 덩어리 한 곳만 누르면 책상을 팔로 오렸고,
    # 오린 모양 전체를 보이면 SigLIP 이 팔 모양 구멍에 속아 책상도 '피부'(0.98)라 했다
    ys, xs = np.nonzero(skin)
    rng = np.random.default_rng(int(key[:8], 16))
    idx = rng.choice(len(ys), min(8, len(ys)), replace=False)
    pts = [[[int(xs[i]), int(ys[i])]] for i in idx]
    with torch.no_grad():
        inp = P(images=Image.fromarray(bg), input_points=[pts], input_labels=[[[1]] * len(pts)], return_tensors="pt")
        o = M(**inp, multimask_output=True)
    cands = [m for ms in P.post_process_masks(o.pred_masks, inp["original_sizes"])[0].numpy() > 0.5 for m in ms]
    ok = [m for m in cands if m.mean() > 0.03 and skin_prob(bg, m) >= 0.8]
    m = max(ok, key=lambda q: q.sum()) if ok else np.zeros(sm.shape, bool)  # 피부가 없으면 이 배경은 안 씀
    cache.mkdir(parents=True, exist_ok=True)
    Image.fromarray(m.astype(np.uint8) * 255).save(f)
    return m


_SIG: tuple | None = None
SKIN_LABELS = ["human skin, an arm or a hand", "a wooden table or floor", "fabric or clothing", "a wall", "furniture or an object"]


def skin_prob(bg: np.ndarray, m: np.ndarray) -> float:
    """마스크 안쪽 정사각 조각 4개(짧은 변의 12%)가 사람 피부일 확률 평균 — SigLIP2 base(Apache-2.0). 모양이 아닌 질감만 본다"""
    global _SIG
    import torch
    from transformers import AutoModel, AutoProcessor
    if _SIG is None:
        P = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
        M = AutoModel.from_pretrained("google/siglip2-base-patch16-224").eval()
        with torch.no_grad():
            t = M.get_text_features(**P(text=[f"a photo of {x}" for x in SKIN_LABELS], padding="max_length", max_length=64, return_tensors="pt"))
        t = getattr(t, "pooler_output", t)
        _SIG = (P, M, t / t.norm(dim=-1, keepdim=True))
    P, M, T = _SIG
    k = int(min(m.shape) * 0.12)
    er = cv2.erode(m.astype(np.uint8), np.ones((k, k), np.uint8))
    er[:k], er[-k:], er[:, :k], er[:, -k:] = 0, 0, 0, 0
    ys, xs = np.nonzero(er)
    if len(ys) < 10:
        return 0.0
    idx = np.random.default_rng(0).choice(len(ys), min(4, len(ys)), replace=False)
    crops = [Image.fromarray(bg[ys[i] - k // 2:ys[i] + k // 2, xs[i] - k // 2:xs[i] + k // 2]) for i in idx]
    with torch.no_grad():
        e = M.get_image_features(**P(images=crops, return_tensors="pt"))
    e = getattr(e, "pooler_output", e)
    return float(torch.softmax((e / e.norm(dim=-1, keepdim=True)) @ T.T * 100, -1)[:, 0].mean())


def color_kind(alpha: np.ndarray, color: np.ndarray) -> str | None:
    """색 도안 세분: red(잉크 색 대부분이 빨강) · point(먹 위 일부만 색, 색 잉크 30% 미만) · full(풀컬러)"""
    ink = alpha > 0.3
    if ink.sum() < 50:
        return None
    c = color[ink]
    sat = c.max(1) - c.min(1)
    col = sat > 0.25
    frac = col.mean()
    if frac < 0.08:
        return None  # 사실상 먹
    hsv = cv2.cvtColor((c[col][None] * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)[0]
    red = ((hsv[:, 0] < 8) | (hsv[:, 0] > 170)).mean()
    if frac >= 0.3 and red >= 0.7:
        return "red"
    return "point" if frac < 0.3 else "full"


def place(bg: np.ndarray, sm: np.ndarray, alpha: np.ndarray, color: np.ndarray, frac: float, geo: dict | None = None, strict: bool = False):
    """도안을 피부에 입힌다. geo 를 넘기면 같은 자리·각도·진하기로(블랙/블랙앤그레이 짝). 돌려주는 값: 그림, 알파, 상자, 자리(geo)
    strict: 피부 밖으로 5% 넘게 잘리면 버린다(옷·배경·손가락 위에 걸친 것)"""
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
    ang = geo["ang"] if geo else float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))) - 90 + random.uniform(-20, 20)
    s = geo["s"] if geo else np.sqrt(frac * H * W / (alpha.shape[0] * alpha.shape[1]))
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
    if geo:
        x0, y0, strength = geo["x0"], geo["y0"], geo["strength"]
    else:
        j = random.randrange(len(cy))
        x0, y0, strength = int(np.clip(cx[j] - nw // 2, 0, W - nw)), int(np.clip(cy[j] - nh // 2, 0, H - nh)), random.uniform(0.75, 0.95)
    A = np.zeros((H, W), np.float32)
    C = np.zeros((H, W, 3), np.float32)
    A[y0:y0 + nh, x0:x0 + nw] = a
    C[y0:y0 + nh, x0:x0 + nw] = c
    # 잉크는 피부에만, 가장자리 살짝 번짐, 진하기 무작위
    # (피부색 판정만 쓰면 피부색 벽·바닥까지 칠해져 가장 큰 피부 덩어리 안으로 제한)
    inside = cv2.GaussianBlur(comp.astype(np.float32), (0, 0), 1.5)
    A0 = cv2.GaussianBlur(A, (0, 0), 0.7) * strength
    A = A0 * np.clip((sm - 0.3) / 0.4, 0, 1) * inside
    if A.sum() < A0.sum() * (0.95 if strict else 0.75):
        return None  # 도안이 피부 밖으로 나감
    img = bg.astype(np.float32) / 255
    tint = geo["tint"] if geo else [random.uniform(0.08, 0.2), random.uniform(0.1, 0.22), random.uniform(0.12, 0.27)]
    if C.max() < 0.05:
        T = np.broadcast_to(np.array(tint, np.float32), img.shape)
    else:
        T = np.where(C.max(2, keepdims=True) < 0.05, np.array([0.12, 0.14, 0.17], np.float32), C * 0.9)
    L = img @ np.array([0.299, 0.587, 0.114], np.float32)
    hl = np.clip(L - cv2.GaussianBlur(L, (0, 0), max(2, min(H, W) * 0.01)), 0, 1)[..., None]
    out = img * (1 - A[..., None]) + (img * T + hl * 0.7) * A[..., None]
    ys, xs = np.nonzero(A > 0.1)
    if len(ys) < 30:
        return None
    g = {"ang": ang, "s": s, "x0": x0, "y0": y0, "strength": strength, "tint": tint}
    return (np.clip(out, 0, 1) * 255).astype(np.uint8), A, [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", required=True, help="dot-candidates manifest.json 글롭")
    ap.add_argument("--designs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--maxside", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default="", help="배경에서 뺄 사진 목록(시험 전용 맨살 팔 등, 한 줄에 경로 하나)")
    ap.add_argument("--mode", default="mixed", choices=["mixed", "pairs", "color", "pointpairs"],
                    help="mixed: 스타일 무작위 · pairs: 같은 도안·자리의 black/blackgrey 짝 · color: 색 도안만(color_kind: red/point/full)")
    ap.add_argument("--strict", action="store_true", help="피부 밖으로 5% 넘게 잘린 합성은 버림")
    ap.add_argument("--sizes", default=",".join(SIZES), help="쓸 크기(쉼표로)")
    ap.add_argument("--sam", action="store_true", help="SAM 2.1 로 팔·손 영역을 오려 그 안에만(배경 사진마다 한 번, out/_arm 에 캐시)")
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
        size = random.choice(a.sizes.split(","))
        dp = random.choice(designs)
        styles = {"pairs": ["blackgrey", "black"], "color": ["color"], "pointpairs": ["color", "gray"]}.get(a.mode, [random.choice(["black", "blackgrey", "color"])])
        ds = [design(dp, st) for st in styles]
        if any(d is None for d in ds):
            continue
        ck = color_kind(*ds[0]) if styles[0] == "color" else None
        if styles[0] == "color" and ck is None or a.mode == "pointpairs" and ck != "point":
            continue
        if "gray" in styles:  # 색 뺀 판의 이름: 음영 조건을 채우면 blackgrey, 아니면 black
            styles = [st if st != "gray" else ("blackgrey" if design(dp, "blackgrey") else "black") for st in styles]
        im = Image.open(bp).convert("RGB")
        sc = min(1, a.maxside / max(im.size))
        im = im.resize((int(im.width * sc), int(im.height * sc)), Image.BILINEAR)
        bg = np.asarray(im)
        sm = skin_mask(bg)
        if a.sam:
            sm = sm * cv2.GaussianBlur(sam_arm(bg, sm, out / "_arm", hashlib.sha1(f"{bp}:{bg.shape}".encode()).hexdigest()[:16]).astype(np.float32), (0, 0), 1.5)
        frac = random.uniform(*SIZES[size])
        outs, geo = [], None
        for d in ds:
            r = place(bg, sm, *d, frac, geo=geo, strict=a.strict)
            if r is None:
                break
            outs.append(r)
            geo = r[3]
        if len(outs) < len(ds):
            continue
        pair = f"{k:05d}" if len(ds) > 1 else None
        for st, (img, A, bbox, g) in zip(styles, outs):
            name = f"{k:05d}"
            Image.fromarray(img).save(out / "images" / f"{name}.jpg", quality=92)
            Image.fromarray((A * 255).astype(np.uint8)).save(out / "masks" / f"{name}.png")
            labels.write(json.dumps({
                "file": f"images/{name}.jpg", "mask": f"masks/{name}.png", "style": st, **({"color_kind": ck} if ck and st == "color" else {}), **({"pair": pair} if pair else {}),
                "size": size, "area_frac": round((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (img.shape[0] * img.shape[1]), 4), "bbox": bbox,
                "placement": cat, "rotation": round(g["ang"], 1), "background_sha256": hashlib.sha256(bp.read_bytes()).hexdigest(), "design": dp.name,
            }, ensure_ascii=False) + "\n")
            labels.flush()
            k += 1
    print(f"합성 {k}장", flush=True)


if __name__ == "__main__":
    main()
