"""타투 잉크 분할 학습 자료 합성(정답을 아는 합성 사진).

공개된 타투 분할 모델·상업 이용 가능한 데이터가 없어(HDA 합성 타투 DB 는 상업 이용 금지) 직접 만든다:
타투가 없는 사진(메이크업·헤어·네일 사진)의 피부 부분에 무작위 도안(가는 선·원·점묘·블랙워크 면·명암·컬러·한글/영문 레터링)을
잉크 투과율로 곱해 넣고, 그 잉크 알파를 정답으로 저장한다. 얼굴의 눈썹·속눈썹·머리카락·옷 주름·손톱은 정답 0(타투 아님)으로 배운다.

  python synth.py --bg <사진 폴더들> --fonts <글꼴 폴더> --out data --n 8000
  상업용 판: --ledger <FINDE-photos>/rights/ledger.jsonl --commercial-only  (장부에서 commercial_ok 가 true 인 배경 사진만)
  실제 도안: --designs <흰 바탕 타투 도안 그림 폴더> --design-frac 0.5  (그 비율만큼 무작위 획 대신 실제 도안을 피부에 입힘, 절반은 흑백으로)
결과: data/images/00000.jpg (256×256), data/masks/00000.png (잉크 알파 0~255), data/backgrounds.txt (쓴 배경 사진 sha256·경로, 권리 검토용)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

S = 256
WORDS_KO = ["가족", "사랑", "행복", "희망", "엄마", "아빠", "우리", "오늘", "자유", "꿈", "별", "달", "봄날", "그대", "평화", "용기", "감사", "청춘", "하늘", "바다", "너와나", "괜찮아", "함께", "영원히"]
WORDS_EN = ["love", "hope", "faith", "family", "Inner Peace", "carpe diem", "forever", "breathe", "stay strong", "dream", "mom", "dad", "believe", "freedom", "amor", "1994", "XII", "07.21", "blessed", "wanderlust", "let it be", "memento mori", "only one", "Seoul"]


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y, Cr, Cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    m = ((Y > 55) & (Cr > 133) & (Cr < 178) & (Cb > 78) & (Cb < 132)).astype(np.float32)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.GaussianBlur(m, (0, 0), 3)
    return m


def bezier(p: np.ndarray, n: int = 60) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3) * p[0] + 3 * ((1 - t) ** 2) * t * p[1] + 3 * (1 - t) * (t ** 2) * p[2] + (t ** 3) * p[3]


def real_design(files: list[Path], size: int) -> tuple[np.ndarray, np.ndarray]:
    """흰 바탕 타투 도안 그림 → (알파 0~1, 잉크 색 RGB 0~1). 어두울수록·색이 짙을수록 잉크가 많다(회색 음영은 옅은 잉크).
    크기·회전·위치 무작위, 절반은 흑백(블랙앤그레이)"""
    im = Image.open(random.choice(files)).convert("RGB")
    sc = random.uniform(0.35, 1.1) * size / max(im.size)
    im = im.resize((max(8, int(im.width * sc)), max(8, int(im.height * sc))), Image.BILINEAR)
    im = im.rotate(random.uniform(-35, 35), resample=Image.BILINEAR, expand=True, fillcolor=(255, 255, 255))
    if random.random() < 0.5:
        im = im.convert("L").convert("RGB")
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(im, (random.randint(-im.width // 4, max(0, size - im.width * 3 // 4)), random.randint(-im.height // 4, max(0, size - im.height * 3 // 4))))
    rgb = np.asarray(canvas, np.float32) / 255
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    sat = rgb.max(2) - rgb.min(2)
    alpha = np.clip(np.maximum(1 - lum, sat) * 1.15 - 0.05, 0, 1)
    color = np.where(alpha[..., None] > 1e-3, np.clip((rgb - (1 - alpha[..., None])) / np.maximum(alpha[..., None], 1e-3), 0, 1), 0)
    return alpha.astype(np.float32), color.astype(np.float32)


def draw_design(fonts: list[Path], size: int) -> tuple[np.ndarray, np.ndarray]:
    """2배 크기에 그린 도안 → (알파 0~1, 잉크 색 RGB 0~1). 위치는 크롭 안 아무 곳"""
    K = size * 2
    a = Image.new("L", (K, K), 0)
    col = Image.new("RGB", (K, K), (0, 0, 0))
    da, dc = ImageDraw.Draw(a), ImageDraw.Draw(col)
    kind = random.choices(["fine", "geo", "text", "dot", "black", "shade", "color", "mix"], [22, 12, 22, 8, 8, 8, 10, 10])[0]
    cx, cy = random.uniform(0.25, 0.75) * K, random.uniform(0.25, 0.75) * K
    R = random.uniform(0.12, 0.45) * K
    ink = (0, 0, 0)
    if kind == "color" or (kind == "mix" and random.random() < 0.5):
        ink = random.choice([(200, 30, 40), (30, 70, 170), (40, 140, 70), (120, 40, 150), (230, 120, 30), (220, 80, 140)])

    def stroke(pts, w, c=None, alpha=255):
        pts = [tuple(map(float, q)) for q in pts]
        da.line(pts, fill=alpha, width=max(1, int(w)), joint="curve")
        dc.line(pts, fill=c or ink, width=max(1, int(w)), joint="curve")

    if kind in ("fine", "mix", "color"):
        for _ in range(random.randint(2, 12)):
            p = np.array([[cx + random.uniform(-R, R), cy + random.uniform(-R, R)] for _ in range(4)])
            stroke(bezier(p), random.choice([1, 2, 2, 3, 4, 6]))
    if kind in ("geo", "mix"):
        for _ in range(random.randint(1, 5)):
            r = random.uniform(0.2, 1) * R
            box = [cx - r, cy - r, cx + r, cy + r]
            w = random.choice([2, 3, 4, 6])
            if random.random() < 0.5:
                da.ellipse(box, outline=255, width=w); dc.ellipse(box, outline=ink, width=w)
            else:
                s0 = random.uniform(0, 360)
                da.arc(box, s0, s0 + random.uniform(60, 300), fill=255, width=w); dc.arc(box, s0, s0 + random.uniform(60, 300), fill=ink, width=w)
            if random.random() < 0.4:
                ang = random.uniform(0, math.pi)
                stroke([(cx - math.cos(ang) * r * 1.4, cy - math.sin(ang) * r * 1.4), (cx + math.cos(ang) * r * 1.4, cy + math.sin(ang) * r * 1.4)], w)
    if kind == "dot" or (kind == "mix" and random.random() < 0.3):
        for _ in range(random.randint(60, 500)):
            r2 = R * math.sqrt(random.random())
            t = random.uniform(0, 2 * math.pi)
            x, y, d = cx + r2 * math.cos(t), cy + r2 * math.sin(t), random.uniform(1, 4)
            da.ellipse([x - d, y - d, x + d, y + d], fill=255); dc.ellipse([x - d, y - d, x + d, y + d], fill=ink)
    if kind == "black":
        pts = [(cx + R * math.cos(t) * random.uniform(0.5, 1), cy + R * math.sin(t) * random.uniform(0.5, 1)) for t in np.linspace(0, 2 * math.pi, random.randint(3, 9), endpoint=False)]
        da.polygon(pts, fill=255); dc.polygon(pts, fill=ink)
    if kind == "text" or (kind == "mix" and random.random() < 0.4):
        f = random.choice(fonts)
        word = random.choice(WORDS_KO if ("Nanum" in f.name or "Gaegu" in f.name or "Gowun" in f.name) and random.random() < 0.8 else WORDS_EN)
        fs = int(random.uniform(0.08, 0.22) * K)
        try:
            font = ImageFont.truetype(str(f), fs)
        except OSError:
            font = ImageFont.load_default()
        ta = Image.new("L", (K, K), 0)
        ImageDraw.Draw(ta).text((cx, cy), word, font=font, fill=255, anchor="mm")
        ta = ta.rotate(random.uniform(-40, 40), resample=Image.BILINEAR, center=(cx, cy))
        a = Image.fromarray(np.maximum(np.asarray(a), np.asarray(ta)))
        tc = np.asarray(col).copy()
        tc[np.asarray(ta) > 0] = ink
        col = Image.fromarray(tc)
    alpha = np.asarray(a, np.float32) / 255
    color = np.asarray(col, np.float32) / 255
    if kind in ("shade", "mix", "color") and random.random() < 0.7:
        # 명암·색 채움: 부드러운 덩어리(투명도 낮게)
        g = np.zeros((K, K), np.float32)
        for _ in range(random.randint(1, 4)):
            x, y, r = cx + random.uniform(-R, R) * 0.6, cy + random.uniform(-R, R) * 0.6, random.uniform(0.15, 0.5) * R
            cv2.circle(g, (int(x), int(y)), int(r), 1.0, -1)
        g = cv2.GaussianBlur(g, (0, 0), R * 0.15 + 1) * random.uniform(0.25, 0.7)
        if kind == "color":
            fill = np.array(random.choice([(220, 60, 70), (60, 110, 200), (90, 170, 90), (170, 90, 190), (240, 170, 60)]), np.float32) / 255
            color = np.where((g > alpha)[..., None], fill, color)
        alpha = np.maximum(alpha, g)
    # 문신 바늘 퍼짐: 살짝 흐림
    alpha = cv2.GaussianBlur(alpha, (0, 0), random.uniform(0.6, 1.6))
    alpha = cv2.resize(alpha, (size, size), interpolation=cv2.INTER_AREA)
    color = cv2.resize(color, (size, size), interpolation=cv2.INTER_AREA)
    return np.clip(alpha, 0, 1), color


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", nargs="+", required=True)
    ap.add_argument("--fonts", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ledger", default="", help="FINDE-photos rights/ledger.jsonl (비공개 권리 장부)")
    ap.add_argument("--designs", default="", help="흰 바탕 실제 타투 도안 그림 폴더")
    ap.add_argument("--design-frac", type=float, default=0.5)
    ap.add_argument("--commercial-only", action="store_true", help="장부에서 commercial_ok 가 true 인 배경 사진만 쓴다(상업용 판)")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    files = [p for d in args.bg for p in Path(d).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    sha = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if args.commercial_only:
        if not args.ledger:
            raise SystemExit("--commercial-only 에는 --ledger 가 필요하다")
        ok = set()
        for line in Path(args.ledger).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("commercial_ok") is True:
                    ok.add(row["sha256"])
        files = [p for p in files if sha[p] in ok]
        print(f"상업용 판: 장부에서 commercial_ok 인 배경 {len(files)}장")
        if not files:
            raise SystemExit("상업 이용이 확인된 배경 사진이 없다")
    fonts = sorted(Path(args.fonts).glob("*.ttf"))
    designs = sorted(p for p in Path(args.designs).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")) if args.designs else []
    if designs:
        print(f"실제 도안 {len(designs)}장 (비율 {args.design_frac})")
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    print(f"배경 {len(files)}장, 글꼴 {len(fonts)}개")
    (out / "backgrounds.txt").write_text("".join(f"{sha[p]}\t{p}\n" for p in files), encoding="utf-8")
    cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    k = 0
    tries = 0
    while k < args.n and tries < args.n * 20:
        tries += 1
        f = random.choice(files)
        if f not in cache:
            try:
                im = Image.open(f).convert("RGB")
            except Exception:
                continue
            sc = 900 / max(im.size)
            im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))))
            rgb = np.asarray(im)
            cache[f] = (rgb, skin_mask(rgb))
            if len(cache) > 300:
                cache.pop(next(iter(cache)))
        rgb, sm = cache[f]
        H, W = sm.shape
        c = int(random.uniform(0.25, 0.7) * min(H, W))
        if c < 96:
            continue
        x0, y0 = random.randint(0, W - c), random.randint(0, H - c)
        skin = sm[y0:y0 + c, x0:x0 + c]
        if skin.mean() < 0.35:
            continue
        bg = cv2.resize(rgb[y0:y0 + c, x0:x0 + c], (S, S), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        skin = cv2.resize(skin, (S, S))
        if random.random() < 0.25:
            alpha = np.zeros((S, S), np.float32)
            img = bg
        else:
            alpha, color = real_design(designs, S) if designs and random.random() < args.design_frac else draw_design(fonts, S)
            # 잉크는 피부에만(경계는 부드럽게), 진하기 무작위(옅은 파인라인 ~ 진한 블랙)
            alpha = alpha * np.clip((skin - 0.3) / 0.4, 0, 1) * random.uniform(0.55, 1.0)
            # 투과율: 검은 잉크는 피부를 어둡게(오래된 잉크는 푸르스름하게), 색 잉크는 그 색으로
            if color.max() < 0.05:
                tint = np.array([random.uniform(0.08, 0.25), random.uniform(0.1, 0.27), random.uniform(0.12, 0.32)], np.float32)
                T = np.broadcast_to(tint, color.shape)
            else:
                T = color * random.uniform(0.7, 1.0)
            img = bg * (1 - alpha[..., None] + alpha[..., None] * T)
        # 조명·잡음·흐림·JPEG
        g = np.linspace(random.uniform(0.85, 1.1), random.uniform(0.85, 1.1), S, dtype=np.float32)
        img = img * (g[None, :, None] if random.random() < 0.5 else g[:, None, None])
        img = img + np.random.normal(0, random.uniform(0, 0.02), img.shape).astype(np.float32)
        if random.random() < 0.3:
            img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.4, 1.0))
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(u8).save(buf, "JPEG", quality=random.randint(60, 95))
        Image.open(buf).save(out / "images" / f"{k:05d}.jpg", quality=95)
        Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8)).save(out / "masks" / f"{k:05d}.png")
        k += 1
        if k % 500 == 0:
            print(k, flush=True)
    print("만듦", k)


if __name__ == "__main__":
    main()
