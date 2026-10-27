"""네일 '색만 다른 짝' 합성: 같은 손·같은 손톱에 색만 바꾼 사진 묶음(라벨 nail.color, 같은 손은 pair 가 같음).

  python pairs.py --hands /tmp/claude-0/pc/collections/hands-2026-10-26 --out /tmp/claude-0/share2/nail_color --photos 40 --colors 4
- 손톱 찾기: Grounding DINO tiny 'fingernail.' 상자 → SAM 2.1 tiny 상자 → 마스크(모두 Apache-2.0).
  manifest 의 nail_state=bare, visible_nails 와 찾은 개수가 맞는 사진만(어색한 합성 거르기)
- 색 입히기: Lab 에서 밝기 = 목표 밝기 × 손톱의 조명 굴곡 + 잔결·반짝임(원래 손톱에서), 색도 = 목표 색. 가장자리 1px 부드럽게
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

torch.set_num_threads(int(os.environ.get("THREADS", "1")))

# taxonomy nail.color 이름 → 대표 색(sRGB)
COLORS = {
    "누드·베이지": (214, 172, 152), "밀키화이트": (238, 230, 224), "연핑크": (240, 184, 196), "코랄·피치": (238, 128, 104),
    "레드": (186, 22, 36), "버건디": (96, 18, 34), "블랙": (26, 23, 26), "네이비·블루": (32, 48, 112),
    "라벤더·퍼플": (182, 160, 214), "민트·그린": (152, 212, 188), "브라운·모카": (112, 76, 56), "더스티로즈": (190, 126, 132),
}


def nails(im: Image.Image, D, G, SP, SM) -> list[np.ndarray]:
    W, H = im.size
    with torch.no_grad():
        x = D(images=im, text="fingernail.", return_tensors="pt", size={"shortest_edge": 512, "longest_edge": 768})
        o = G(**x)
        r = D.post_process_grounded_object_detection(o, x.input_ids, threshold=0.25, text_threshold=0.2, target_sizes=[(H, W)])[0]
    boxes = [b for b in r["boxes"].tolist() if (b[2] - b[0]) * (b[3] - b[1]) < 0.03 * W * H]
    if not boxes:
        return []
    with torch.no_grad():
        si = SP(images=im, input_boxes=[boxes], return_tensors="pt")
        so = SM(**si, multimask_output=False)
    ms = SP.post_process_masks(so.pred_masks, si["original_sizes"])[0][:, 0].numpy() > 0.5
    out = []
    for b, m in zip(boxes, ms):
        x0, y0, x1, y1 = map(int, b)
        box = np.zeros_like(m)
        box[y0:y1 + 1, x0:x1 + 1] = True
        fill = (m & box).sum() / max(1, box.sum())
        if 0.3 <= fill <= 0.95 and (m & ~box).sum() < 0.1 * m.sum():
            out.append(m & box)
    # 겹치는 것(같은 손톱 두 번) 정리
    keep: list[np.ndarray] = []
    for m in sorted(out, key=lambda q: -q.sum()):
        if all((m & k).sum() < 0.3 * m.sum() for k in keep):
            keep.append(m)
    # 손톱판은 볼록: SAM 테두리의 들쭉날쭉함을 볼록 껍질로 다듬는다(칠한 덩어리처럼 보였다)
    hulls = []
    for m in keep:
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        h = np.zeros(m.shape, np.uint8)
        cv2.fillPoly(h, [cv2.convexHull(max(cs, key=cv2.contourArea))], 1)
        hulls.append(h.astype(bool))
    return hulls


def paint(rgb: np.ndarray, masks: list[np.ndarray], color: tuple[int, int, int]) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    out = lab.copy()
    A = np.zeros(rgb.shape[:2], np.float32)
    for m in masks:
        mu = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        if mu.sum() < 30:
            continue
        w = max(2.0, np.sqrt(mu.sum()) * 0.15)
        L = lab[..., 0]
        base = cv2.GaussianBlur(L * mu, (0, 0), w) / np.maximum(cv2.GaussianBlur(mu.astype(np.float32), (0, 0), w), 1e-3)
        shade = np.clip(base / max(1.0, float(base[mu].mean())), 0.6, 1.4)  # 손톱 위 조명 굴곡
        detail = L - base  # 잔결·반짝임
        newL = tgt[0] * shade ** 0.6 + detail * 0.5 + np.clip(detail, 0, None) * 0.8  # 밝은 반짝임은 더 살림(젤 광택)
        # 젤 광택 띠: 손톱 긴 축을 따라 한쪽으로 치우친 가늘고 부드러운 빛(맨 손톱엔 반짝임이 약해 없으면 무광 물감처럼 보였다)
        ys, xs = np.nonzero(mu)
        c = np.array([xs.mean(), ys.mean()])
        ev, evec = np.linalg.eigh(np.cov(np.stack([xs, ys]).astype(np.float32)))
        ax, side = evec[:, 1], evec[:, 0]
        half_w, half_l = 2 * np.sqrt(max(ev[0], 1)), 2 * np.sqrt(max(ev[1], 1))
        yy, xx = np.mgrid[0:L.shape[0], 0:L.shape[1]]
        u = ((xx - c[0]) * ax[0] + (yy - c[1]) * ax[1]) / half_l
        v = ((xx - c[0]) * side[0] + (yy - c[1]) * side[1]) / half_w
        band = np.exp(-((v - 0.35) / 0.13) ** 2) * np.clip(1 - np.abs(u) / 0.75, 0, 1) ** 0.7
        newL = newL + band * (22 + 0.25 * (100 - tgt[0] * 100 / 255) * 2.55 * 0.4)
        out[..., 0] = np.where(mu, newL, out[..., 0])
        out[..., 1] = np.where(mu, tgt[1], out[..., 1])
        out[..., 2] = np.where(mu, tgt[2], out[..., 2])
        A = np.maximum(A, mu.astype(np.float32))
    A = cv2.GaussianBlur(A, (0, 0), 0.8)[..., None]
    res = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    return (rgb * (1 - A) + res * A).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", required=True, help="hands 묶음 폴더(manifest.json 이 있는 곳의 상위 collections 기준 경로를 씀)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--photos", type=int, default=40)
    ap.add_argument("--colors", type=int, default=4, help="손마다 입힐 색 개수(짝)")
    ap.add_argument("--maxside", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
    D = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    G = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").eval()
    SP = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")
    SM = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").eval()
    hands = Path(a.hands)
    root = hands.parent.parent  # manifest 의 file 은 collections/... 로 시작
    items = [it for it in json.loads((hands / "manifest.json").read_text()) if it.get("nail_state") == "bare" and it.get("hands") == 1 and it.get("visible_nails", 0) >= 4]
    random.Random(a.seed).shuffle(items)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["pair"] for l in (out / "labels.jsonl").read_text().splitlines()} if (out / "labels.jsonl").exists() else set()
    seen = {l.strip() for l in (out / "seen.txt").read_text().splitlines()} if (out / "seen.txt").exists() else set()
    labels = open(out / "labels.jsonl", "a", encoding="utf-8")
    n = len(done)
    for it in items:
        if n >= a.photos:
            break
        p = root / it["file"]
        pid = it["sha256"][:12]
        if pid in done or pid in seen or not p.exists():
            continue
        with open(out / "seen.txt", "a") as f:
            f.write(pid + "\n")
        im = Image.open(p).convert("RGB")
        s = min(1, a.maxside / max(im.size))
        im = im.resize((int(im.width * s), int(im.height * s)), Image.BILINEAR)
        ms = nails(im, D, G, SP, SM)
        if not (it["visible_nails"] - 1 <= len(ms) <= it["visible_nails"]):
            print("손톱 개수 다름", pid, len(ms), it["visible_nails"], flush=True)
            continue
        rgb = np.asarray(im)
        Image.fromarray((np.any(ms, 0) * 255).astype(np.uint8)).save(out / "masks" / f"{pid}.png")
        rng = random.Random(int(pid, 16))
        for name in rng.sample(list(COLORS), a.colors):
            f = f"images/{pid}_{list(COLORS).index(name):02d}.jpg"
            Image.fromarray(paint(rgb, ms, COLORS[name])).save(out / f, quality=92)
            labels.write(json.dumps({"file": f, "mask": f"masks/{pid}.png", "nail.color": name, "nail.design": "원컬러", "pair": pid,
                                     "nails": len(ms), "pose": it.get("pose"), "source_sha256": it["sha256"]}, ensure_ascii=False) + "\n")
        labels.flush()
        n += 1
        print(f"{n}/{a.photos} {pid} 손톱 {len(ms)}", flush=True)


if __name__ == "__main__":
    main()
