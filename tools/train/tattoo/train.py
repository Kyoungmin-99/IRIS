"""타투 잉크 분할 모델 학습(브라우저용 작은 U-Net, CPU 로도 학습 가능).

synth.py 가 만든 data/images, data/masks 를 쓴다. 완전 합성곱이라 학습은 256, 앱에서는 아무 크기(8의 배수)로 돌린다.
  python train.py --data data --out out --epochs 12
실제 사진 자료(teacher.py → crops.py, weights/ 포함)를 섞을 때: --data data real:3  (':3' = 그 묶음을 3번 반복)
weights/{이름}.png 가 있으면 픽셀별 학습 가중치(0 = 모름, 배우지 않음), 없으면 전부 1.
결과: out/best.pt, out/tattoo-ink.onnx(입력 image [n,3,H,W] 0~1, 출력 logit [n,1,H,W]), out/samples/*.png
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nail"))
from train import UNet  # noqa: E402  손톱 모델과 같은 작은 U-Net
from unet import load as load_unet  # noqa: E402

S = 256


class InkData(torch.utils.data.Dataset):
    def __init__(self, items: list[tuple[Path, str]], train: bool) -> None:
        self.items, self.train = items, train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        root, n = self.items[i]
        x = np.asarray(Image.open(root / "images" / f"{n}.jpg").convert("RGB"), np.float32) / 255
        y = np.asarray(Image.open(root / "masks" / f"{n}.png"), np.float32) / 255
        wp = root / "weights" / f"{n}.png"
        w = np.asarray(Image.open(wp), np.float32) / 255 if wp.exists() else np.ones_like(y)
        if self.train:
            k = random.randint(0, 3)
            x, y, w = np.rot90(x, k), np.rot90(y, k), np.rot90(w, k)
            if random.random() < 0.5:
                x, y, w = x[:, ::-1], y[:, ::-1], w[:, ::-1]
            # 색·밝기 흔들기
            x = np.clip(x * random.uniform(0.8, 1.2) + random.uniform(-0.05, 0.05), 0, 1)
        # 옅은 잉크도 잉크로: 알파 0.08 넘으면 정답 쪽으로(진하기 그대로 쓰면 파인라인을 놓친다)
        y = np.clip(y / 0.35, 0, 1)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731
        return t(x).permute(2, 0, 1), t(y)[None], t(w)[None]


def save_atomic(obj, path):
    """저장 도중 컨테이너가 꺼져도 잘린 파일이 남지 않게: 임시 파일에 쓴 뒤 바꿔치기"""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    if isinstance(obj, str):
        tmp.write_text(obj)
    else:
        torch.save(obj, tmp)
    os.replace(tmp, path)


def save_samples(x, y, p, path: Path) -> None:
    rows = []
    for i in range(min(6, x.shape[0])):
        a = (x[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        b = np.repeat((y[i, 0].cpu().numpy() * 255).astype(np.uint8)[..., None], 3, 2)
        c = np.repeat((p[i, 0].cpu().numpy() * 255).astype(np.uint8)[..., None], 3, 2)
        rows.append(np.concatenate([a, b, c], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data"], help="자료 폴더(여럿 가능, '폴더:반복 수')")
    ap.add_argument("--out", default="out")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=0, help="이 배치 수에서 멈춤(0 = 끝까지). 컨테이너가 자주 다시 시작돼 1에포크(약 1,700 배치)를 못 마쳐 점수가 안 나왔다")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--step0", type=int, default=0, help="이 바퀴에서 이미 한 배치 수(컨테이너 재시작 뒤 이어 하기). 저장할 때 step0+한 배치를 out/round_steps 에 적는다")
    ap.add_argument("--ch", default="", help="단계별 채널(예: 16,32,64,128,256). 비우면 저장본과 같게(없으면 4단계)")
    ap.add_argument("--freeze-bn", action="store_true", help="BatchNorm 을 고정(평가 상태). 앱 ONNX 에서 되살린 가중치는 BN 이 합성곱에 접혀 있어(항등 BN) 학습 중 배치 통계로 바꾸면 출력이 무너진다")
    ap.add_argument("--resume", default="", help="이어서 학습할 가중치(best.pt). 컨테이너가 다시 시작되면 학습이 끊긴다")
    a = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    tr, val = [], []
    for d in a.data:
        root, rep = (Path(d.rsplit(":", 1)[0]), int(d.rsplit(":", 1)[1])) if ":" in d else (Path(d), 1)
        names = sorted(p.stem for p in (root / "masks").glob("*.png"))
        random.shuffle(names)
        nval = max(16, len(names) // 20)
        val += [(root, n) for n in names[:nval]]
        tr += [(root, n) for n in names[nval:]] * rep
        print(f"{root}: {len(names)}장 × {rep}", flush=True)
    print(f"학습 {len(tr)}장, 검증 {len(val)}장", flush=True)
    dl = torch.utils.data.DataLoader(InkData(tr, True), batch_size=a.batch, shuffle=True, num_workers=2, drop_last=True)
    dv = torch.utils.data.DataLoader(InkData(val, False), batch_size=a.batch, shuffle=False, num_workers=1)
    ch = tuple(int(c) for c in a.ch.split(",")) if a.ch else None
    model = UNet(ch) if ch else UNet()
    if a.resume:
        model = load_unet(torch.load(a.resume, map_location="cpu"), ch)
        print(f"이어서 학습: {a.resume}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * len(dl) if not a.steps else min(a.steps, a.epochs * len(dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total)
    out = Path(a.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    best = 0.0
    if a.resume:
        best = float("-inf")  # 이어서 할 때는 첫 회 결과부터 저장
    step = 0
    t0 = saved = time.time()
    for ep in range(a.epochs):
        model.train()
        if a.freeze_bn:
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)
        tot = 0.0
        for x, y, w in dl:
            logit = model(x)
            # 잉크 픽셀이 적어 양성에 가중(파인라인). w = 0 인 곳(모름)은 배우지 않는다
            bce = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(3.0), reduction="none")
            p = torch.sigmoid(logit) * w
            inter = (p * y).sum((1, 2, 3))
            dice = 1 - ((2 * inter + 1) / (p.sum((1, 2, 3)) + (y * w).sum((1, 2, 3)) + 1)).mean()
            loss = (bce * w).sum() / w.sum().clamp(min=1) + dice
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            step += 1
            if step % 20 == 0:
                print(f"  {step} 배치 ({(time.time() - t0) / step:.1f}초/배치)", flush=True)
            if time.time() - saved > 90:
                # 컨테이너가 자주 다시 시작돼(차례가 끝나면 4~5분 안에 잠듦) 5분마다 저장하면 120 배치 안팎이 계속 날아갔다: 90초마다 가중치 저장
                save_atomic(model.state_dict(), out / "last.pt")
                save_atomic(str(a.step0 + step), out / "round_steps")
                saved = time.time()
                print(f"  {step} 배치 저장 (손실 {tot / ((step - 1) % len(dl) + 1):.3f})", flush=True)
            if step >= total:
                break
        model.eval()
        inter = union = 0.0
        with torch.no_grad():
            for k, (x, y, w) in enumerate(dv):
                p = (torch.sigmoid(model(x)) > 0.5).float() * (w > 0.5)
                t = (y > 0.5).float() * (w > 0.5)
                inter += (p * t).sum().item()
                union += ((p + t) > 0).float().sum().item()
                if k == 0:
                    save_samples(x, y, p, out / "samples" / f"ep{ep + 1:03d}.png")
        iou = inter / max(union, 1)
        print(f"[{ep + 1}/{a.epochs}] 손실 {tot / max(1, (step - 1) % len(dl) + 1):.3f}  검증 IoU {iou:.3f}", flush=True)
        if iou > best:
            best = iou
            save_atomic(model.state_dict(), out / "best.pt")
    print(f"가장 좋은 검증 IoU {best:.3f}", flush=True)
    model.load_state_dict(torch.load(out / "best.pt"))
    model.eval()
    torch.onnx.export(model, torch.zeros(1, 3, S, S), str(out / "tattoo-ink.onnx"), input_names=["image"], output_names=["logit"],
                      dynamic_axes={"image": {0: "n", 2: "h", 3: "w"}, "logit": {0: "n", 2: "h", 3: "w"}}, opset_version=17, external_data=False)
    print(f"ONNX 저장: {out / 'tattoo-ink.onnx'}", flush=True)


if __name__ == "__main__":
    main()
