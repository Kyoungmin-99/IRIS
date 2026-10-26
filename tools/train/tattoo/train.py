"""타투 잉크 분할 모델 학습(브라우저용 작은 U-Net, CPU 로도 학습 가능).

synth.py 가 만든 data/images, data/masks 를 쓴다. 완전 합성곱이라 학습은 256, 앱에서는 아무 크기(8의 배수)로 돌린다.
  python train.py --data data --out out --epochs 12
결과: out/best.pt, out/tattoo-ink.onnx(입력 image [n,3,H,W] 0~1, 출력 logit [n,1,H,W]), out/samples/*.png
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nail"))
from train import UNet, dice_loss  # noqa: E402  손톱 모델과 같은 작은 U-Net

S = 256


class InkData(torch.utils.data.Dataset):
    def __init__(self, root: Path, names: list[str], train: bool) -> None:
        self.root, self.names, self.train = root, names, train

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int):
        n = self.names[i]
        x = np.asarray(Image.open(self.root / "images" / f"{n}.jpg").convert("RGB"), np.float32) / 255
        y = np.asarray(Image.open(self.root / "masks" / f"{n}.png"), np.float32) / 255
        if self.train:
            k = random.randint(0, 3)
            x, y = np.rot90(x, k), np.rot90(y, k)
            if random.random() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]
            # 색·밝기 흔들기
            x = np.clip(x * random.uniform(0.8, 1.2) + random.uniform(-0.05, 0.05), 0, 1)
        # 옅은 잉크도 잉크로: 알파 0.08 넘으면 정답 쪽으로(진하기 그대로 쓰면 파인라인을 놓친다)
        y = np.clip(y / 0.35, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1), torch.from_numpy(np.ascontiguousarray(y))[None]


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
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--resume", default="", help="이어서 학습할 가중치(best.pt). 컨테이너가 다시 시작되면 학습이 끊긴다")
    a = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    root = Path(a.data)
    names = sorted(p.stem for p in (root / "masks").glob("*.png"))
    random.shuffle(names)
    nval = max(16, len(names) // 20)
    val, tr = names[:nval], names[nval:]
    print(f"학습 {len(tr)}장, 검증 {len(val)}장", flush=True)
    dl = torch.utils.data.DataLoader(InkData(root, tr, True), batch_size=a.batch, shuffle=True, num_workers=2, drop_last=True)
    dv = torch.utils.data.DataLoader(InkData(root, val, False), batch_size=a.batch, shuffle=False, num_workers=1)
    model = UNet()
    if a.resume:
        model.load_state_dict(torch.load(a.resume, map_location="cpu"))
        print(f"이어서 학습: {a.resume}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl))
    out = Path(a.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    best = 0.0
    if a.resume:
        best = float("-inf")  # 이어서 할 때는 첫 회 결과부터 저장
    step = 0
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for x, y in dl:
            logit = model(x)
            # 잉크 픽셀이 적어 양성에 가중(파인라인)
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(3.0)) + dice_loss(logit, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            step += 1
            if step % 200 == 0:
                # 컨테이너가 자주 다시 시작돼 에포크(30분 넘게) 단위로만 저장하면 계속 날아갔다: 200 배치마다 가중치 저장
                torch.save(model.state_dict(), out / "last.pt")
                print(f"  {step} 배치 저장 (손실 {tot / ((step - 1) % len(dl) + 1):.3f})", flush=True)
        model.eval()
        inter = union = 0.0
        with torch.no_grad():
            for k, (x, y) in enumerate(dv):
                p = (torch.sigmoid(model(x)) > 0.5).float()
                t = (y > 0.5).float()
                inter += (p * t).sum().item()
                union += ((p + t) > 0).float().sum().item()
                if k == 0:
                    save_samples(x, y, p, out / "samples" / f"ep{ep + 1:03d}.png")
        iou = inter / max(union, 1)
        print(f"[{ep + 1}/{a.epochs}] 손실 {tot / len(dl):.3f}  검증 IoU {iou:.3f}", flush=True)
        if iou > best:
            best = iou
            torch.save(model.state_dict(), out / "best.pt")
    print(f"가장 좋은 검증 IoU {best:.3f}", flush=True)
    model.load_state_dict(torch.load(out / "best.pt"))
    model.eval()
    torch.onnx.export(model, torch.zeros(1, 3, S, S), str(out / "tattoo-ink.onnx"), input_names=["image"], output_names=["logit"],
                      dynamic_axes={"image": {0: "n", 2: "h", 3: "w"}, "logit": {0: "n", 2: "h", 3: "w"}}, opset_version=17, external_data=False)
    print(f"ONNX 저장: {out / 'tattoo-ink.onnx'}", flush=True)


if __name__ == "__main__":
    main()
