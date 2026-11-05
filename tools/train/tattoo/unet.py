"""잉크 모델 U-Net 만들기 · 넓히기(손톱 모델과 같은 UNet, 단계 수만 다름).

4단계(16,32,64,128)는 한 픽셀이 보는 범위가 좁아 넓은 음영 도안(강아지 초상·태극)을 놓쳐 5단계(…,256)를 더한다.
5단계는 입력 H·W 가 16의 배수여야 한다(4단계는 8).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nail"))
from train import UNet  # noqa: E402


def ch_of(state: dict) -> tuple[int, ...]:
    """저장한 가중치에서 단계별 채널 수"""
    n = len({k.split(".")[1] for k in state if k.startswith("enc.")})
    return tuple(int(state[f"enc.{i}.0.weight"].shape[0]) for i in range(n))


def load(state: dict, ch: tuple[int, ...] | None = None) -> UNet:
    """가중치로 모델 만들기. ch 가 저장본보다 깊으면 겹치는 층을 옮긴다:
    enc·head 는 이름 그대로, 위로 올라가는 쪽(up·dec)은 새 바닥 단계가 앞에 붙어 번호가 하나씩 밀린다"""
    old = ch_of(state)
    ch = ch or old
    model = UNet(ch)
    if ch == old:
        model.load_state_dict(state)
        return model
    shift = len(ch) - len(old)
    assert shift > 0 and ch[:len(old)] == old, (old, ch)
    mine = model.state_dict()
    for k, v in state.items():
        p = k.split(".")
        if p[0] in ("up", "dec"):
            p[1] = str(int(p[1]) + shift)
        k2 = ".".join(p)
        if k2 in mine and mine[k2].shape == v.shape:
            mine[k2] = v
    model.load_state_dict(mine)
    return model


def from_onnx(path: str, calib: float = 0.5) -> UNet:
    """앱 ONNX(export.py 출력)에서 가중치 되살리기: 저장본(.pt)이 사라졌을 때(컨테이너 교체) 이어 학습하려고.
    내보낼 때 BatchNorm 이 합성곱에 접혀 있어 BN 은 항등(평균 0·분산 1)으로 두므로 train.py --freeze-bn 으로 이어 학습한다.
    calib: export.py --calib 로 뺀 log(p/(1-p)) 를 머리 bias 에 되돌림"""
    import math

    import onnx
    from onnx import numpy_helper

    m = onnx.load(path)
    W = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    convs = [n for n in m.graph.node if n.op_type in ("Conv", "ConvTranspose")]
    k = sum(1 for n in convs if n.op_type == "ConvTranspose")  # 위로 올라가는 단계 수 = 단계 수 - 1
    ch = tuple(int(W[convs[2 * i].input[1]].shape[0]) for i in range(k + 1))
    model = UNet(ch)
    mods = [mm for i in range(k + 1) for mm in (model.enc[i][0], model.enc[i][3])]
    mods += [mm for j in range(k) for mm in (model.up[j], model.dec[j][0], model.dec[j][3])] + [model.head]
    assert len(convs) == len(mods), (len(convs), len(mods))
    for n, mod in zip(convs, mods):
        w = W[n.input[1]]
        assert tuple(w.shape) == tuple(mod.weight.shape), (n.name, w.shape, tuple(mod.weight.shape))
        mod.weight.data = torch.from_numpy(w.copy())
        if len(n.input) > 2:
            mod.bias.data = torch.from_numpy(W[n.input[2]].copy())
    for mm in model.modules():
        if isinstance(mm, torch.nn.BatchNorm2d):
            mm.weight.data.fill_(1)
            mm.bias.data.zero_()
            mm.running_mean.zero_()
            mm.running_var.fill_(1 - mm.eps)
    if calib != 0.5:
        model.head.bias.data += math.log(calib / (1 - calib))
    return model
