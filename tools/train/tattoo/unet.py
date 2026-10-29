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
