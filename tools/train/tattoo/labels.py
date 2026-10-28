"""선생님 정답(teacher.py 출력) 읽기 공통: 너무 빽빽한 상자는 '모름'으로.

상자 안 가중 픽셀의 60% 넘게 잉크인 정답은 대부분 틀렸다(몸통·팔 전체 채움, 얼굴 가리개, 옷 로고, 신문 지면 — 1,327장 중 73장).
진짜 꽉 찬 먹 도안도 있지만 SAM 영역을 통째로 칠한 경우와 가를 수 없어 학습·채점 모두에서 그 상자를 가중 0 으로 둔다.
"""
from __future__ import annotations

import numpy as np

DENSE = 0.6


def drop_dense(m: dict, ink: np.ndarray, w: np.ndarray) -> np.ndarray:
    """ink·w: bool (H, W). 빽빽한 상자 안을 w=False 로 바꾼 새 가중치"""
    w = w.copy()
    for r in m.get("regions", []):
        x0, y0, x1, y1 = map(int, m["boxes"][r["box"]])
        bw = w[y0:y1 + 1, x0:x1 + 1]
        n = int(bw.sum())
        if n and (ink[y0:y1 + 1, x0:x1 + 1] & bw).sum() > DENSE * n:
            w[y0:y1 + 1, x0:x1 + 1] = False
    return w
