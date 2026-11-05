// 손등 피부 확률: 몸 분할(셀피 여러 부류)의 '몸 피부'는 가까이 찍은 손등을 군데군데만 피부로 읽어(사용자 손 h68 평균 0.24)
// 손등에 놓은 타투가 점 몇 개로만 구워졌다. 손 관절점으로 손등 다각형을 잡고, 그 안에서 손 자체의 피부색(가운데값)과
// 색도가 가까운 픽셀을 피부로 채운다. DOM 없음(앱에서 그대로 쓴다).

import type { RGBAImage } from './runner.ts';

type P = { x: number; y: number };

/** 손등 다각형에 쓰는 손 관절점(MediaPipe 21점 번호): 손목·엄지 뿌리 둘·검지·중지·약지·새끼 뿌리 */
const BACK = [0, 1, 2, 5, 9, 13, 17];

function hull(ps: P[]): P[] {
  const s = [...ps].sort((a, b) => a.x - b.x || a.y - b.y);
  const cross = (o: P, a: P, b: P) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lo: P[] = [], up: P[] = [];
  for (const p of s) {
    while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop();
    lo.push(p);
  }
  for (let i = s.length - 1; i >= 0; i--) {
    const p = s[i];
    while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], p) <= 0) up.pop();
    up.push(p);
  }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}

function inside(poly: P[], x: number, y: number): boolean {
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    if ((b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x) < 0) return false;
  }
  return true;
}

/**
 * base(몸 분할 피부, 그림 크기)에 손등 피부를 더한 새 배열.
 * 다각형: 손등 관절점의 볼록 껍질을 무게중심에서 1.15배로 키운 것. 그 안 픽셀 중 밝기 40 넘는 것의 색도(r·g 비율) 가운데값과의
 * 거리 d 로 피부 정도 = 1 − smooth(0.025, 0.05, d). 어두운 바탕(검정 매트)·소매는 밝기·색도로 빠진다.
 */
export function handBackSkin(img: RGBAImage, pts: P[], base: Float32Array | null): Float32Array {
  const { data, width: W, height: H } = img;
  const out = base ? Float32Array.from(base) : new Float32Array(W * H);
  const hp = hull(BACK.map((k) => pts[k]).filter(Boolean));
  if (hp.length < 3) return out;
  const c = hp.reduce((s, p) => ({ x: s.x + p.x / hp.length, y: s.y + p.y / hp.length }), { x: 0, y: 0 });
  const poly = hp.map((p) => ({ x: c.x + (p.x - c.x) * 1.15, y: c.y + (p.y - c.y) * 1.15 }));
  const x0 = Math.max(0, Math.floor(Math.min(...poly.map((p) => p.x)))), x1 = Math.min(W - 1, Math.ceil(Math.max(...poly.map((p) => p.x))));
  const y0 = Math.max(0, Math.floor(Math.min(...poly.map((p) => p.y)))), y1 = Math.min(H - 1, Math.ceil(Math.max(...poly.map((p) => p.y))));
  if (x1 <= x0 || y1 <= y0) return out;
  const rs: number[] = [], gs: number[] = [];
  const step = Math.max(1, Math.round(Math.sqrt(((x1 - x0) * (y1 - y0)) / 40000)));
  for (let y = y0; y <= y1; y += step) for (let x = x0; x <= x1; x += step) {
    if (!inside(poly, x, y)) continue;
    const i = (y * W + x) * 4, s = data[i] + data[i + 1] + data[i + 2];
    if (s < 120) continue;
    rs.push(data[i] / s);
    gs.push(data[i + 1] / s);
  }
  if (rs.length < 20) return out;
  const med = (a: number[]) => a.sort((p, q) => p - q)[a.length >> 1];
  const mr = med(rs), mg = med(gs);
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    if (!inside(poly, x, y)) continue;
    const i = y * W + x, s = data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2];
    if (s < 120) continue;
    const d = Math.hypot(data[i * 4] / s - mr, data[i * 4 + 1] / s - mg);
    const t = Math.min(1, Math.max(0, (d - 0.025) / 0.025));
    const k = 1 - t * t * (3 - 2 * t);
    if (k > out[i]) out[i] = k;
  }
  return out;
}
