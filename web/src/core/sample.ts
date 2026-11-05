// 그림 다루기(순수 계산): 넓이 평균 줄이기·양선형 읽기·어파인으로 잘라 내기. DOM·캔버스 없이 웹·앱에서 같게 돈다.
import type { RGBAImage } from './runner.ts';

/** RGBA → NCHW(1×3×H×W, 0~1) 넓이 평균으로 줄여 담기. 출력 크기 outW×outH 안 (sw×sh) 만 채우고 나머지는 0(검정) */
export function toCHW(img: RGBAImage, sw: number, sh: number, outW = sw, outH = sh): Float32Array {
  const { data: d, width: w, height: h } = img;
  const x = new Float32Array(3 * outW * outH);
  const P = outW * outH;
  for (let y = 0; y < sh; y++) for (let xx = 0; xx < sw; xx++) {
    const x0 = Math.floor((xx * w) / sw), x1 = Math.max(x0 + 1, Math.floor(((xx + 1) * w) / sw));
    const y0 = Math.floor((y * h) / sh), y1 = Math.max(y0 + 1, Math.floor(((y + 1) * h) / sh));
    let r = 0, g = 0, b = 0, n = 0;
    for (let yy = y0; yy < y1; yy++) for (let q = x0; q < x1; q++) {
      const o = (yy * w + q) * 4;
      r += d[o]; g += d[o + 1]; b += d[o + 2]; n++;
    }
    const i = y * outW + xx;
    x[i] = r / n / 255;
    x[P + i] = g / n / 255;
    x[2 * P + i] = b / n / 255;
  }
  return x;
}

/** 한 채널(gw×gh 격자, 유효 영역 vw×vh)을 w×h 로 양선형으로 키우며 f 를 씌운다 */
export function upsample(grid: Float32Array, gw: number, vw: number, vh: number, w: number, h: number, f: (v: number) => number = (v) => v): Float32Array {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const fy = Math.min(vh - 1.001, Math.max(0, ((y + 0.5) * vh) / h - 0.5));
    const yi = Math.floor(fy), ty = fy - yi;
    for (let x = 0; x < w; x++) {
      const fx = Math.min(vw - 1.001, Math.max(0, ((x + 0.5) * vw) / w - 0.5));
      const xi = Math.floor(fx), tx = fx - xi;
      const v = (grid[yi * gw + xi] * (1 - tx) + grid[yi * gw + xi + 1] * tx) * (1 - ty) + (grid[(yi + 1) * gw + xi] * (1 - tx) + grid[(yi + 1) * gw + xi + 1] * tx) * ty;
      out[y * w + x] = f(v);
    }
  }
  return out;
}

/**
 * 어파인 틀로 S×S 를 잘라 NCHW(0~1) 의 k 번째 칸에 담는다: 크롭 (x, y) 칸 가운데 → 원본 p0 + a·(x+0.5) + b·(y+0.5) 를 양선형으로 읽고,
 * 원본 밖은 검정(학습 크롭과 같게)
 */
export function warpToCHW(img: RGBAImage, p0: { x: number; y: number }, a: { x: number; y: number }, b: { x: number; y: number }, S: number, out: Float32Array, k: number): void {
  const { data: d, width: w, height: h } = img;
  const P = S * S, base = k * 3 * P;
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const sx = p0.x + a.x * (x + 0.5) + b.x * (y + 0.5) - 0.5, sy = p0.y + a.y * (x + 0.5) + b.y * (y + 0.5) - 0.5;
    const i = y * S + x;
    if (sx < -0.5 || sy < -0.5 || sx > w - 0.5 || sy > h - 0.5) {
      out[base + i] = out[base + P + i] = out[base + 2 * P + i] = 0;
      continue;
    }
    const x0 = Math.max(0, Math.min(w - 1, Math.floor(sx))), y0 = Math.max(0, Math.min(h - 1, Math.floor(sy)));
    const x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1);
    const tx = Math.max(0, Math.min(1, sx - x0)), ty = Math.max(0, Math.min(1, sy - y0));
    for (let c = 0; c < 3; c++) {
      const v = (d[(y0 * w + x0) * 4 + c] * (1 - tx) + d[(y0 * w + x1) * 4 + c] * tx) * (1 - ty) + (d[(y1 * w + x0) * 4 + c] * (1 - tx) + d[(y1 * w + x1) * 4 + c] * tx) * ty;
      out[base + c * P + i] = v / 255;
    }
  }
}
