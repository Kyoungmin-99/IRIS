/// <reference types="node" />
// 모델 앞처리·뒤처리 순수 계산(core/sample.ts) 단위 테스트
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { toCHW, upsample, warpToCHW } from '../src/core/sample.ts';
import { tattooInkProb } from '../src/core/tattoo-ink.ts';

const img = (w: number, h: number, f: (x: number, y: number) => [number, number, number]) => {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data.set([...f(x, y), 255], (y * w + x) * 4);
  return { data, width: w, height: h };
};

test('toCHW: 넓이 평균으로 줄여 채널 평면에, 남는 칸은 0', () => {
  const im = img(4, 2, (x) => (x < 2 ? [255, 0, 0] : [0, 0, 255]));
  const t = toCHW(im, 2, 1, 4, 2);
  assert.equal(t.length, 3 * 8);
  assert.equal(t[0], 1); // R, (0,0)
  assert.equal(t[8 * 2 + 1], 1); // B, (1,0)
  assert.equal(t[4], 0); // 채운 칸(아래 줄)
});

test('warpToCHW: 단위 틀이면 원본 그대로, 원본 밖은 검정', () => {
  const im = img(3, 3, (x, y) => [x * 100, y * 100, 50]);
  const out = new Float32Array(3 * 16);
  warpToCHW(im, { x: 0, y: 0 }, { x: 1, y: 0 }, { x: 0, y: 1 }, 4, out, 0);
  assert.ok(Math.abs(out[1 * 4 + 2] - 200 / 255) < 1e-6); // R at (2,1) = x*100
  assert.ok(Math.abs(out[16 + 1 * 4 + 2] - 100 / 255) < 1e-6); // G = y*100
  assert.equal(out[3 * 4 + 3], 0); // (3,3) 원본 밖
});

test('upsample: 상수는 상수로, 시그모이드 적용', () => {
  const g = new Float32Array(16).fill(0);
  const u = upsample(g, 4, 4, 4, 9, 7, (v) => 1 / (1 + Math.exp(-v)));
  assert.equal(u.length, 63);
  assert.ok(u.every((v) => Math.abs(v - 0.5) < 1e-9));
});

test('tattooInkProb: 실행기 입력은 8의 배수 NCHW, 출력은 원래 크기 확률', async () => {
  let dims: number[] = [];
  const runner = { run: async (t: { data: Float32Array; dims: number[] }) => { dims = t.dims; return { data: new Float32Array(t.dims[2] * t.dims[3]).fill(10), dims: [1, 1, t.dims[2], t.dims[3]] }; } };
  const p = await tattooInkProb(img(30, 20, () => [10, 10, 10]), runner, 768);
  assert.deepEqual(dims, [1, 3, 24, 32]);
  assert.equal(p.length, 600);
  assert.ok(p[0] > 0.99);
});
