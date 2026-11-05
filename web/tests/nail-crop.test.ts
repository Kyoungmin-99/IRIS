/// <reference types="node" />
// 사진 모드 손톱 모델 크롭(손가락 끝마디 → 128×128) 단위 테스트
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { backProject, cropFrame, pickNail, CROP } from '../src/core/nail-crop.ts';

test('크롭 틀: 손끝이 크롭 위쪽, 크기는 끝마디 길이의 2.4배', () => {
  // 손가락이 위를 가리킴(끝 y 작음)
  const f = cropFrame({ x: 100, y: 50 }, { x: 100, y: 90 })!;
  const at = (x: number, y: number) => ({ x: f.p0.x + f.a.x * x + f.b.x * y, y: f.p0.y + f.a.y * x + f.b.y * y });
  const top = at(64, 0), bottom = at(64, 128);
  assert.ok(top.y < bottom.y, `${top.y} ${bottom.y}`);
  assert.ok(Math.abs(bottom.y - top.y - 40 * CROP.side) < 1e-6);
  // 가운데는 끝에서 끝마디 쪽으로 0.15L
  const c = at(64, 64);
  assert.ok(Math.abs(c.x - 100) < 1e-6 && Math.abs(c.y - (50 + 0.15 * 40)) < 1e-6, JSON.stringify(c));
});

test('손톱 성분: 가운데 손끝 아래 덩어리만 남기고 가장자리 조각은 버린다', () => {
  const S = 128, prob = new Float32Array(S * S);
  const box = (x0: number, y0: number, x1: number, y1: number) => { for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) prob[y * S + x] = 0.9; };
  box(50, 40, 78, 90); // 손톱
  box(2, 2, 20, 20); // 옆 손가락 조각
  const n = pickNail(prob)!;
  assert.ok(n[60 * S + 64] > 0.5);
  assert.equal(n[10 * S + 10], 0);
  assert.equal(pickNail(new Float32Array(S * S)), null);
});

test('되돌리기: 크롭 손톱이 원본의 같은 자리로', () => {
  const S = 128, prob = new Float32Array(S * S);
  for (let y = 40; y < 90; y++) for (let x = 50; x < 78; x++) prob[y * S + x] = 1;
  const f = cropFrame({ x: 100, y: 50 }, { x: 100, y: 90 })!;
  const m = backProject(prob, f, 200, 200);
  // 크롭 (64, 64) → 원본 (100, 56)
  assert.equal(m[56 * 200 + 100], 1);
  assert.equal(m[150 * 200 + 100], 0);
  let a = 0;
  for (const v of m) a += v;
  // 크롭 28×50 칸, 한 칸 = 96/128 원본 픽셀
  const expect = 28 * 50 * (96 / 128) ** 2;
  assert.ok(Math.abs(a - expect) / expect < 0.1, `${a} ${expect}`);
});
