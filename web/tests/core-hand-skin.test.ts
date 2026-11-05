/// <reference types="node" />
// 손등 피부 확률(core/hand-skin.ts) 단위 테스트
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { handBackSkin } from '../src/core/hand-skin.ts';

/** 100×100 검정 바탕, 가운데 50×50 살색 손등 */
function scene(): { data: Uint8ClampedArray; width: number; height: number } {
  const W = 100, H = 100, data = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = (y * W + x) * 4, on = x >= 25 && x < 75 && y >= 25 && y < 75;
    data[i] = on ? 210 : 15; data[i + 1] = on ? 160 : 15; data[i + 2] = on ? 130 : 15; data[i + 3] = 255;
  }
  return { data, width: W, height: H };
}
/** 손 21점: 손목(0)이 아래, 손가락 뿌리(5·9·13·17)가 위 */
function pts(): { x: number; y: number }[] {
  const p = Array.from({ length: 21 }, () => ({ x: 50, y: 50 }));
  p[0] = { x: 50, y: 70 }; p[1] = { x: 33, y: 62 }; p[2] = { x: 30, y: 50 };
  p[5] = { x: 35, y: 30 }; p[9] = { x: 47, y: 28 }; p[13] = { x: 58, y: 29 }; p[17] = { x: 68, y: 33 };
  return p;
}

test('손등 다각형 안 살색은 피부, 바깥과 검정 바탕은 그대로', () => {
  const img = scene();
  const s = handBackSkin(img, pts(), new Float32Array(100 * 100));
  assert.ok(s[50 * 100 + 50] > 0.99, '손등 가운데');
  assert.equal(s[5 * 100 + 5], 0, '검정 바탕');
  assert.equal(s[95 * 100 + 50], 0, '다각형 밖');
});

test('색이 다른 픽셀(옷 색)은 피부가 아님, 몸 분할 값은 줄이지 않음', () => {
  const img = scene();
  const i = (50 * 100 + 45) * 4;
  img.data[i] = 60; img.data[i + 1] = 90; img.data[i + 2] = 160;
  const base = new Float32Array(100 * 100);
  base[50 * 100 + 52] = 0.7;
  const s = handBackSkin(img, pts(), base);
  assert.ok(s[50 * 100 + 45] < 0.01);
  assert.ok(s[50 * 100 + 52] >= 0.7);
});
