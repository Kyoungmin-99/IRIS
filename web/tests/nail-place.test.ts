/// <reference types="node" />
// 손 점 → 손톱 자리 단위 테스트
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { nailQuads, type HandPoints } from '../src/beauty/nail-place.ts';

const hand = (pts: number[][], handed = 'Right'): HandPoints => ({ p: pts.map(([x, y]) => ({ x, y })), z: pts.map(() => 0), handed });
const best = (h: HandPoints) => {
  const a = nailQuads(h, 1), b = nailQuads(h, -1);
  const sum = (q: typeof a) => q.reduce((t, x) => t + x.vis, 0);
  return sum(a) >= sum(b) ? a : b;
};

test('손톱 자리: 손등을 위에서 찍어 손가락이 앞으로 늘어진 손(손목은 사진 밖)도 네 손가락 손톱이 보인다', () => {
  // 실제 사진(네일 84_8a89)에서 잡힌 손 점: 손목(0)이 사진 밖 왼쪽으로 추정되어 손목→뿌리 방향과 손끝 방향이 반대다
  const h = hand([[-117, 383], [45, 340], [196, 291], [325, 268], [451, 254], [60, 97], [204, 291], [299, 459], [373, 591], [-42, 164], [133, 369], [241, 535], [315, 670], [-133, 272], [53, 468], [176, 623], [281, 731], [-211, 398], [-46, 551], [71, 666], [173, 752]]);
  const q = best(h).filter((x) => x.finger > 0 && x.vis > 0.3);
  assert.equal(q.length, 4, `보이는 손톱 ${q.map((x) => x.finger)}`);
});

test('손톱 자리: 주먹(끝마디가 첫 마디와 손목 쪽 모두 반대로 접힘)은 네 손가락 손톱이 안 보인다', () => {
  // 손목 아래(0,300), 뿌리 위쪽, 둘째 마디는 더 위, 끝마디는 다시 아래(손바닥 쪽)로 접힘
  const pts: number[][] = [[0, 300], [-40, 260], [-60, 220], [-70, 190], [-75, 170]];
  for (const x of [-30, 0, 30, 60]) pts.push([x, 150], [x, 100], [x, 120], [x, 150]);
  const h = hand(pts);
  const q = best(h).filter((x) => x.finger > 0 && x.vis > 0.3);
  assert.equal(q.length, 0, `보이는 손톱 ${q.map((x) => x.finger)}`);
});
