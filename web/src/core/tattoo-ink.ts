// 타투 잉크 확률 — 자체 학습 타투 잉크 분할 모델(tattoo-ink.onnx)의 앞처리·뒤처리(순수 계산, 웹·앱 공통). 입출력: docs/models.md
// 그림을 긴 변 maxSide 로 넓이 평균해 줄이고(16의 배수로 검정 채움: 5단계 U-Net 은 4번 반으로 줄인다) 한 번에 돌린 뒤, 로짓을 원래 크기로 양선형으로 키워 시그모이드.
// maxSide 가 여럿이면 크기마다 돌려 확률을 평균한다.
import type { ModelRunner, RGBAImage } from './runner.ts';
import { toCHW, upsample } from './sample.ts';

/**
 * 그림 → 잉크 확률(0~1, width×height).
 * maxSide 에 여러 크기를 주면 크기마다 돌려 확률을 평균한다: 모델은 긴 변 768 사진 조각으로 배워 크기에 민감하다.
 * 실제 사진 301장(원본에서 줄여 돌림)에서 1024 한 번은 IoU 0.499 · 맨살 팔 오검출 9.5%, [768, 576] 평균은 0.553 · 4.8%
 * (계산량은 1024 한 번보다 적다: 768² + 576² < 1024²). 그림이 작아 두 크기가 같아지면 한 번만 돌린다.
 */
export async function tattooInkProb(img: RGBAImage, runner: ModelRunner, maxSide: number | readonly number[] = 768): Promise<Float32Array> {
  const { width: w, height: h } = img;
  const sides = typeof maxSide === 'number' ? [maxSide] : maxSide;
  let acc: Float32Array | null = null, n = 0;
  const seen = new Set<string>();
  for (const side of sides) {
    const s = Math.min(1, side / Math.max(w, h));
    const sw = Math.max(8, Math.round(w * s)), sh = Math.max(8, Math.round(h * s));
    if (seen.has(`${sw}x${sh}`)) continue;
    seen.add(`${sw}x${sh}`);
    const W = Math.ceil(sw / 16) * 16, H = Math.ceil(sh / 16) * 16;
    const out = await runner.run({ data: toCHW(img, sw, sh, W, H), dims: [1, 3, H, W] });
    const p = upsample(out.data, W, sw, sh, w, h, (v) => 1 / (1 + Math.exp(-v)));
    if (!acc) acc = p;
    else for (let i = 0; i < p.length; i++) acc[i] += p[i];
    n++;
  }
  if (!acc) return new Float32Array(w * h);
  if (n > 1) for (let i = 0; i < acc.length; i++) acc[i] /= n;
  return acc;
}
