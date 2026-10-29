// 타투 잉크 확률 — 자체 학습 타투 잉크 분할 모델(tattoo-ink.onnx)의 앞처리·뒤처리(순수 계산, 웹·앱 공통). 입출력: docs/models.md
// 그림을 긴 변 maxSide 로 넓이 평균해 줄이고(16의 배수로 검정 채움: 5단계 U-Net 은 4번 반으로 줄인다) 한 번에 돌린 뒤, 로짓을 원래 크기로 양선형으로 키워 시그모이드.
import type { ModelRunner, RGBAImage } from './runner.ts';
import { toCHW, upsample } from './sample.ts';

/** 그림 → 잉크 확률(0~1, width×height) */
export async function tattooInkProb(img: RGBAImage, runner: ModelRunner, maxSide = 768): Promise<Float32Array> {
  const { width: w, height: h } = img;
  const s = Math.min(1, maxSide / Math.max(w, h));
  const sw = Math.max(8, Math.round(w * s)), sh = Math.max(8, Math.round(h * s));
  const W = Math.ceil(sw / 16) * 16, H = Math.ceil(sh / 16) * 16;
  const out = await runner.run({ data: toCHW(img, sw, sh, W, H), dims: [1, 3, H, W] });
  return upsample(out.data, W, sw, sh, w, h, (v) => 1 / (1 + Math.exp(-v)));
}
