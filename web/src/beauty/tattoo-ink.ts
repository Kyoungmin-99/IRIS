// 웹: 타투 잉크 분할 모델 실행(onnxruntime-web). 앞처리·뒤처리는 core/tattoo-ink.ts(웹·앱 공통), 입출력은 docs/models.md.
// 공개된 타투 분할 모델·상업 이용 가능한 데이터가 없어, 타투 없는 사진의 피부에 무작위 도안(가는 선·점묘·블랙워크·명암·컬러·한글/영문 레터링)을
// 합성한 자료로 직접 학습했다(tools/train/tattoo). 눈썹·속눈썹·머리카락·옷 주름·손톱은 잉크 아님으로 배움.
// 모델: public/models/tattoo-ink.onnx(작은 U-Net, 완전 합성곱이라 크기 자유)
import { tattooInkProb as inkCore } from '../core/tattoo-ink.ts';
import { ortWebRunner } from '../platform/ort-web.ts';

const runner = ortWebRunner('models/tattoo-ink.onnx');

/** 그림(RGBA, w×h) → 잉크 확률(0~1, w×h) */
export function tattooInkProb(rgba: Uint8ClampedArray, w: number, h: number, maxSide: number | readonly number[] = 768): Promise<Float32Array> {
  return inkCore({ data: rgba, width: w, height: h }, runner, maxSide);
}
