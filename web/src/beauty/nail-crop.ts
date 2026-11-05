// 웹: 손톱 전용 분할 모델 실행(onnxruntime-web). 앞처리·뒤처리는 core/nail-crop.ts(웹·앱 공통), 입출력은 docs/models.md.
// 모델: public/models/nail-crop-128.onnx — AI 저장소 models/nail/nail_seg.onnx 그대로(MobileNetV3-small + U-Net 5.5MB,
// 학습 데이터 vpapenko/nails-segmentation-dataset CC0-1.0 + 상업 이용 가능 자유 라이선스 손 사진, 맨손톱·칠한 손톱 모두).
// SAM 은 사진 전체에서 짧은 맨손톱을 손가락 끝마디째 잡기 쉬워(손 편 사진 5장에서 손톱 25개 중 9개) 손 점이 있으면 이 모델을 먼저 쓴다.
import { nailCropMasks as nailCropCore } from '../core/nail-crop.ts';
import { ortWebRunner } from '../platform/ort-web.ts';

export { CROP, TIPS, cropFrame, pickNail, backProject } from '../core/nail-crop.ts';

const runner = ortWebRunner('models/nail-crop-128.onnx');

/** 캔버스 그림 → 손마다 손톱 마스크(core/nail-crop.ts 의 nailCropMasks) */
export function nailCropMasks(img: HTMLCanvasElement, hands: { p: { x: number; y: number }[] }[]): ReturnType<typeof nailCropCore> {
  const data = img.getContext('2d', { willReadFrequently: true })!.getImageData(0, 0, img.width, img.height).data;
  return nailCropCore({ data, width: img.width, height: img.height }, hands, runner);
}
