// 모델 실행기 인터페이스: 앞처리·뒤처리(core/*.ts)는 플랫폼과 상관없는 순수 계산이고, 실제 추론만 이 인터페이스로 넘긴다.
// 웹: platform/ort-web.ts(onnxruntime-web), 앱(React Native · Expo): onnxruntime-react-native 로 같은 모양의 실행기를 만들어 넣는다.
// 입출력 형식은 docs/models.md.

/** 실수 텐서(행 우선, NCHW) */
export interface Tensor {
  data: Float32Array;
  dims: number[];
}

/** 입력 하나 → 출력 하나인 ONNX 모델 실행기(우리 모델은 모두 그렇다) */
export interface ModelRunner {
  run(input: Tensor): Promise<Tensor>;
}

/** RGBA 그림(0~255, 행 우선) */
export interface RGBAImage {
  data: Uint8ClampedArray | Uint8Array;
  width: number;
  height: number;
}
