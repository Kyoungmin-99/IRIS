# 모델 입출력 · 앞처리 · 뒤처리 (웹 · React Native(Expo) 앱 공통)

IRIS 사진 모드가 쓰는 모델과, 모델 바깥의 계산(앞처리·뒤처리)을 정리한다. 앱(React Native · Expo)에 옮길 때 이 문서와 `web/src/core/` 를 기준으로 한다.

## 구조 — 화면 코드와 모델 코드를 나눈다

| 층 | 위치 | 하는 일 | 앱으로 옮길 때 |
|---|---|---|---|
| 순수 계산(앞처리·뒤처리) | `web/src/core/*.ts` | RGBA 배열 → 텐서, 텐서 → 마스크·확률. DOM·캔버스·네트워크 없음 | **그대로 가져다 씀**(TypeScript, Hermes/JSC 에서 돈다) |
| 모델 실행기 | `web/src/core/runner.ts` 의 `ModelRunner` 인터페이스 | `run({data, dims}) → {data, dims}` 한 가지 | 앱은 `onnxruntime-react-native` 로 같은 모양의 실행기를 만든다(아래 예) |
| 웹 실행기 | `web/src/platform/ort-web.ts` | onnxruntime-web(WASM)으로 `ModelRunner` 구현 | 쓰지 않음 |
| 웹 화면 연결 | `web/src/beauty/nail-crop.ts`, `beauty/tattoo-ink.ts` 등 | 캔버스 → RGBA 로 바꿔 core 를 부름 | 앱은 카메라·갤러리 그림을 RGBA 로 바꿔 core 를 부른다 |

- 그림 형식(`RGBAImage`): `{ data: Uint8ClampedArray | Uint8Array, width, height }`, 행 우선 RGBA 0~255.
- 텐서 형식(`Tensor`): `{ data: Float32Array, dims: number[] }`, 행 우선 NCHW.
- 모든 자체 모델은 입력 하나·출력 하나, float32, 입력 값 범위 **RGB 0~1(평균·표준편차 정규화 없음, 필요하면 모델 안에서 함)**, 출력은 **로짓**(시그모이드 전).

### 앱 실행기 예 (onnxruntime-react-native)

```ts
import { InferenceSession, Tensor as OrtTensor } from 'onnxruntime-react-native';
import type { ModelRunner } from './core/runner';

export async function ortNativeRunner(modelPath: string): Promise<ModelRunner> {
  const session = await InferenceSession.create(modelPath); // 앱 번들 또는 내려받은 파일 경로
  return {
    async run({ data, dims }) {
      const out = await session.run({ [session.inputNames[0]]: new OrtTensor('float32', data, dims) });
      const o = out[session.outputNames[0]];
      return { data: o.data as Float32Array, dims: [...o.dims] };
    },
  };
}
```

- Expo 는 `onnxruntime-react-native` 가 네이티브 모듈이라 **개발 빌드(expo prebuild / EAS dev client)** 가 필요하다(Expo Go 에서는 안 돈다). 공식 config plugin 이 있다.
- 사진을 RGBA 로 바꾸기: `expo-image-manipulator` 로 크기를 줄이고 `react-native-skia`(`Skia.Image.MakeImageFromEncoded` → `readPixels`) 등으로 RGBA 를 얻는다.

---

## 1. 손톱 분할 — `nail-crop-128.onnx`

| 항목 | 값 |
|---|---|
| 파일 · 크기 | `web/public/models/nail-crop-128.onnx`, 5.3MB |
| 출처 · 라이선스 | AI 저장소 `models/nail/nail_seg.onnx`(자체 학습). MobileNetV3-small(torchvision BSD-3 초기값) + U-Net. 학습 데이터 vpapenko/nails-segmentation-dataset(CC0-1.0) + 상업 이용 가능 자유 라이선스 손 사진 |
| 입력 | `image` float32 `[n, 3, 128, 128]`, RGB 0~1, 손가락 n 개를 한 번에 |
| 출력 | `logit` float32 `[n, 1, 128, 128]` |
| core | `web/src/core/nail-crop.ts` — `nailCropMasks(img, hands, runner)` |

**앞처리** (`cropFrame`, `warpToCHW`): 손 관절점(MediaPipe Hand Landmarker 21점, 그림 픽셀 좌표)에서 손가락마다 끝(TIP)·끝마디 뿌리(DIP) 두 점을 쓴다(엄지 4·3, 검지 8·7, 중지 12·11, 약지 16·15, 새끼 20·19).
끝마디 길이 L = |TIP − DIP|, 손끝 방향 u, 크롭 가운데 = TIP − 0.15L·u, 크롭 한 변 = 2.4L, **손끝이 크롭 위쪽**. 크롭 칸 (x, y) 의 가운데 → 원본 `p0 + a·(x+0.5) + b·(y+0.5)` 를 양선형으로 읽고, 원본 밖은 검정.

**뒤처리** (`pickNail`, `backProject`): 시그모이드 → 0.5 넘는 연결 성분 중 크롭 가운데 칸(가로 36~64%, 세로 30~78%)과 가장 많이 겹치는 것(크롭의 0.8% 넘게)만 남김(둘레 한 칸은 확률 유지) → 원본 픽셀마다 크롭 좌표로 되돌려 양선형으로 읽어 0.5 넘으면 손톱(0/1 마스크, 원본 크기).

**쓰는 곳**: 손톱이 보이는 손가락(손등 쪽, `nailQuads` vis > 0.3)만, 마스크 넓이가 손 관절점으로 본 손톱 넓이의 0.2~8배인 것만 쓴다. 못 찾은 손가락은 SAM 으로(앱에서는 생략 가능).

## 2. 타투 잉크 분할 — `tattoo-ink.onnx` (자체 학습)

| 항목 | 값 |
|---|---|
| 파일 · 크기 | `web/public/models/tattoo-ink.onnx`, 1.9MB (합성 8,000장 4 에포크, 검증 IoU 0.39. 학습을 이어 가며 나은 것으로 바꾼다) |
| 출처 · 라이선스 | IRIS 자체 학습(`tools/train/tattoo`): 타투 없는 사진 피부에 무작위 도안(OFL 글꼴 레터링 포함)을 합성한 자료. 손톱 모델과 같은 작은 U-Net |
| 입력 | `image` float32 `[1, 3, H, W]`, RGB 0~1, **H·W 는 8의 배수면 아무 크기**(완전 합성곱) |
| 출력 | `logit` float32 `[1, 1, H, W]` |
| core | `web/src/core/tattoo-ink.ts` — `tattooInkProb(img, runner, maxSide)` |

**앞처리** (`toCHW`): 긴 변을 maxSide(웹 1024, 폰 768 권장)로 넓이 평균해 줄이고, 8의 배수로 오른쪽·아래를 검정으로 채운다.
**뒤처리** (`upsample`): 채운 칸을 뺀 유효 영역만 원래 크기로 양선형으로 키우며 시그모이드 → 잉크 확률(0~1, 원본 크기).
**쓰는 곳**: 도안 뽑기(`tattoo-extract.ts`)의 `InkSource.inkProb` — 확률 0.6 넘는 곳은 잉크로 더하고, 덩어리 평균 0.06 아래는 빼고, 0.5 넘는 덩어리는 몸 윤곽선으로 보지 않는다.

### 손등 피부 (모델 아님) — `core/hand-skin.ts`
손만 찍은 사진에 타투를 굽기 전에, 몸 분할(MediaPipe selfie multiclass)의 몸 피부 확률에 손등 피부를 더한다(`handBackSkin(img, handPoints, bodySkin)`): 손 관절점 0·1·2·5·9·13·17 의 볼록 껍질을 1.15배로 키운 다각형 안에서, 밝기 합 120 넘는 픽셀의 색도(r/(r+g+b), g/(r+g+b)) 가운데값과의 거리가 0.025 아래면 1, 0.05 위면 0.

## 3. 머리카락 매팅 — `hair-matte-256.onnx` / `hair-matte-384.onnx`

| 항목 | 값 |
|---|---|
| 파일 · 크기 | `web/public/models/hair-matte-256.onnx` 1.6MB (384 도 1.6MB) |
| 출처 | IRIS 자체 학습(`tools/train/hair`, CelebAMask-HQ 라벨 + 선생님 모델 정밀화) |
| 입력 | `image` float32 `[n, 3, 256, 256]`(또는 384), RGB 0~1, 머리 둘레를 잘라 정사각형으로 |
| 출력 | `logit` float32 `[n, 1, N, N]` → 시그모이드 = 머리카락 알파 |
| 코드 | `web/src/engine/hairnet.ts`(실시간 실험용) — core 로 옮길 예정 |

## 4. 얼굴 분할 — `face-parse-512.onnx`

| 항목 | 값 |
|---|---|
| 파일 · 크기 | `web/public/models/face-parse-512.onnx` 26MB (폰은 fp16·int8 양자화본 7~13MB 로 바꿀 예정) |
| 출처 · 라이선스 | BiSeNet 얼굴 파싱(zllrunning/face-parsing.PyTorch, MIT, CelebAMask-HQ) 변환(`tools/train/face_parse_export.py`) |
| 입력 | `image` float32 `[1, 3, 512, 512]`, RGB 0~1(정규화는 모델 안), 얼굴 둘레 정사각형(밖은 검정) |
| 출력 | `parts` float32 `[1, 4, 512, 512]` — 채널 0 머리카락, 1 얼굴(피부·눈·눈썹·코·입·귀), 2 목, 3 안경의 **확률**(0~1, 이미 소프트맥스) |
| 코드 | `web/src/beauty/face-parse.ts` — core 로 옮길 예정 |

## 5. SAM(누른 곳 오려 내기) — SlimSAM-77

- Transformers.js 가 Hugging Face `Xenova/slimsam-77-uniform`(Apache-2.0, q8 약 14MB)를 받아 쓴다(`web/src/beauty/sam.ts`). 인코더가 폰 CPU 에서 수 초라 **앱에서는 대비책으로만**(손톱은 1번 모델이 먼저). 앱은 MobileSAM·EfficientSAM ONNX 로 바꾸거나 생략.

## 6. MediaPipe Tasks (손·얼굴·포즈·몸 분할)

| 모델 | 파일 | 출력 |
|---|---|---|
| 손 | `hand_landmarker.task` | 손마다 21점(그림 비율 0~1 → 픽셀로 곱함), 왼/오른손 |
| 얼굴 | `face_landmarker.task` | 478점 + 표정 계수 |
| 포즈 | `pose_landmarker_{lite,full,heavy}.task` | 33점 + 보이는 정도(vis) |
| 몸 분할 | `selfie_multiclass_256x256.tflite` | 배경·머리카락·몸 피부·얼굴 피부·옷·기타 6채널 |

- 웹은 `@mediapipe/tasks-vision`. 앱은 공식 RN 모듈이 없어 **네이티브(iOS·Android) MediaPipe Tasks 를 Expo 모듈로 감싸거나**, `react-native-vision-camera` 프레임 처리기 플러그인을 쓴다. 출력 좌표 규칙(그림 픽셀, 손 21점 번호)은 웹과 같게 맞춘다.

## 7. 서버 쪽 (폰에서 못 도는 것)

- 생성형(AnyDoor · Stable-Makeup · HairFastGAN, `tools/realistic`)은 확산·GAN 모델이라 폰에서 돌릴 수 없다 → 앱은 서버 API 호출(또는 끔).
- 스타일 AI(FashionSigLIP · SigLIP-L 임베딩 + 헤드)는 AI 저장소 쪽 규격을 따른다.

## 바꿀 때 지킬 것
- 새 모델은 ONNX(opset 17, 표준 연산만), 입력 RGB 0~1 NCHW float32, 출력 로짓 하나. 가능하면 10MB 아래.
- 앞처리·뒤처리는 `web/src/core/` 에 순수 함수로(DOM·캔버스·fetch 금지), 단위 테스트(`web/tests/core-*.test.ts`)와 함께.
- 이 문서의 표(파일·크기·라이선스·입출력)를 함께 고친다.
