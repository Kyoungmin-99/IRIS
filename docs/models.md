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
| 출처 · 라이선스 | AI 저장소 손톱 분할 **r3**(`training-handoff` 가지 `data/nail_seg_r3/nail_seg_r3.onnx`, 2026-10-27 교체; 그 전은 `models/nail/nail_seg.onnx`). MobileNetV3-small(torchvision BSD-3 초기값) + U-Net. 학습 데이터 vpapenko/nails-segmentation-dataset(CC0-1.0) + 기존 수집 사진 가짜 정답 + Kaggle Nail Segmentation Dataset V2(표기 CC0, 원 출처 Roboflow 로 알려짐 — 미확인) 1,220장. AI 쪽 평가(손톱 1,819개): IoU 0.670 → 0.715, 엄지 0.49 → 0.62 |
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
| 파일 · 크기 | `web/public/models/tattoo-ink.onnx`, 7.8MB — **2026-10-29 판: 5단계 U-Net**(채널 16·32·64·128·256, 매개변수 약 194만 · 4단계 판 가중치를 옮겨 이어 학습(`tools/train/tattoo/unet.py`), 빽빽한 선생님 정답 상자는 '모름'(`labels.py`), `export.py --calib 0.85`). 4단계(약 50만)는 반복 학습 6바퀴째 IoU 0.51 에서 멈추고 넓은 음영 도안을 놓쳤다. (이전) 2026-10-28 판: 합성(ep4)에서 이어 **실제 타투 사진 선생님 정답**(아래) + 라벨 붙은 합성 사진 + 맨살 손·팔 조각으로 학습, `export.py --calib 0.9`(학습 모델 확률 0.9 를 앱에서 0.5 로 읽게 출력 logit 에 log 9 를 뺌: 다리털 같은 오검출을 줄이려고) |
| 실제 사진 점수 | (10-29 5단계 판, 시험 244장 · 빽빽한 정답 뺌) IoU 0.507 → 0.539, 사진 평균 0.496 → 0.543, 상자 밖 오검출 0.76 → 0.31%, 맨살 팔 28장 오검출 4 → 2장. 손 12 · 전신 12: '가족' 글씨 깔끔, 굵은 먹 십자가는 도안 뽑기 고침(아래 쓰는 곳)과 함께 거의 같음(교차점 아래 아주 옅은 홈), 나머지 같음. (10-28 저녁 판, 학습 반복 2바퀴: 사진 직접 수집 + 선생님 정답 2,247장) 시험 사진 232장 IoU 0.478 → 0.499(사진 평균 0.487 → 0.517), 상자 밖 오검출 0.82 → 0.56%. 손 12장: 색 꽃 하트가 장미까지 온전, 나머지 같음. 전신 12장: 기타 도안 깨끗, 입술 타투는 앱 쪽 고침(피부 위만 셈)과 함께 같음 이상. (이전 판) 선생님 정답 시험 사진 122장 IoU(문턱 0.9 기준) 0.08 → 0.56, 타투 팔 검출 77 → 92%, 타투 사진 검출 89 → 99%. 손 h68 × 타투 12장: 색 꽃 초승달이 선 몇 가닥 → 색 꽃 전체, 나비·나침반 더 온전, 나머지 같음. 굵고 꽉 찬 먹(십자가)은 여전히 약해 규칙(거르기 풀기)이 살린다 |
| 선생님 정답 | `tools/train/tattoo/teacher.py`: Grounding DINO tiny 'a tattoo.' 상자 → SigLIP2 로 상자 안 다시 판정(로션·털·점 거름) → SAM 2.1 tiny 영역 → 영역 안 잉크 픽셀(블랙햇 · 피부색 차이). 모두 Apache-2.0. `crops.py` 로 256 조각(시험 전용 사진 제외), `eval_real.py` 로 점수 |
| 출처 · 라이선스 | IRIS 자체 학습(`tools/train/tattoo`): 타투 없는 사진 피부에 무작위 도안(OFL 글꼴 레터링 포함)을 합성한 자료. 손톱 모델과 같은 작은 U-Net. 배경 사진은 IRIS testdata 수집 사진(colla2)·사용자 손 사진·코덱스 맨살 팔 사진 → 그 사진들의 권리 기록을 따름. 2026-10-27 부터 연구용 판은 실제 타투 도안 그림(Hugging Face `Drozdik/tattoo_v0` 4,370장, CC BY-NC-SA 4.0 — **연구용 판만**)을 절반 섞어 합성 |
| 입력 | `image` float32 `[1, 3, H, W]`, RGB 0~1, **H·W 는 16의 배수면 아무 크기**(완전 합성곱, 5단계라 4번 반으로 줄임) |
| 출력 | `logit` float32 `[1, 1, H, W]` |
| core | `web/src/core/tattoo-ink.ts` — `tattooInkProb(img, runner, maxSide)` |

**앞처리** (`toCHW`): 긴 변을 maxSide(웹 1024, 폰 768 권장)로 넓이 평균해 줄이고, 16의 배수로 오른쪽·아래를 검정으로 채운다.
**뒤처리** (`upsample`): 채운 칸을 뺀 유효 영역만 원래 크기로 양선형으로 키우며 시그모이드 → 잉크 확률(0~1, 원본 크기).
**쓰는 곳**: 도안 뽑기(`tattoo-extract.ts`)의 `InkSource.inkProb` — 확신 잉크(0.6 넘음)가 처음 영역 밖에 절반 넘으면 `sureRect`(옷·배경 위는 뺀 확신 잉크 무리 상자)로 다시 고르고, 확신 잉크와 이어진 덩어리(규칙 잉크 + 확률 0.3 넘음)는 확률 0.1~0.3 도 잉크로(굵은 먹 막대 가운데), 확률 0.6 넘는 곳은 잉크로 더하고, 덩어리 평균 0.06 아래는 빼고, 0.5 넘는 덩어리는 몸 윤곽선으로 보지 않는다.

### 손등 피부 (모델 아님) — `core/hand-skin.ts`
손만 찍은 사진에 타투를 굽기 전에, 몸 분할(MediaPipe selfie multiclass)의 몸 피부 확률에 손등 피부를 더한다(`handBackSkin(img, handPoints, bodySkin)`): 손 관절점 0·1·2·5·9·13·17 의 볼록 껍질을 1.15배로 키운 다각형 안에서, 밝기 합 120 넘는 픽셀의 색도(r/(r+g+b), g/(r+g+b)) 가운데값과의 거리가 0.025 아래면 1, 0.05 위면 0.

## 3. 머리카락 매팅 — `hair-matte-256.onnx` / `hair-matte-384.onnx`

| 항목 | 값 |
|---|---|
| 파일 · 크기 | `web/public/models/hair-matte-256.onnx` 1.6MB (384 도 1.6MB) |
| 출처 · 권리 | IRIS 자체 학습(`tools/train/hair`, CelebAMask-HQ 라벨 + 선생님 모델 정밀화). CelebAMask-HQ 는 **비상업 연구용** → 상업 앱에는 다시 학습 필요 |
| 입력 | `image` float32 `[n, 3, 256, 256]`(또는 384), RGB 0~1, 머리 둘레를 잘라 정사각형으로 |
| 출력 | `logit` float32 `[n, 1, N, N]` → 시그모이드 = 머리카락 알파 |
| 코드 | `web/src/engine/hairnet.ts`(실시간 실험용) — core 로 옮길 예정 |

## 4. 얼굴 분할 — `face-parse-512.onnx`

| 항목 | 값 |
|---|---|
| 파일 · 크기 | `web/public/models/face-parse-512.onnx` 26MB (폰은 fp16·int8 양자화본 7~13MB 로 바꿀 예정) |
| 출처 · 라이선스 | BiSeNet 얼굴 파싱(zllrunning/face-parsing.PyTorch, 코드 MIT) 변환(`tools/train/face_parse_export.py`). 가중치는 CelebAMask-HQ(**비상업 연구용**)로 학습 → 상업 앱에는 다른 모델·재학습 필요 |
| 입력 | `image` float32 `[1, 3, 512, 512]`, RGB 0~1(정규화는 모델 안), 얼굴 둘레 정사각형(밖은 검정) |
| 출력 | `parts` float32 `[1, 4, 512, 512]` — 채널 0 머리카락, 1 얼굴(피부·눈·눈썹·코·입·귀), 2 목, 3 안경의 **확률**(0~1, 이미 소프트맥스) |
| 코드 | `web/src/beauty/face-parse.ts` — core 로 옮길 예정 |

## 5. SAM(누른 곳 오려 내기) — SlimSAM-77

- Transformers.js 가 Hugging Face `Xenova/slimsam-77-uniform`(Apache-2.0, q8 약 14MB)를 받아 쓴다(`web/src/beauty/sam.ts`). 인코더가 폰 CPU 에서 수 초라 **앱에서는 대비책으로만**(손톱은 1번 모델이 먼저). 앱은 MobileSAM·EfficientSAM ONNX 로 바꾸거나 생략.

## 5-1. 스타일 AI 분류기 — r8 (AI 채팅)

| 항목 | 값 |
|---|---|
| 파일 · 크기 | `web/public/style-ai/`(코드 7개 · 헤드 6.7MB · 임베딩 · 모티브) + `style-ai/models/Marqo/marqo-fashionSigLIP/onnx/vision_model_quantized.onnx` 94MB(q8, opset 14) |
| 출처 | AI 채팅 r8(FINDE-photos `shared/models/style-r8`). 바탕 Marqo/marqo-fashionSigLIP(Apache-2.0)를 미세 조정 |
| 입력 | `pixel_values` float32 `[N,3,224,224]` = (픽셀/255 − 0.5)/0.5, RGB. 224 줄이기는 canvas 방식(`resizeBilinearQ16`) — bicubic 이면 1위가 바뀜. 사진은 먼저 긴 변 896 이하로 |
| 출력 | `image_embeds` `[N,768]` → 헤드·kNN·제로샷 → `category`(hair·nail·makeup·tattoo)와 그룹별 `label`·`score` |
| 앱 | AI 채팅 문서 `shared/models/style-r8/code/docs/MODEL_IO.md` 와 `code/core/`(DOM 없음, onnxruntime 주입) 를 그대로 쓴다. 인코더와 헤드는 한 쌍 |

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

## 앱에 포즈 33점이 없을 때 (코덱스 답 2026-10-27: 앱 네이티브 모듈에 얼굴·손 점·셀피/머리카락 분할은 있고 포즈는 없음)
- 된다: 네일(손 21점 + 손톱 모델), 손등 타투(손 21점 + 셀피 분할), 메이크업(얼굴 478점), 헤어 염색(머리카락 분할 + 얼굴 점), 사용자가 자리를 직접 고르는 실시간 타투.
- 안 된다: 팔·다리·몸통 타투를 고른 사진과 같은 자리에 맞춰 놓기(팔 마디·어깨·엉덩이 점이 필요).

## 바꿀 때 지킬 것
- 새 모델은 ONNX(opset 17, 표준 연산만), 입력 RGB 0~1 NCHW float32, 출력 로짓 하나. 가능하면 10MB 아래.
- 앞처리·뒤처리는 `web/src/core/` 에 순수 함수로(DOM·캔버스·fetch 금지), 단위 테스트(`web/tests/core-*.test.ts`)와 함께.
- 이 문서의 표(파일·크기·라이선스·입출력)를 함께 고친다.
