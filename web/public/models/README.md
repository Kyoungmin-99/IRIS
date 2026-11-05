# 자체 학습 모델 자리

- `hair-matte-256.onnx` / `hair-matte-384.onnx`: tools/train/hair 로 학습한 머리카락 매팅 모델. 주소 뒤에 `?hairmodel=models/hair-matte-256.onnx` 를 붙이면 쓴다.
- `nail-seg.onnx`: tools/train/nail 로 학습한 손톱 분할 모델(연결 예정).
- `face-parse-512.onnx`: 사진 모드 정밀 얼굴 분할(머리카락·얼굴·목·안경 확률). BiSeNet 얼굴 파싱(zllrunning/face-parsing.PyTorch, MIT) 공개 가중치를 `tools/train/face_parse_export.py` 로 변환(26MB).
- `nail-crop-128.onnx`: 사진 모드 내 손톱 찾기. 손가락마다 끝마디를 128×128 로 돌려 잘라 손톱만 오리는 분할 모델(MobileNetV3-small + U-Net, 5.5MB). 사용자의 AI 저장소(rudals906377/AI) `models/nail/nail_seg.onnx` 를 그대로 가져옴 — 학습 데이터 vpapenko/nails-segmentation-dataset(CC0-1.0) + 상업 이용 가능 자유 라이선스 손 사진(nd·nc 제외), 맨손톱·칠한 손톱 모두. 크롭 정의는 그 저장소 `nail.js` 의 CROP 과 같아야 한다(`src/beauty/nail-crop.ts`).
- 사진 모드 네일의 손톱 오려 내기(손톱 모델이 못 찾은 손가락·손 점이 없는 사진)는 SlimSAM(Xenova/slimsam-77-uniform, Apache-2.0, 양자화 약 14MB). 파일은 `public/style-ai/models/Xenova/slimsam-77-uniform/` 에 두어 사이트에서 받는다(Hugging Face 만 쓰면 429 로 막힐 수 있었음).
