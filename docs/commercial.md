# 상업용으로 쓸 수 있게 하기 (계획 · 진행표)

목표: IRIS 가 쓰는 모델·자료를 **모두 상업 이용이 허락된 것**으로 바꾼다(사용자 지시 2026-10-27: "무조건 추후에 상업용으로 사용할 수 있게").
지금 테스트 사이트에 있는 모델 중 일부는 상업 이용 근거가 없어 아래 순서로 바꾸거나 다시 학습한다.

## 원칙
1. **학습 사진은 상업 이용이 허락된 것만**: CC0 · 퍼블릭 도메인 · CC BY · CC BY-SA · Pexels/Pixabay 라이선스 · 직접 촬영(촬영자 동의) · 동의서를 받은 인물 사진.
   NC(비상업) · ND(변경금지) · '학술 목적' · 라이선스 미표기 · 출처 미상 사진은 상업용 학습에서 뺀다(시험 비교용으로만).
2. **사진마다 권리 기록**: FINDE-photos 비공개 장부 `rights/ledger.jsonl`(사진 sha256 기준: 출처 URL · 작성자 · 라이선스 · 라이선스 URL · 받은 날짜 · 인물 동의 · `commercial_ok`).
   공개 manifest 에는 넣지 않는다(사용자 결정 2026-10-27: 비공개 장부에 따로).
3. **모델마다 학습 자료 목록**: 이 문서 표와 `docs/models.md` 에 어떤 묶음으로 학습했는지 적고, `commercial_ok=true` 인 사진만 쓴 판을 '상업판'으로 따로 만든다.
4. **바탕 모델 · 코드 라이선스**도 상업 허락(Apache-2.0 · MIT · BSD · OpenRAIL-M 등)인지 확인한다.

## 구성 요소별 상태

| 구성 요소 | 지금 | 상업 이용 | 할 일 |
|---|---|---|---|
| MediaPipe 손 · 얼굴 · 포즈 · 셀피 분할 | 그대로 | 가능(Apache-2.0) | — |
| SlimSAM(누른 곳 오려 내기) | 그대로 | 가능(Apache-2.0) | — |
| onnxruntime-web · Transformers.js | 그대로 | 가능(MIT · Apache-2.0) | — |
| 타투 잉크 `tattoo-ink.onnx` | IRIS 합성 학습 | 도안·글꼴(OFL)은 가능, **배경 사진**(testdata colla2 · 손 사진) 권리 확인 필요 | 상업 허락 배경 사진(요청 2 · 7, 장부 `commercial_ok`)으로 합성 자료를 다시 만들어 재학습 |
| 손톱 분할 `nail-crop-128.onnx`(r3) | AI 채팅 r3 | 근거 부족(Kaggle 원 출처 미확인 · 수집 사진 가짜 정답) | IRIS 가 상업판 재학습: vpapenko 손톱 데이터(CC0) + 직접 촬영 손 + 장부 `commercial_ok` 손 사진(요청 1 · 9) |
| 스타일 분석 r8(인코더 · 헤드) | AI 채팅 r8 | 근거 부족(출처 기록 없는 학습 사진 포함) | AI 채팅이 장부 `commercial_ok` 사진만으로 상업판 재학습(바탕 Marqo-fashionSigLIP 은 Apache-2.0) |
| 얼굴 분할 `face-parse-512.onnx` | BiSeNet + CelebAMask-HQ | 불가(CelebAMask-HQ 비상업 연구용) | MediaPipe 얼굴 점 + 셀피 분할(Apache)로 대체하거나, 상업 허락 사진에 MediaPipe 결과를 선생님 삼아 자체 학습 |
| 머리카락 매팅 `hair-matte-*.onnx` | IRIS 학습(CelebAMask-HQ 라벨) | 불가(같은 이유) | 상업 허락 얼굴 사진(요청 5)에 MediaPipe 머리카락 분할을 선생님 삼아 다시 학습 |
| 생성 서버(tools/realistic): InsightFace inswapper | 연구용 | 불가(비상업) | 상업판에서 빼거나 상업 허락 모델로 교체 |
| 생성 서버: HairFastGAN · GFPGAN · Stable-Makeup · Stable-Hair | 코드 MIT/Apache | 가중치 학습 자료(FFHQ 등) 확인 필요 | 가중치별 라이선스 확인 → 불가면 상업판에서 뺌 |
| 시험 사진(web/public/testdata) | 시험 · 회귀용 | 앱에 넣지 않음 | 학습 배경으로 쓰는 것만 위 원칙 적용 |

## 순서
1. 장부 규칙을 코덱스에게 전달(요청 게시판 공통 규칙) → 새 수집분부터 `commercial_ok` 기록.
2. 상업 허락 사진이 모이는 대로: 타투 잉크(배경) → 손톱 분할 → 머리카락 매팅 순으로 IRIS 가 상업판 재학습, 지금 판과 같은 시험(손 h68 타투 12장 · 사용자 손 5장 손톱 · 코덱스 손 채점)으로 비교.
3. 얼굴 분할은 MediaPipe 대체를 먼저 시도(모델 없이도 되면 가장 간단).
4. AI 채팅의 스타일 분석 상업판이 나오면 r8 자리에 바꿔 끼움.
5. 모든 칸이 '가능'이 되면 상업판 빌드(비상업 모델이 하나도 안 들어간 빌드)를 따로 만든다.
