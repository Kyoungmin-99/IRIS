# 스타일 AI 분류기 r8 (휴대폰용 marqo-fashionSigLIP)

AI 채팅이 FINDE-photos `shared/models/style-r8` 로 넘겨준 묶음을 그대로 둔다(2026-10-27).

- `analyzer.js taxonomy.js encoders.js describe.js trends.js motif.js motifs.js` — AI 저장소 커밋 6d40a0a 의 웹 코드. **taxonomy.js 를 고치지 말 것**(헤드 해시 b866f542 와 맞아야 학습 헤드를 쓴다).
- `heads/marqo-fashionSigLIP.json` — r8 학습 헤드·보정 온도·kNN·부위 헤드(6.7MB).
- `embeddings/marqo-fashionSigLIP.json` — 라벨 문장 임베딩, `embeddings/motifs-marqo-fashionSigLIP.json` — 타투 세부 모티브.
- `models/Marqo/marqo-fashionSigLIP/` — **r8 비전 인코더**(q8, 94,008,252 bytes, sha256 f907c69e…)와 설정 2개. 헤드와 한 쌍이라 Hugging Face 원본 인코더와 섞으면 결과가 조용히 틀린다.
  `web/src/beauty/style-ai.ts` 가 Transformers.js `env.localModelPath` 를 이 폴더로 둬 사이트 안에서 받는다.
- 점수(한국 평가 433장): 원본 인코더 + 옛 헤드 75.8 → r8 86.5 (헤어 86.6 · 네일 90.6 · 메이크업 81.5 · 타투 83.4).
- 라벨 합치기(r8): `매직 스트레이트`→`생머리`, `언더컷`→`투블럭컷`(헤드에서 옛 라벨은 거의 1위가 안 됨).
