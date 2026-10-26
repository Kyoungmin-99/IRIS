# 스타일 AI 분류기 파일 (복사본)

`rudals906377/AI` 저장소 커밋 `1e98992` 의 파일을 그대로 복사했다(같은 사용자의 저장소, MIT).

- `analyzer.js`, `taxonomy.js`, `encoders.js`, `describe.js` — 분류기 코드(`@huggingface/transformers` 는 index.html importmap 이 CDN 으로 연결)
- `embeddings/marqo-fashionSigLIP.json`, `heads/marqo-fashionSigLIP.json` — 미리 계산한 라벨 임베딩·분류 헤드

AI 저장소가 비공개로 바뀌어 jsDelivr(`cdn.jsdelivr.net/gh/...`)가 404 를 돌려주게 되어, 사이트 안에서 받도록 옮겼다.
비전 모델(Marqo/marqo-fashionSigLIP, 약 94MB)은 그대로 Hugging Face 에서 받는다. 분류기를 바꿀 때는 AI 저장소의 새 커밋에서 이 파일들을 함께 다시 복사한다.
