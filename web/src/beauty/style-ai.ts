// 뷰티 스타일 AI 분류기(별도 저장소 rudals906377/AI) 불러오기.
// AI 채팅이 FINDE-photos shared/models/style-r8 로 넘겨준 휴대폰용 r8 묶음(코드: AI 커밋 6d40a0a)을 public/style-ai 에 그대로 두고
// 같은 사이트에서 받는다(AI 저장소가 비공개라 jsDelivr 는 404).
// r8 의 헤드·kNN·모티브는 r8 에서 미세 조정한 비전 인코더의 임베딩 공간에서 학습됐다 → 인코더도 r8 것(public/style-ai/models, q8 94MB)을
// 써야 한다. Hugging Face 원본 인코더와 섞으면 오류 없이 결과만 틀린다(AI 채팅 안내). 한국 평가 433장 75.8 → 86.5점.
// 사진은 기기 밖으로 나가지 않는다(모델 파일만 내려받는다).
// index.html의 importmap이 '@huggingface/transformers'를 CDN 주소로 연결해야 한다.

import type { StyleAIResult } from './style-attributes.ts';

/** 분류기 파일 위치(사이트 안). 원본: FINDE-photos shared/models/style-r8 (code/web · embeddings · heads · Marqo 인코더) */
const AI_DIR = 'style-ai';
const MODEL = 'Marqo/marqo-fashionSigLIP';

interface Analyzer {
  analyze(source: ImageBitmap | HTMLCanvasElement | HTMLImageElement | Blob, opts?: { category?: string; topk?: number }): Promise<StyleAIResult>;
  dispose(): void;
}

let loading: Promise<Analyzer> | null = null;

export interface AIProgress {
  status?: string;
  file?: string;
  progress?: number;
}

/** 분류기를 (한 번만) 불러온다. onProgress로 내려받기 진행률을 알린다. */
export function loadStyleAI(onProgress?: (p: AIProgress) => void): Promise<Analyzer> {
  loading ??= (async () => {
    const AI_BASE = new URL(AI_DIR, document.baseURI).href;
    // 인코더는 사이트 안 r8 파일에서(로컬 먼저, 없는 모델(SAM 등)은 그대로 Hugging Face 에서)
    const tfId = '@huggingface/transformers';
    const tf = (await import(/* @vite-ignore */ tfId)) as { env: { allowLocalModels: boolean; localModelPath: string } };
    tf.env.allowLocalModels = true;
    tf.env.localModelPath = `${AI_BASE}/models/`;
    const json = (p: string) => fetch(`${AI_BASE}/${p}`).then((r) => (r.ok ? r.json() : undefined));
    const [mod, labelEmbeddings, heads, motifs] = await Promise.all([
      import(/* @vite-ignore */ `${AI_BASE}/analyzer.js`) as Promise<{ createAnalyzer: (o: Record<string, unknown>) => Promise<Analyzer> }>,
      json('embeddings/marqo-fashionSigLIP.json'),
      json('heads/marqo-fashionSigLIP.json'),
      json('embeddings/motifs-marqo-fashionSigLIP.json'),
    ]);
    // 휴대폰 설정(AI 앱과 같게): q8 하나, 추론 뒤 작업 메모리를 바로 돌려준다
    return mod.createAnalyzer({
      model: MODEL, dtype: 'q8', device: 'wasm', labelEmbeddings, heads, motifs, onProgress,
      sessionOptions: { enableCpuMemArena: false, enableMemPattern: false },
    });
  })();
  loading.catch(() => {
    loading = null;
  });
  return loading;
}

/** 사진 한 장을 분류한다. category를 주면 그 분야로 고정. */
export async function analyzeStyle(source: ImageBitmap | Blob, category: 'auto' | 'hair' | 'nail' | 'makeup' | 'tattoo' = 'auto', onProgress?: (p: AIProgress) => void): Promise<StyleAIResult> {
  const a = await loadStyleAI(onProgress);
  // RawImage.read는 Blob·URL·캔버스를 받는다. ImageBitmap은 캔버스로 바꿔 넘긴다
  let src: HTMLCanvasElement | Blob = source as Blob;
  if (source instanceof ImageBitmap) {
    const c = document.createElement('canvas');
    c.width = source.width;
    c.height = source.height;
    c.getContext('2d')!.drawImage(source, 0, 0);
    src = c;
  }
  return a.analyze(src, { category });
}
