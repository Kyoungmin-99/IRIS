// 모델별 인코더 어댑터 — CLIP 계열과 SigLIP 계열의 차이를 여기서 흡수한다.
//   CLIP   : CLIP*ModelWithProjection → image_embeds / text_embeds
//   SigLIP : Siglip*Model → pooler_output, 텍스트는 64 토큰 고정 길이로 패딩해야 한다
import {
  AutoTokenizer,
  AutoProcessor,
  CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection,
  SiglipTextModel,
  SiglipVisionModel,
  RawImage,
} from '@huggingface/transformers';

export const familyOf = (model) => (/siglip/i.test(model) ? 'siglip' : 'clip');

export async function loadVisionEncoder(model, { device, dtype, progress_callback } = {}) {
  const fam = familyOf(model);
  const opts = { progress_callback };
  if (device) opts.device = device;
  if (dtype) opts.dtype = dtype;
  const Vision = fam === 'siglip' ? SiglipVisionModel : CLIPVisionModelWithProjection;
  const [processor, vision] = await Promise.all([
    AutoProcessor.from_pretrained(model, { progress_callback }),
    Vision.from_pretrained(model, opts),
  ]);
  async function embed(image) {
    const inputs = await processor(image);
    const out = await vision(inputs);
    return pickEmbeds(out, ['image_embeds', 'pooler_output']).tolist().map(normalize);
  }
  return {
    family: fam,
    // images: RawImage 1장 또는 배열 → 정규화된 벡터 배열
    async embed(images) {
      const list = Array.isArray(images) ? images : [images];
      const out = [];
      for (const im of list) out.push(...(await embed(im)));
      return out;
    },
    dispose: () => vision.dispose?.(),
  };
}

export async function loadTextEncoder(model, { device, dtype, progress_callback } = {}) {
  const fam = familyOf(model);
  const opts = { progress_callback };
  if (device) opts.device = device;
  if (dtype) opts.dtype = dtype;
  const Text = fam === 'siglip' ? SiglipTextModel : CLIPTextModelWithProjection;
  const [tokenizer, text] = await Promise.all([
    AutoTokenizer.from_pretrained(model, { progress_callback }),
    Text.from_pretrained(model, opts),
  ]);
  return {
    family: fam,
    async embed(texts) {
      const out = [];
      for (let i = 0; i < texts.length; i += 32) {
        const batch = texts.slice(i, i + 32);
        const inputs = fam === 'siglip'
          ? tokenizer(batch, { padding: 'max_length', max_length: 64, truncation: true })
          : tokenizer(batch, { padding: true, truncation: true });
        const o = await text(inputs);
        out.push(...pickEmbeds(o, ['text_embeds', 'pooler_output']).tolist().map(normalize));
      }
      return out;
    },
    dispose: () => text.dispose?.(),
  };
}

// 모델마다 출력 이름이 다르다 (예: Marqo 는 SigLIP 이지만 text_embeds 로 내보낸다)
function pickEmbeds(out, names) {
  for (const n of names) if (out[n]) return out[n];
  throw new Error(`임베딩 출력을 찾을 수 없습니다: ${Object.keys(out).join(', ')}`);
}

// 좌우 반전 (테스트 타임 증강용)
export function flipHorizontal(image) {
  const { width, height, channels, data } = image;
  const out = new data.constructor(data.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const src = (y * width + x) * channels;
      const dst = (y * width + (width - 1 - x)) * channels;
      for (let c = 0; c < channels; c++) out[dst + c] = data[src + c];
    }
  }
  return new RawImage(out, width, height, channels);
}

export function normalize(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  const n = Math.sqrt(s) || 1;
  return v.map((x) => x / n);
}
