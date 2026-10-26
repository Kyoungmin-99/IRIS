// 타투 잉크 분할 모델(자체 학습, tools/train/tattoo): 고른 사진 → 픽셀마다 '타투 잉크' 확률.
// 공개된 타투 분할 모델·상업 이용 가능한 데이터가 없어, 타투 없는 사진의 피부에 무작위 도안(가는 선·점묘·블랙워크·명암·컬러·한글/영문 레터링)을
// 합성한 자료로 학습했다(눈썹·속눈썹·머리카락·옷 주름·손톱은 잉크 아님으로 배움). 피부 분할이 맨살을 옷으로 잘못 읽어도 잉크를 잡는다.
// 모델: public/models/tattoo-ink.onnx(작은 U-Net, 완전 합성곱이라 크기 자유), onnxruntime-web(WASM).

const ORT_VERSION = '1.30.0';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const MODEL_URL = 'models/tattoo-ink.onnx';

type Session = { inputNames: string[]; outputNames: string[]; run(feeds: Record<string, unknown>): Promise<Record<string, { data: Float32Array; dims: number[] }>> };
type Ort = { env: { wasm: { wasmPaths: string } }; InferenceSession: { create(url: string, opts: { executionProviders: string[] }): Promise<Session> }; Tensor: new (type: 'float32', data: Float32Array, dims: number[]) => unknown };
let loading: Promise<{ ort: Ort; session: Session }> | null = null;
function load() {
  loading ??= (async () => {
    const ort = (await import(/* @vite-ignore */ `${ORT_BASE}ort.webgpu.bundle.min.mjs`)) as Ort;
    ort.env.wasm.wasmPaths = ORT_BASE;
    const url = new URL(MODEL_URL, document.baseURI).href;
    return { ort, session: await ort.InferenceSession.create(url, { executionProviders: ['wasm'] }) };
  })();
  loading.catch(() => {
    loading = null;
  });
  return loading;
}

/** 그림(RGBA, w×h) → 잉크 확률(0~1, w×h). 긴 변 maxSide 로 줄여(8의 배수로 채워) 한 번에 돌리고 되키운다 */
export async function tattooInkProb(rgba: Uint8ClampedArray, w: number, h: number, maxSide = 768): Promise<Float32Array> {
  const { ort, session } = await load();
  const s = Math.min(1, maxSide / Math.max(w, h));
  const sw = Math.max(8, Math.round(w * s)), sh = Math.max(8, Math.round(h * s));
  const W = Math.ceil(sw / 8) * 8, H = Math.ceil(sh / 8) * 8;
  const x = new Float32Array(3 * W * H);
  // 넓이 평균으로 줄이기(가는 선이 끊기지 않게)
  for (let y = 0; y < sh; y++) for (let xx = 0; xx < sw; xx++) {
    const x0 = Math.floor((xx * w) / sw), x1 = Math.max(x0 + 1, Math.floor(((xx + 1) * w) / sw));
    const y0 = Math.floor((y * h) / sh), y1 = Math.max(y0 + 1, Math.floor(((y + 1) * h) / sh));
    let r = 0, g = 0, b = 0, n = 0;
    for (let yy = y0; yy < y1; yy++) for (let q = x0; q < x1; q++) {
      const o = (yy * w + q) * 4;
      r += rgba[o]; g += rgba[o + 1]; b += rgba[o + 2]; n++;
    }
    const i = y * W + xx;
    x[i] = r / n / 255;
    x[W * H + i] = g / n / 255;
    x[2 * W * H + i] = b / n / 255;
  }
  const out = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', x, [1, 3, H, W]) });
  const logit = out[session.outputNames[0]].data;
  const p = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const fy = Math.min(sh - 1.001, Math.max(0, ((y + 0.5) * sh) / h - 0.5));
    const yi = Math.floor(fy), ty = fy - yi;
    for (let xx = 0; xx < w; xx++) {
      const fx = Math.min(sw - 1.001, Math.max(0, ((xx + 0.5) * sw) / w - 0.5));
      const xi = Math.floor(fx), tx = fx - xi;
      const v = (logit[yi * W + xi] * (1 - tx) + logit[yi * W + xi + 1] * tx) * (1 - ty) + (logit[(yi + 1) * W + xi] * (1 - tx) + logit[(yi + 1) * W + xi + 1] * tx) * ty;
      p[y * w + xx] = 1 / (1 + Math.exp(-v));
    }
  }
  return p;
}
