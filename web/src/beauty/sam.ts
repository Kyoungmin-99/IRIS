// 눌러서 오려 내기(Segment Anything, SlimSAM-77, Apache-2.0, 양자화 약 14MB): 사진 모드에서 손톱을 찾는다.
// 그림 한 장의 특징을 한 번 계산(수 초)해 두고, 점 하나마다 0.3초 안팎으로 그 물체의 마스크를 얻는다.
// Transformers.js 가 Hugging Face 에서 받아 브라우저에 캐시한다(사진은 기기 밖으로 나가지 않음).

const MODEL = 'Xenova/slimsam-77-uniform';

type Tensor = { data: ArrayLike<number> & { [i: number]: number }; dims: number[] };
type TF = {
  SamModel: { from_pretrained(id: string, o?: Record<string, unknown>): Promise<SamModelT> };
  AutoProcessor: { from_pretrained(id: string): Promise<ProcessorT> };
  RawImage: { fromCanvas(c: HTMLCanvasElement | OffscreenCanvas): Promise<unknown> };
};
type ProcessorT = ((image: unknown, o?: Record<string, unknown>) => Promise<Record<string, Tensor>>) & {
  post_process_masks(masks: Tensor, orig: Tensor, reshaped: Tensor): Promise<Tensor[]>;
};
type SamModelT = ((inputs: Record<string, unknown>) => Promise<{ pred_masks: Tensor; iou_scores: Tensor }>) & {
  get_image_embeddings(inputs: Record<string, Tensor>): Promise<Record<string, unknown>>;
};

let loading: Promise<{ tf: TF; model: SamModelT; processor: ProcessorT }> | null = null;
function load() {
  loading ??= (async () => {
    const id = '@huggingface/transformers';
    const tf = (await import(/* @vite-ignore */ id)) as unknown as TF;
    const [model, processor] = await Promise.all([tf.SamModel.from_pretrained(MODEL, { dtype: 'q8' }), tf.AutoProcessor.from_pretrained(MODEL)]);
    return { tf, model, processor };
  })();
  loading.catch(() => {
    loading = null;
  });
  return loading;
}

export interface SamImage {
  w: number;
  h: number;
  /** 점(그림 픽셀) → 마스크(0/1, w×h)와 점수 */
  /** alts: 고르지 않은 나머지 후보(넓이 오름차순). size 를 주면 그 크기로 줄인 마스크(큰 사진을 여러 점 훑을 때: 되키우기가 대부분의 시간) */
  segment(x: number, y: number, size?: { w: number; h: number }): Promise<{ mask: Uint8Array; score: number; alts: { mask: Uint8Array; score: number; area: number }[] }>;
}

/** 그림(캔버스) 하나의 특징을 계산해 둔다 */
export async function samImage(canvas: HTMLCanvasElement): Promise<SamImage> {
  const { tf, model, processor } = await load();
  const image = await tf.RawImage.fromCanvas(canvas);
  const first = await processor(image, { input_points: [[[canvas.width / 2, canvas.height / 2]]] });
  const emb = await model.get_image_embeddings(first);
  // 처리기가 점을 옮긴 비율(그림 → 1024 크기)
  const fx = first.input_points.data[0] / (canvas.width / 2), fy = first.input_points.data[1] / (canvas.height / 2);
  const PointTensor = first.input_points.constructor as new (type: string, data: Float32Array, dims: number[]) => Tensor;
  const w = canvas.width;
  const h = canvas.height;
  return {
    w,
    h,
    async segment(x: number, y: number, size?: { w: number; h: number }) {
      // 점 좌표만 처리기와 같은 비율로 바꾼다: 점마다 처리기를 부르면 사진 전체를 다시 줄이고 정규화해 한 점에 수백 ms 가 들었다
      const input_points = new PointTensor('float32', Float32Array.of(x * fx, y * fy), first.input_points.dims);
      const out = await model({ ...emb, input_points, input_labels: first.input_labels });
      const masks = await processor.post_process_masks(out.pred_masks, (size ? [[size.h, size.w]] : first.original_sizes) as unknown as Tensor, first.reshaped_input_sizes);
      const sc = Array.from(out.iou_scores.data as ArrayLike<number>);
      // 세 후보(작은 부분 → 큰 덩어리) 중 손톱 하나: 점수가 가장 좋은 것에 가까운(0.08 안) 것 가운데 가장 작은 것.
      // 가장 좋은 점수만 고르면 손톱+손가락 전체 같은 큰 덩어리가 자주 이긴다. 사진의 8% 넘는 덩어리는 깎는다
      const m = masks[0].data;
      const n = size ? size.w * size.h : w * h;
      const cand: { k: number; area: number; s: number }[] = [];
      for (let k = 0; k < sc.length; k++) {
        let area = 0;
        for (let i = 0; i < n; i++) if (m[k * n + i]) area++;
        if (area > 20) cand.push({ k, area, s: sc[k] - (area / n > 0.08 ? 0.5 : 0) });
      }
      const top = Math.max(-1, ...cand.map((c) => c.s));
      let best = -1;
      let bestScore = -1;
      let bestArea = Infinity;
      for (const c of cand) {
        if (c.s >= top - 0.08 && c.area < bestArea) {
          best = c.k;
          bestScore = c.s;
          bestArea = c.area;
        }
      }
      const mask = new Uint8Array(n);
      if (best >= 0) for (let i = 0; i < n; i++) mask[i] = m[best * n + i] ? 1 : 0;
      const alts = cand.filter((c) => c.k !== best).sort((p, q) => p.area - q.area).map((c) => {
        const am = new Uint8Array(n);
        for (let i = 0; i < n; i++) am[i] = m[c.k * n + i] ? 1 : 0;
        return { mask: am, score: c.s, area: c.area };
      });
      return { mask, score: bestScore, alts };
    },
  };
}
