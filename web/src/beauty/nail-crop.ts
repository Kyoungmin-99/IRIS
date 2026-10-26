// 사진 모드 내 손톱 찾기: 손가락마다 끝마디(DIP → TIP)를 손끝이 위로 오게 128×128 로 돌려 잘라, 손톱 전용 분할 모델로 손톱만 오린다.
// 모델: public/models/nail-crop-128.onnx — AI 저장소(rudals906377/AI) models/nail/nail_seg.onnx 를 그대로 가져옴
// (MobileNetV3-small + U-Net, 5.5MB, 학습 데이터 vpapenko/nails-segmentation-dataset CC0-1.0 + 상업 이용 가능 자유 라이선스 손 사진,
// 맨손톱·칠한 손톱 모두). 크롭 정의는 그 저장소 nail.js 의 CROP 과 같아야 한다(크기 = 끝마디 길이 L 의 2.4배, 중심 = TIP 에서 DIP 쪽 0.15L).
// SAM 은 사진 전체에서 짧은 맨손톱을 손가락 끝마디째 잡기 쉬워(손 편 사진 5장에서 손톱 25개 중 9개) 손 점이 있으면 이 모델을 먼저 쓴다.

const ORT_VERSION = '1.30.0';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const MODEL_URL = 'models/nail-crop-128.onnx';
export const CROP = { size: 128, side: 2.4, cen: 0.15 };
/** 손가락 [끝, 끝마디 뿌리] 랜드마크 번호(엄지 → 새끼) */
export const TIPS: [number, number][] = [[4, 3], [8, 7], [12, 11], [16, 15], [20, 19]];

type P = { x: number; y: number };
export interface CropFrame {
  /** 크롭 (0,0) 의 원본 좌표와 크롭 x·y 한 칸의 원본 벡터: src = p0 + a·x + b·y */
  p0: P;
  a: P;
  b: P;
}

/** 끝·끝마디 뿌리 점 → 크롭 틀(위가 손끝) */
export function cropFrame(tip: P, dip: P): CropFrame | null {
  const ux = tip.x - dip.x, uy = tip.y - dip.y, L = Math.hypot(ux, uy);
  if (L < 4) return null;
  const u = { x: ux / L, y: uy / L }, v = { x: -u.y, y: u.x };
  const c = { x: tip.x - u.x * CROP.cen * L, y: tip.y - u.y * CROP.cen * L };
  const half = (CROP.side * L) / 2, k = (2 * half) / CROP.size;
  // 크롭 x → v 방향, 크롭 y(아래로) → 손끝 반대 방향
  return { p0: { x: c.x - v.x * half + u.x * half, y: c.y - v.y * half + u.y * half }, a: { x: v.x * k, y: v.y * k }, b: { x: -u.x * k, y: -u.y * k } };
}

/**
 * 크롭 확률(S×S) 중 이 손가락의 손톱 성분만(0.5 넘는 연결 성분 중 가운데 손끝 아래 칸과 가장 많이 겹치는 것, 크롭의 0.8% 넘게).
 * 반환: 그 성분만 남긴 확률(나머지 0), 없으면 null
 */
export function pickNail(prob: Float32Array, S = CROP.size): Float32Array | null {
  const N = S * S, lab = new Int32Array(N);
  let best = -1, bestHit = 0, nlab = 0;
  for (let i = 0; i < N; i++) {
    if (prob[i] <= 0.5 || lab[i]) continue;
    nlab++;
    let area = 0, hit = 0;
    const st = [i];
    lab[i] = nlab;
    while (st.length) {
      const k = st.pop()!, x = k % S, y = (k / S) | 0;
      area++;
      if (x >= S * 0.36 && x <= S * 0.64 && y >= S * 0.3 && y <= S * 0.78) hit++;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const X = x + dx, Y = y + dy;
        if (X < 0 || Y < 0 || X >= S || Y >= S) continue;
        const j = Y * S + X;
        if (prob[j] > 0.5 && !lab[j]) { lab[j] = nlab; st.push(j); }
      }
    }
    if (hit > bestHit && area >= N * 0.008) { bestHit = hit; best = nlab; }
  }
  if (best < 0) return null;
  // 성분 둘레 한 칸까지는 확률을 남겨 되돌릴 때 경계가 부드럽게
  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    if (lab[i] === best) { out[i] = prob[i]; continue; }
    const x = i % S, y = (i / S) | 0;
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const X = x + dx, Y = y + dy;
      if (X >= 0 && Y >= 0 && X < S && Y < S && lab[Y * S + X] === best) { out[i] = prob[i]; break; }
    }
  }
  return out;
}

/** 크롭 확률을 원본 크기 마스크(0/1)로: 원본 픽셀마다 크롭 좌표를 구해 양선형으로 읽고 0.5 넘으면 손톱 */
export function backProject(prob: Float32Array, f: CropFrame, W: number, H: number, S = CROP.size): Uint8Array {
  const mask = new Uint8Array(W * H);
  const corners = [[0, 0], [S, 0], [0, S], [S, S]].map(([x, y]) => ({ x: f.p0.x + f.a.x * x + f.b.x * y, y: f.p0.y + f.a.y * x + f.b.y * y }));
  const x0 = Math.max(0, Math.floor(Math.min(...corners.map((c) => c.x)))), x1 = Math.min(W - 1, Math.ceil(Math.max(...corners.map((c) => c.x))));
  const y0 = Math.max(0, Math.floor(Math.min(...corners.map((c) => c.y)))), y1 = Math.min(H - 1, Math.ceil(Math.max(...corners.map((c) => c.y))));
  const det = f.a.x * f.b.y - f.a.y * f.b.x;
  if (!det) return mask;
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const dx = x + 0.5 - f.p0.x, dy = y + 0.5 - f.p0.y;
    // 크롭 픽셀 가운데 기준
    const cx = (dx * f.b.y - dy * f.b.x) / det - 0.5, cy = (f.a.x * dy - f.a.y * dx) / det - 0.5;
    if (cx < 0 || cy < 0 || cx > S - 1 || cy > S - 1) continue;
    const xi = Math.min(S - 2, Math.floor(cx)), yi = Math.min(S - 2, Math.floor(cy)), fx = cx - xi, fy = cy - yi;
    const p = (prob[yi * S + xi] * (1 - fx) + prob[yi * S + xi + 1] * fx) * (1 - fy) + (prob[(yi + 1) * S + xi] * (1 - fx) + prob[(yi + 1) * S + xi + 1] * fx) * fy;
    if (p > 0.5) mask[y * W + x] = 1;
  }
  return mask;
}

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

/** 손마다 다섯 손가락의 손톱 마스크(원본 크기). 못 찾은 손가락은 빠진다. finger: 0 엄지 ~ 4 새끼, dir: 끝마디 방향(뿌리 → 끝) */
export async function nailCropMasks(img: HTMLCanvasElement, hands: { p: P[] }[]): Promise<{ hand: number; finger: number; mask: Uint8Array; dir: P }[]> {
  const S = CROP.size, W = img.width, H = img.height;
  const jobs: { hand: number; finger: number; f: CropFrame; dir: P }[] = [];
  hands.forEach((hp, hi) => TIPS.forEach(([t, d], fi) => {
    const f = cropFrame(hp.p[t], hp.p[d]);
    if (!f) return;
    const L = Math.hypot(hp.p[t].x - hp.p[d].x, hp.p[t].y - hp.p[d].y);
    jobs.push({ hand: hi, finger: fi, f, dir: { x: (hp.p[t].x - hp.p[d].x) / L, y: (hp.p[t].y - hp.p[d].y) / L } });
  }));
  if (!jobs.length) return [];
  const { ort, session } = await load();
  const cc = document.createElement('canvas');
  cc.width = S;
  cc.height = S;
  const g = cc.getContext('2d', { willReadFrequently: true })!;
  const input = new Float32Array(jobs.length * 3 * S * S);
  jobs.forEach(({ f }, i) => {
    // 크롭 → 원본 변환의 역으로 원본을 그린다(바깥은 검정: 학습 크롭과 같게)
    const det = f.a.x * f.b.y - f.a.y * f.b.x;
    const ia = f.b.y / det, ib = -f.a.y / det, ic = -f.b.x / det, id = f.a.x / det;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.fillStyle = '#000';
    g.fillRect(0, 0, S, S);
    g.setTransform(ia, ib, ic, id, -(ia * f.p0.x + ic * f.p0.y), -(ib * f.p0.x + id * f.p0.y));
    g.drawImage(img, 0, 0);
    const px = g.getImageData(0, 0, S, S).data;
    for (let j = 0; j < S * S; j++) for (let c = 0; c < 3; c++) input[i * 3 * S * S + c * S * S + j] = px[j * 4 + c] / 255;
  });
  const out = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', input, [jobs.length, 3, S, S]) });
  const logit = out[session.outputNames[0]].data;
  const res: { hand: number; finger: number; mask: Uint8Array; dir: P }[] = [];
  jobs.forEach((j, i) => {
    const prob = new Float32Array(S * S);
    for (let k = 0; k < S * S; k++) prob[k] = 1 / (1 + Math.exp(-logit[i * S * S + k]));
    const nail = pickNail(prob, S);
    if (!nail) return;
    const mask = backProject(nail, j.f, W, H, S);
    if (mask.some((v) => v)) res.push({ hand: j.hand, finger: j.finger, mask, dir: j.dir });
  });
  return res;
}
