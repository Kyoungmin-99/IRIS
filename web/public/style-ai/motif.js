// 타투 세부 모티브 판단 (장미 · 늑대 · 나침반 …) — embeddings/motifs-<모델>.json 을 읽어 쓴다
//
//   const judge = createMotifJudge(json);            // json: build-motifs.mjs + train-motifs.py 결과
//   judge(vec, subjectDist) → [{ ko, parent, score, level }] (확률 높은 순 3개)
//     vec: 사진 임베딩(정규화) · subjectDist: tattoo '도안' 그룹 확률 (taxonomy 라벨 순서)
//
// 판단 = 헤드(선형 α + 문장 유사도 1−α) × mix + 제로샷(문장 유사도 × 부모 도안 확률^w) × (1−mix), 그다음 비슷한 라벨 사진 kNN 을 β 만큼 섞는다.
// 학습 정보(head · knn)가 없으면 제로샷 × 부모 도안 확률만 쓴다.
import { MOTIFS } from './motifs.js';
import { TAXONOMY } from './taxonomy.js';

const SUBJECTS = TAXONOMY.tattoo.groups.find((g) => g.key === 'subject').labels.map((l) => l.ko);
const LEVEL = (p) => (p >= 0.55 ? 'high' : p >= 0.3 ? 'mid' : 'low');

export function createMotifJudge(json) {
  if (!json?.data || !json.names) return null;
  const f32 = (b64) => { const u8 = b64bytes(b64); return new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4); };
  const dim = json.dim, n = json.count;
  const T = f32(json.data);
  const byName = new Map(MOTIFS.map((m) => [m.ko, m]));
  const meta = json.names.map((ko) => byName.get(ko) || { ko, parent: null });
  const pidx = meta.map((m) => SUBJECTS.indexOf(m.parent));
  const P = { scale: 50, w: 0.5, mix: 0.6, beta: 0.35, k: 10, tau: 20, alpha: 0.5, ...(json.params || {}) };
  const head = json.head ? { W: f32(json.head.W), b: json.head.b } : null;
  const knn = json.knn ? { q: new Int8Array(b64bytes(json.knn.q).buffer.slice(0)), scale: json.knn.scale, labels: json.knn.labels } : null;
  const rowDot = (M, i, v) => { let s = 0; const o = i * dim; for (let j = 0; j < dim; j++) s += M[o + j] * v[j]; return s; };

  return function judge(vec, subjectDist = null) {
    const sims = Array.from({ length: n }, (_, i) => rowDot(T, i, vec));
    let zs = softmax(sims.map((s) => s * P.scale));
    if (subjectDist) zs = norm(zs.map((p, i) => p * Math.pow(Math.max(subjectDist[pidx[i]] ?? 0, 1e-6), P.w)));
    let p = zs;
    if (head) {
      const hp = softmax(head.b.map((b, i) => rowDot(head.W, i, vec) + b));
      const z100 = softmax(sims.map((s) => s * 100));
      const ph = hp.map((x, i) => P.alpha * x + (1 - P.alpha) * z100[i]);
      p = ph.map((x, i) => P.mix * x + (1 - P.mix) * zs[i]);
    }
    if (knn && P.beta > 0) {
      const rows = knn.labels.length;
      const s = Array.from({ length: rows }, (_, r) => { let t = 0; const o = r * dim; for (let j = 0; j < dim; j++) t += knn.q[o + j] * vec[j]; return t / knn.scale; });
      const order = s.map((_, i) => i).sort((a, b) => s[b] - s[a]).slice(0, P.k);
      const vote = new Array(n).fill(0);
      for (const r of order) { const w = Math.exp(P.tau * (s[r] - 1)); for (const l of knn.labels[r]) vote[l] += w / knn.labels[r].length; }
      const z = vote.reduce((a, b) => a + b, 0);
      if (z > 0) p = p.map((x, i) => (1 - P.beta) * x + P.beta * vote[i] / z);
    }
    return p.map((score, i) => ({ ko: meta[i].ko, parent: meta[i].parent, score }))
      .sort((a, b) => b.score - a.score).slice(0, 3).map((m) => ({ ...m, level: LEVEL(m.score) }));
  };
}

function b64bytes(b64) {
  if (typeof Buffer !== 'undefined') return new Uint8Array(Buffer.from(b64, 'base64'));
  const s = atob(b64); const u8 = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u8[i] = s.charCodeAt(i);
  return u8;
}
function softmax(xs) { const m = Math.max(...xs); const e = xs.map((x) => Math.exp(x - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); }
function norm(xs) { const s = xs.reduce((a, b) => a + b, 0) || 1; return xs.map((x) => x / s); }
