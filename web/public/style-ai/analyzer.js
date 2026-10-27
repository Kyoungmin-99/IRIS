// 핵심 분석 모듈 — UI 에 의존하지 않는다 (브라우저 · Node 둘 다 동작)
//
//   const analyzer = await createAnalyzer({ model, labelEmbeddings, heads, onProgress });
//   const result   = await analyzer.analyze(fileOrUrl, { category: 'auto' });
//
// 동작 원리
//   1) 속성 사전의 라벨 문장 임베딩은 미리 계산한 파일을 쓴다 (없으면 텍스트 모델로 계산)
//   2) 사진 1장 = 비전 임베딩 1번 (선택: 좌우 반전 평균)
//   3) 그룹마다 학습된 헤드가 있으면 헤드로, 없으면 제로샷(코사인 유사도)으로 확률 계산
//      메이크업처럼 작은 부위를 보는 그룹은 얼굴 · 눈 · 입술만 잘라낸 임베딩으로 판단한다 (regions 옵션, 헤드 파일의 regions)
//   4) 톤은 컬러 라벨이 가진 톤 정보와 사진에서 직접 읽은 톤을 합쳐 판단
//   5) 한국어 문장 · 해시태그 · SNS 트렌드 키워드 조립 (describe.js)

import { RawImage } from '@huggingface/transformers';
import { TAXONOMY, CATEGORY_ORDER, CONFIDENCE, OTHER_DETECT } from './taxonomy.js';
import { loadVisionEncoder, loadTextEncoder, flipHorizontal, normalize } from './encoders.js';
import { createMotifJudge } from './motif.js';
import { compose } from './describe.js';

export const DEFAULT_MODEL = 'Marqo/marqo-fashionSigLIP'; // 권장 기본 모델 (heads/ · embeddings/ 와 짝)
const LOGIT_SCALE = 100; // 제로샷 소프트맥스 온도 (CLIP 의 exp(logit_scale))
const PARTNER_W = 0.25;   // 정밀 모드에서 짝(기본 모델) 확률 비중 (attributesFor 참고)

const SECONDARY_MIN = 0.25;
const boxesOf = new WeakMap();   // 부위 임베딩 → 그 부위 상자 // 2위 카테고리 확률이 이 이상이면 '함께 보이는 스타일'로 함께 설명

export async function createAnalyzer({
  model = DEFAULT_MODEL,
  device,          // 'webgpu' | 'wasm' | undefined(자동)
  dtype,           // 'q4' | 'q8' | 'fp16' | 'fp32' | undefined(자동)
  labelEmbeddings, // buildLabelIndex() 결과(JSON). 있으면 텍스트 모델을 내려받지 않는다
  heads,           // tools/train-heads.py 결과(JSON). 있으면 학습된 분류 헤드를 쓴다
  tta = false,     // true 면 좌우 반전 이미지까지 평균 (정확도 ↑, 시간 2배)
  regions,         // (source, wanted) => { face?, eye?, lip?, hair?, bangs?, head? : [x0, y0, x1, y1] (0~1 비율) } — 부위 잘라 보기 (없으면 전체 사진으로만 판단)
  motifs,          // embeddings/motifs-*.json (타투 세부 모티브). 있으면 타투 '도안'에 motifs: [{ ko, parent, score, level }] 를 붙인다
  sessionOptions,  // ONNX Runtime 세션 옵션 (휴대폰: 메모리 아레나 끄기)
  onProgress,
} = {}) {
  const progress = (p) => onProgress?.(p);
  const hash = taxonomyHash();
  const usePrecomputed = labelEmbeddings && labelEmbeddings.hash === hash && labelEmbeddings.model === model;
  if (labelEmbeddings && !usePrecomputed) console.warn('[analyzer] 사전 계산 임베딩이 현재 속성 사전과 달라 직접 계산합니다.');
  const useHeads = heads && heads.hash === hash && heads.model === model;
  if (heads && !useHeads) console.warn('[analyzer] 학습된 헤드가 현재 속성 사전·모델과 달라 제로샷으로만 판단합니다.');

  const vision = await loadVisionEncoder(model, { device, dtype, progress_callback: progress, session_options: sessionOptions });

  let index;
  if (usePrecomputed) index = decodeIndex(labelEmbeddings);
  else {
    progress({ status: 'embedding-labels' });
    index = (await buildLabelIndex({ model, onProgress })).index;
  }
  const headIndex = useHeads ? decodeHeads(heads) : {};
  // 확률 보정 온도 (학습 때 폴드 밖 예측으로 맞춘 값). 표시 확률이 실제 적중률에 가깝도록 한다
  const calib = useHeads ? heads.calib || {} : {};
  // 그룹 → 잘라 볼 부위 (예: 'makeup.eye' → 'eye'). 그 부위의 헤드 · 보정 온도는 '그룹@부위' 키로 저장돼 있다
  const regionOf = useHeads ? heads.regions || {} : {};
  // 카테고리 → 그 카테고리 그룹들이 보는 부위 이름들 (예: makeup → eye, lip, face)
  const regionsByCat = {};
  for (const [k, r] of Object.entries(regionOf)) (regionsByCat[k.split('.')[0]] ??= new Set()).add(r);
  // 그룹별 단정 기준: 폴드 밖 예측에서 '이 확률 이상이면 95% 맞는다'는 값 (없으면 공통 기준)
  const conf = useHeads ? heads.conf || {} : {};
  const highOf = (key) => conf[key] ?? CONFIDENCE.high;
  const levelOf = (key, p) => (p >= highOf(key) ? 'high' : p >= CONFIDENCE.mid ? 'mid' : 'low');
  // 비슷한 학습 사진(kNN): 학습 사진 특징값(8비트) + 그룹마다 섞는 비율 β
  const knn = useHeads && heads.knn ? decodeKnn(heads.knn) : null;
  const motifJudge = motifs && motifs.model === model ? createMotifJudge(motifs) : null;
  progress({ status: 'ready' });

  async function embedImage(source) {
    const image = source instanceof RawImage ? source : await RawImage.read(source);
    const [v] = await vision.embed(image);
    if (!tta) return { image, vec: v };
    const [w] = await vision.embed(flipHorizontal(image));
    return { image, vec: normalize(v.map((x, i) => x + w[i])) };
  }

  // 한 그룹의 확률 분포: 학습 헤드가 있으면 (헤드와 제로샷을 alpha 로 섞어) 사용
  // regionVecs 에 이 그룹이 보는 부위의 임베딩이 있으면 그것으로 판단한다 (제로샷 · 헤드 · 보정 모두 부위 기준)
  function groupProbs(key, vec, textEmbs, regionVecs) {
    const reg = regionOf[key];
    let hk = key;
    if (reg && regionVecs?.[reg]) { vec = regionVecs[reg]; hk = `${key}@${reg}`; }
    const h = headIndex[hk];
    // zb: 학습 사진이 없는 라벨의 감점 (교차 검증으로 고른 값, 헤드 파일에 저장). 사진 없는 새 라벨이 다른 라벨의 답을 빼앗지 않게
    const zs = softmax(textEmbs.map((e, i) => dot(vec, e) * LOGIT_SCALE + (h?.zb?.[i] ?? 0)));
    let p = zs;
    if (h) {
      const hp = softmax(h.W.map((row, i) => dot(vec, row) + h.b[i]));
      p = hp.map((x, i) => h.alpha * x + (1 - h.alpha) * zs[i]);
    }
    // 비슷한 학습 사진들의 라벨 투표를 섞는다 (전체 사진으로 판단하는 그룹만, β 는 교차 검증으로 고른 값)
    const kg = hk === key ? knn?.groups[key] : null;
    if (kg && kg.beta > 0) {
      const pk = knnVote(knn, kg, vec, p.length);
      if (pk) p = p.map((x, i) => (1 - kg.beta) * x + kg.beta * pk[i]);
    }
    const t = calib[hk];
    return t ? softmax(p.map((x) => Math.log(Math.max(x, 1e-12)) / t)) : p;
  }

  // 한 카테고리의 속성 그룹별 결과 (확률 높은 순 후보 · 신뢰도 단계)
  // partner: 다른 모델의 같은 카테고리 그룹 분포 (groupDists 결과). 있으면 두 모델 확률을 섞는다 (교차 검증 +2%p).
  // 짝(정밀 모드의 기본 모델) 비중 PARTNER_W: 평가 세트 v3 에서 0.25 가 반반(0.5)보다 나았다 (새 평가 사진 86.8 → 87.9%, 기존 사진은 같음)
  // 두 모델 모두 부위를 잘라 본 판단끼리 평균한다 (부위 그룹도 평균이 한 모델 단독보다 대체로 낫다)
  function attributesFor(cat, vec, topk, regionVecs, partner = null) {
    const def = TAXONOMY[cat];
    const dist = {};
    for (const g of def.groups) {
      dist[g.key] = groupProbs(`${cat}.${g.key}`, vec, index[cat].groups[g.key], regionVecs);
      const pd = partner?.dists[g.key];
      if (pd && pd.length === dist[g.key].length) dist[g.key] = dist[g.key].map((x, i) => (1 - PARTNER_W) * x + PARTNER_W * pd[i]);
    }
    fuseTone(def, dist);
    // 타투 '도안' 아래 세부 모티브 (장미 · 늑대 …): 도안 확률을 부모 확률로 쓴다
    const motifsOf = cat === 'tattoo' && motifJudge && dist.subject ? motifJudge(vec, dist.subject) : null;
    return def.groups.map((g) => {
      const items = g.labels
        .map((l, i) => ({ label: l.ko, label_en: l.en, score: dist[g.key][i], hidden: !!l.hidden, tone: l.tone }))
        .sort((a, b) => b.score - a.score);
      const top = items[0];
      const key = `${cat}.${g.key}`;
      const level = levelOf(key, top.score);
      // 계열: 1위가 애매해도 같은 계열 라벨들의 확률 합이 높으면 '보브 계열'처럼 묶어서 말한다
      const fam = familyOf(g, top.label, items, (p) => levelOf(key, p), level);
      const reg = (regionOf[key] && regionVecs?.[regionOf[key]] ? regionOf[key] : null) || partner?.regionGroups[g.key] || null;
      return { group: g.key, group_label: g.label, label: top.label, label_en: top.label_en, score: top.score, level,
        alternatives: items.slice(1, topk), all: items, ...(reg ? { region: reg } : {}), ...(fam ? { family: fam } : {}), high: highOf(key),
        ...(g.key === 'subject' && motifsOf ? { motifs: motifsOf } : {}) };
    });
  }

  // 부위 잘라 보기: 얼굴 · 눈 · 입술 · 머리카락 · 앞머리 · 머리 상자(0~1 비율)를 받아 그 부분만 임베딩한다. 못 찾으면 null (전체 사진으로 판단)
  // wanted: 필요한 부위 이름들 (없으면 regions 가 돌려준 상자 모두)
  async function regionEmbeddings(source, image, wanted = null) {
    if (!regions) return null;
    let boxes = null;
    try { boxes = await regions(source, wanted); } catch (e) { console.warn('[analyzer] 부위를 찾지 못해 전체 사진으로 판단합니다', e); }
    if (!boxes) return null;
    const names = [], crops = [];
    for (const [name, [x0, y0, x1, y1]] of Object.entries(boxes)) {
      if (wanted && !wanted.has(name)) continue;
      const X0 = Math.max(0, Math.floor(x0 * image.width)), Y0 = Math.max(0, Math.floor(y0 * image.height));
      const X1 = Math.min(image.width, Math.ceil(x1 * image.width)), Y1 = Math.min(image.height, Math.ceil(y1 * image.height));
      if (X1 - X0 < 24 || Y1 - Y0 < 16) continue;
      names.push(name); crops.push(await image.crop([X0, Y0, X1 - 1, Y1 - 1]));
    }
    if (!crops.length) return null;
    const used = Object.fromEntries(names.map((n) => [n, boxes[n]]));
    // 한 장씩 돌린다 (묶어 돌리면 약간 빠르지만 값이 미세하게 달라져, 한 장씩 계산한 값으로 학습한 부위 헤드와 어긋난다)
    const out = {};
    for (const [i, c] of crops.entries()) out[names[i]] = (await vision.embed(c))[0];
    boxesOf.set(out, used);   // 화면에 'AI 가 본 곳'을 그릴 수 있게 상자도 함께 기억
    return out;
  }

  // 한 카테고리의 그룹별 확률 분포만 (톤 합치기 전). 다른 모델과 평균 낼 때 짝 모델로 쓰인다
  async function groupDists(source, cat) {
    const { image, vec } = await embedImage(source);
    const wanted = regionsByCat[cat] || new Set();
    const regionVecs = wanted.size ? await regionEmbeddings(source, image, wanted) : null;
    const dists = {}, regionGroups = {};
    for (const g of TAXONOMY[cat].groups) {
      const key = `${cat}.${g.key}`;
      dists[g.key] = groupProbs(key, vec, index[cat].groups[g.key], regionVecs);
      if (regionOf[key] && regionVecs?.[regionOf[key]]) regionGroups[g.key] = regionOf[key];
    }
    return { model, dists, regionGroups, regions: regionVecs ? Object.keys(regionVecs) : [] };
  }

  // partner: 함께 쓸 다른 분석기 (정밀 모드에서 기본 모델). 있으면 주 카테고리의 그룹 확률을 두 모델이 평균한다
  // 사진 속 주인공이 사람이 아니면(동물 · 인형) 무엇인지 알아본다 — 분석은 그대로 하고 결과 위 한마디에만 쓴다 (embeddings/*.json 의 guard)
  const guard = labelEmbeddings?.guard?.kinds?.length ? labelEmbeddings.guard : null;
  function subjectOf(vec) {
    if (!guard) return null;
    const sp = Math.max(...guard.person.map((e) => dot(vec, e)));
    let best = null;
    for (const k of guard.kinds) { const s = Math.max(...k.vecs.map((e) => dot(vec, e))); if (!best || s > best.s) best = { k, s }; }
    if (!best.k.ko) return null;                               // 풍경 · 음식 · 물건은 한마디 없이
    const gap = best.s - sp;
    return gap >= (guard.margin?.[best.k.type] ?? 1) ? { ko: best.k.ko, type: best.k.type, gap } : null;
  }

  async function analyze(source, { category = 'auto', topk = 3, partner = null } = {}) {
    const t0 = now();
    const { image, vec } = await embedImage(source);
    const subject = subjectOf(vec);

    // 카테고리 판별 (학습 헤드 'category' 가 있으면 함께 사용)
    const detectSims = CATEGORY_ORDER.map((cat) => Math.max(...index[cat].detect.map((e) => dot(vec, e))));
    let catProbs = softmax(detectSims.map((s) => s * LOGIT_SCALE));
    const zsCatProbs = catProbs; // 제로샷 카테고리 확률 (함께 보이는 스타일 판단용: 카테고리마다 따로 보이는 정도)
    if (headIndex.category) {
      const h = headIndex.category;
      const hp = softmax(h.W.map((row, i) => dot(vec, row) + h.b[i]));
      catProbs = catProbs.map((p, i) => h.alpha * hp[i] + (1 - h.alpha) * p);
    }
    const catRanked = CATEGORY_ORDER.map((cat, i) => ({ key: cat, label: TAXONOMY[cat].label, score: catProbs[i] }))
      .sort((a, b) => b.score - a.score);
    // 뷰티 사진 여부: 가장 가까운 뷰티 카테고리 문장 vs '기타' 문장
    let beautyScore = 1;
    if (index.other?.length) {
      const otherSim = Math.max(...index.other.map((e) => dot(vec, e)));
      beautyScore = softmax([Math.max(...detectSims), otherSim].map((x) => x * LOGIT_SCALE))[0];
    }
    if (headIndex.beauty) { // 학습된 뷰티 판별 헤드 (뷰티 vs 기타)
      const h = headIndex.beauty;
      const hp = softmax(h.W.map((row, i) => dot(vec, row) + h.b[i]));
      beautyScore = h.alpha * hp[0] + (1 - h.alpha) * beautyScore;
    }
    const chosen = category === 'auto' ? catRanked[0].key : category;
    const def = TAXONOMY[chosen];
    // 부위 임베딩은 그것을 쓰는 카테고리(메이크업)가 주 · 보조 결과에 나올 때만 계산한다
    const second = CATEGORY_ORDER.map((key, i) => ({ key, score: zsCatProbs[i] })).filter((c) => c.key !== chosen)
      .sort((a, b) => b.score - a.score)[0];
    const wanted = new Set([...(regionsByCat[chosen] || [])]);
    if (category === 'auto' && second && second.score >= SECONDARY_MIN) for (const r of regionsByCat[second.key] || []) wanted.add(r);
    const regionVecs = wanted.size ? await regionEmbeddings(source, image, wanted) : null;
    let pd = null;
    if (partner) { try { pd = await partner.groupDists(source, chosen); } catch (e) { console.warn('[analyzer] 짝 모델 분석 실패 → 이 모델만 씁니다', e); } }
    const attributes = attributesFor(chosen, vec, topk, regionVecs, pd);
    const text = compose(chosen, attributes);
    const confidence = attributes.reduce((s, a) => s + a.score, 0) / attributes.length;

    // 함께 보이는 스타일: 얼굴 사진에는 헤어와 메이크업이 같이 나오는 경우가 많다.
    // 2위 카테고리도 충분히 뚜렷하면 같은 임베딩으로 한 번 더 분석한다 (추가 비용 거의 없음)
    let secondary = null;
    // 학습된 카테고리 헤드는 '주제 하나'를 고르도록 학습돼 2위 확률이 매우 낮다. 그래서 제로샷 확률로 판단한다
    if (category === 'auto' && second && second.score >= SECONDARY_MIN) {
      const attrs2 = attributesFor(second.key, vec, topk, regionVecs);
      const t2 = compose(second.key, attrs2);
      secondary = { category: second.key, category_label: TAXONOMY[second.key].label, score: second.score, attributes: attrs2,
        genre: t2.genre, headline: t2.headline, description_ko: t2.description, sentences: t2.sentences, paragraphs: t2.paragraphs, tags: t2.tags, trends: t2.trends };
    }
    // 사진 품질 안내: 너무 작은 사진은 세부 속성을 읽기 어렵다
    const warnings = [];
    if (Math.min(image.width, image.height) < 224) warnings.push('사진이 작아 세부 판단이 부정확할 수 있어요. 더 큰 사진을 권장합니다.');

    return {
      category: chosen,
      category_label: def.label,
      category_auto: category === 'auto',
      category_ranking: catRanked,
      subject,                       // 사람이 아니면 { ko: '강아지', type: 'animal' } (아니면 null)
      is_beauty: beautyScore >= 0.5,
      beauty_score: beautyScore,
      attributes,
      headline: text.headline,
      genre: text.genre,
      description_ko: text.description,
      sentences: text.sentences,
      paragraphs: text.paragraphs,
      tags: text.tags,
      trends: text.trends,
      confidence,
      secondary,
      warnings,
      elapsed_ms: Math.round(now() - t0),
      image_size: { width: image.width, height: image.height },
      model,
      trained_heads: Object.keys(headIndex).length,
      region_boxes: regionVecs ? boxesOf.get(regionVecs) || null : null,   // { eye: [x0, y0, x1, y1] (0~1), … } — 부위를 잘라 본 자리
      regions_used: [...new Set([...(regionVecs ? Object.keys(regionVecs) : []), ...(pd?.regions || [])])],
      ensemble: pd ? [model, pd.model] : null,
      // 피드백 학습용 특징값 (사진 대신 저장한다). JSON 출력에는 넣지 않는다
      embedding: vec,
      taxonomy_hash: hash,
    };
  }

  // 같은 모델로 분석을 동시에 두 번 돌리지 않는다 (WebGPU · 작업 스레드에서 'Session already started' 같은 오류가 나는 것을 막는다).
  // 요청은 차례로 처리한다. 짝 모델(partner)은 다른 인스턴스라 서로 기다리지 않는다
  let chain = Promise.resolve();
  const serial = (fn) => (...args) => { const p = chain.then(() => fn(...args)); chain = p.catch(() => {}); return p; };
  return {
    analyze: serial(analyze),
    groupDists: serial(groupDists),
    model,
    precomputed: !!usePrecomputed,
    trainedHeads: Object.keys(headIndex).length,
    dispose: () => vision.dispose?.(),
  };
}

// 톤 판정 = 사진에서 직접 읽은 톤(제로샷) 50% + 컬러 라벨이 가진 톤 50%
// (헤어는 '컬러' 그룹, 메이크업은 '립 컬러' 그룹의 tone 정보를 쓴다)
const TONE_KEYS = { cool: '쿨톤', warm: '웜톤', neutral: '뉴트럴' };
export function fuseTone(def, dist) {
  const toneGroup = def.groups.find((g) => g.key === 'tone');
  if (!toneGroup) return;
  const src = def.groups.find((g) => g.key !== 'tone' && g.labels.some((l) => l.tone));
  if (!src) return;
  const prior = { 쿨톤: 0, 웜톤: 0, 뉴트럴: 0 };
  src.labels.forEach((l, i) => { if (l.tone) prior[TONE_KEYS[l.tone]] += dist[src.key][i]; });
  const zs = dist.tone;
  const fused = toneGroup.labels.map((l, i) => 0.5 * zs[i] + 0.5 * (prior[l.ko] ?? 0));
  const s = fused.reduce((a, b) => a + b, 0) || 1;
  dist.tone = fused.map((x) => x / s);
}

// ---- 라벨 임베딩 계산 (오프라인 사전 계산 스크립트와 브라우저 폴백이 공유) ------
export async function buildLabelIndex({ model = DEFAULT_MODEL, dtype = 'fp32', device, onProgress } = {}) {
  const text = await loadTextEncoder(model, { dtype, device, progress_callback: (p) => onProgress?.(p) });
  // 여러 템플릿 문장의 임베딩을 평균 → 정규화 (prompt ensembling)
  async function embedLabelSet(templates, labelsEn) {
    const prompts = [];
    for (const en of labelsEn) for (const t of templates) prompts.push(t.replace('{}', en));
    const embs = await text.embed(prompts);
    return labelsEn.map((_, i) => normalize(meanVec(embs.slice(i * templates.length, (i + 1) * templates.length))));
  }
  const index = {};
  for (const cat of CATEGORY_ORDER) {
    const def = TAXONOMY[cat];
    const groups = {};
    for (const g of def.groups) groups[g.key] = await embedLabelSet(g.templates, g.labels.map((l) => l.en));
    index[cat] = { detect: await text.embed(def.detect), groups };
  }
  index.other = await text.embed(OTHER_DETECT);
  await text.dispose?.();
  return { index, json: encodeIndex(index, model) };
}

// 속성 사전이 바뀌면 사전 계산 파일 · 학습 헤드를 무효화하기 위한 해시
export function taxonomyHash() {
  const src = JSON.stringify([OTHER_DETECT, CATEGORY_ORDER.map((c) => [TAXONOMY[c].detect, TAXONOMY[c].groups.map((g) => [g.key, g.templates, g.labels.map((l) => l.en)])])]);
  let h = 2166136261;
  for (let i = 0; i < src.length; i++) { h ^= src.charCodeAt(i); h = Math.imul(h, 16777619); }
  return (h >>> 0).toString(16);
}

// index ⇄ JSON (Float32 → base64)
function encodeIndex(index, model) {
  const vecs = [];
  const layout = {};
  for (const cat of CATEGORY_ORDER) {
    layout[cat] = { detect: index[cat].detect.length, groups: {} };
    vecs.push(...index[cat].detect);
    for (const g of TAXONOMY[cat].groups) { layout[cat].groups[g.key] = index[cat].groups[g.key].length; vecs.push(...index[cat].groups[g.key]); }
  }
  if (index.other) { layout.other = index.other.length; vecs.push(...index.other); } // 맨 끝에 붙인다
  const dim = vecs[0].length;
  const buf = new Float32Array(vecs.length * dim);
  vecs.forEach((v, i) => buf.set(v, i * dim));
  return { model, hash: taxonomyHash(), dim, count: vecs.length, layout, data: toBase64(new Uint8Array(buf.buffer)) };
}
function decodeIndex(json) {
  const buf = f32FromBase64(json.data);
  let k = 0;
  const take = () => Array.from(buf.subarray(k * json.dim, ++k * json.dim));
  const index = {};
  for (const cat of CATEGORY_ORDER) {
    const L = json.layout[cat];
    const detect = Array.from({ length: L.detect }, take);
    const groups = {};
    for (const g of TAXONOMY[cat].groups) groups[g.key] = Array.from({ length: L.groups[g.key] }, take);
    index[cat] = { detect, groups };
  }
  if (json.layout.other) index.other = Array.from({ length: json.layout.other }, take);
  return index;
}
// 계열 판단: 1위 라벨이 속한 계열의 확률 합 → 단계가 1위 혼자보다 올라갈 때만 계열로 말한다
const LV_RANK = { low: 0, mid: 1, high: 2 };
function familyOf(group, topLabel, items, levelFn, topLevel) {
  const f = group.families?.find((x) => x.members.includes(topLabel));
  if (!f) return null;
  const mem = items.filter((x) => f.members.includes(x.label));
  if (mem.length < 2) return null;
  const score = Math.min(1, mem.reduce((s, x) => s + x.score, 0));
  const level = levelFn(score);
  if (LV_RANK[level] <= LV_RANK[topLevel]) return null;
  return { name: f.name, label: `${f.name} 계열`, score, level, members: mem.filter((x) => x.score >= 0.05).slice(0, 4).map((x) => ({ label: x.label, score: x.score })) };
}
// kNN 뱅크: { dim, rows, scale, q(base64 int8 rows×dim), groups: { key: { beta, k, tau, rows:[...], labels:[[...]] } } }
function decodeKnn(j) {
  const bytes = typeof Buffer !== 'undefined' ? Uint8Array.from(Buffer.from(j.q, 'base64')) : Uint8Array.from(atob(j.q), (c) => c.charCodeAt(0));
  return { dim: j.dim, scale: j.scale, q: new Int8Array(bytes.buffer), groups: j.groups };
}
function knnVote(bank, g, vec, nLabels) {
  const { dim, q, scale } = bank;
  const sims = g.rows.map((r) => { let s = 0; const o = r * dim; for (let i = 0; i < dim; i++) s += vec[i] * q[o + i]; return s / scale; });
  const order = sims.map((s, i) => i).sort((a, b) => sims[b] - sims[a]).slice(0, g.k);
  if (!order.length) return null;
  const out = new Array(nLabels).fill(0);
  for (const i of order) {
    const w = Math.exp(g.tau * (sims[i] - 1)); const ls = g.labels[i];
    for (const l of ls) out[l] += w / ls.length;
  }
  const z = out.reduce((a, b) => a + b, 0);
  return z > 0 ? out.map((x) => x / z) : null;
}
// 학습 헤드: { groups: { "hair.color": { rows, W(base64 f32 rows×dim), b:[...], alpha } } }
function decodeHeads(json) {
  const out = {};
  for (const [key, h] of Object.entries(json.groups)) {
    const buf = f32FromBase64(h.W);
    const W = Array.from({ length: h.rows }, (_, i) => Array.from(buf.subarray(i * json.dim, (i + 1) * json.dim)));
    out[key] = { W, b: h.b, alpha: h.alpha ?? 1, region: h.region, zb: h.zb };
  }
  return out;
}
function toBase64(u8) {
  if (typeof Buffer !== 'undefined') return Buffer.from(u8).toString('base64');
  let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode(...u8.subarray(i, i + 0x8000));
  return btoa(s);
}
function f32FromBase64(b64) {
  let u8;
  if (typeof Buffer !== 'undefined') u8 = new Uint8Array(Buffer.from(b64, 'base64'));
  else { const s = atob(b64); u8 = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u8[i] = s.charCodeAt(i); }
  return new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4);
}

// ---- 수학 유틸 ----------------------------------------------------------------
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function meanVec(rows) {
  const out = new Array(rows[0].length).fill(0);
  for (const r of rows) for (let i = 0; i < r.length; i++) out[i] += r[i];
  return out.map((x) => x / rows.length);
}
function softmax(xs) {
  const m = Math.max(...xs);
  const ex = xs.map((x) => Math.exp(x - m));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map((e) => e / s);
}
function now() { return (typeof performance !== 'undefined' ? performance : Date).now(); }
