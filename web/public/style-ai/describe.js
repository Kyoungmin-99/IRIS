// 한국어 설명문 생성기 — 한국 인스타그램 · 네이버 뷰티 말투 기준
// analyzer 가 뽑은 그룹별 속성(라벨 + 보정 확률)을 받아
//   - 사람이 직접 쓴 듯한 설명문
//   - 해시태그
//   - SNS 트렌드 키워드 (예: '쇠맛 네일', '레이어드 C컬', '차가운 애쉬브라운')
// 를 만든다.
//
// 문장이 매번 같은 틀로 나오지 않도록 자리마다 표현을 여러 개 두고 사진마다 다르게 고른다.
// 같은 사진은 늘 같은 문장이 나오도록, 고르는 기준(난수 씨앗)은 분석 결과 값으로 만든다.
// 말투는 보정 확률로 정한다: high(≥0.9) 단정 · mid(≥0.5) "~로 보여요 / ~에 가까워요" ·
// low 는 문장에서 빼고 화면의 후보 막대로만 보여 준다 (장르 · 컬러처럼 꼭 필요한 문장만 "추정돼요"로 남긴다).

import { matchTrends } from './trends.js';
import { TAXONOMY, TONE_WORDS, CONFIDENCE } from './taxonomy.js';

const MAX_HEDGES = 2; // 설명문 속 불확실성 언급은 최대 2번 (나머지는 막대그래프로 확인)
let hedgeBudget = MAX_HEDGES;
let omitted = 0; // 확실하지 않아 문장에서 뺀 세부 항목 수
let rand = Math.random;
let cat = 'hair';
let PRE = []; // 트렌드 표에서 조건 2개 이상으로 맞은 이름들 (문장 속 트렌드 줄에 먼저 쓴다)

// 문단을 나누는 자리 표시 (문장 목록 안에서만 쓰고, 결과 문장에는 남지 않는다)
const BR = Symbol('문단');

export function compose(category, attributes) {
  hedgeBudget = MAX_HEDGES;
  omitted = 0;
  cat = category;
  rand = seeded(attributes.map((x) => `${x.group}:${x.label}:${Math.round(x.score * 100)}`).join('|'));
  // 1위 라벨이 애매해도 같은 계열의 확률 합이 높으면 문장에서는 '보브 계열'처럼 계열 이름으로 말한다
  const a = Object.fromEntries(attributes.map((x) => [x.group, x.family ? asFamily(x) : x]));
  const raw = Object.fromEntries(attributes.map((x) => [x.group, x]));
  const P = (group, label) => raw[group]?.all.find((i) => i.label === label)?.score ?? 0;   // 확률은 계열로 묶기 전 값
  // 트렌드 이름: 조건 2개 이상인 표 항목(더 구체적) → 카테고리별 규칙 → 조건 1개인 표 항목
  const table = matchTrends(category, P);
  PRE = table.filter((t) => t.n >= 2).map((t) => trend(t.name, t.why));
  const out = COMPOSERS[category](a, P);
  const trends = uniqBy([...out.trends.filter(Boolean), ...table.filter((t) => t.n < 2)], (t) => t.name).slice(0, 4);
  const tags = uniq([
    ...trends.map((t) => t.name.replace(/[\s·()]/g, '')),
    ...attributes.flatMap((x) => tagFor(category, x)),
  ]).slice(0, 12);
  // 문단: 카테고리마다 문장 사이에 넣은 BR 표시에서 나눈다 (빈 문단 · 같은 문장은 뺀다)
  const paragraphs = [[]];
  const seen = new Set();
  for (const x of out.sentences) {
    if (x === BR) { if (paragraphs.at(-1).length) paragraphs.push([]); continue; }
    if (!x || seen.has(x)) continue;
    seen.add(x);
    paragraphs.at(-1).push(x);
  }
  if (!paragraphs.at(-1).length) paragraphs.pop();
  if (omitted >= 2) paragraphs.push([pick(['나머지 항목은 사진만으로 확실하지 않아 아래 속성별 후보로 보여 드려요.', '확실하지 않은 항목은 문장에서 빼고 아래 후보 막대로 보여 드려요.'])]);
  const sentences = paragraphs.flat();
  return { headline: out.headline, genre: out.genre, sentences, paragraphs, description: paragraphs.map((x) => x.join(' ')).join('\n'), tags, trends };
}

// ---------------------------------------------------------------------------
// 해시태그: 1위 라벨 (확실할 때만)
function tagFor(category, attr) {
  const g = TAXONOMY[category].groups.find((x) => x.key === attr.group);
  // 태그는 1위 라벨만, 트렌드 칩과 같은 기준으로 (보정 확률 0.5 이상)
  return [attr.all[0]]
    .filter((i) => !i.hidden && i.score >= CONFIDENCE.mid)
    .map((i) => {
      const def = g.labels.find((l) => l.ko === i.label);
      if (def?.tag === '') return '';
      if (def?.tag) return def.tag;
      return (i.label + (g.tagSuffix ?? '')).replace(/[\s·()]/g, '');
    })
    .filter(Boolean);
}

// ---------------------------------------------------------------------------
// 표현 고르기: 결과 값으로 씨앗을 만든 난수 (FNV-1a 해시 → mulberry32)
function seeded(text) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) { h ^= text.charCodeAt(i); h = Math.imul(h, 16777619); }
  return () => {
    h = (h + 0x6d2b79f5) | 0;
    let t = Math.imul(h ^ (h >>> 15), 1 | h);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const pick = (x) => (Array.isArray(x) ? x[Math.floor(rand() * x.length)] : x);

// ---------------------------------------------------------------------------
// 한국어 헬퍼
const pct = (x) => `${Math.round(x * 100)}%`;
function lastJong(word) {
  const w = String(word).replace(/\([^)]*\)\s*$/, '').trim();
  for (let i = w.length - 1; i >= 0; i--) {
    const c = w.charCodeAt(i);
    if (c >= 0xac00 && c <= 0xd7a3) return (c - 0xac00) % 28;
    if (/[0-9A-Za-z]/.test(w[i])) return /[013678LMNlmn]/.test(w[i]) ? 1 : 0;
  }
  return 0;
}
// 조사: josa('중단발', '은/는') → '중단발은' (앞: 받침 있을 때, 뒤: 없을 때)
export function josa(word, pair) {
  const [withJong, noJong] = pair.split('/');
  const jong = lastJong(word);
  if (pair === '으로/로') return word + (jong === 0 || jong === 8 ? noJong : withJong); // ㄹ 받침은 '로'
  return word + (jong ? withJong : noJong);
}
const ida = (w) => josa(w, '이에요/예요'); // '숏컷이에요' · '스틸레토예요'

// 같은 스타일의 다른 이름 (taxonomy 의 syn)
function aka(group, label) {
  const g = TAXONOMY[cat].groups.find((x) => x.key === group);
  return g?.labels.find((l) => l.ko === label)?.syn?.[0] || null;
}

// 계열로 말할 때 쓰는 속성: 라벨 = 계열 이름, 확률 = 계열 합, 다른 후보 = 계열 밖 라벨
function asFamily(x) {
  const mem = new Set(x.family.members.map((m) => m.label));
  return { ...x, label: x.family.label, score: x.family.score, level: x.family.level, family_of: x.label,
    all: [{ label: x.family.label, score: x.family.score, hidden: false }, ...x.all.filter((i) => !mem.has(i.label))] };
}

// 불확실할 때 덧붙이는 보충 문장 (최대 MAX_HEDGES 번)
function hedge(attr) {
  if (!attr) return '';
  const alts = attr.all.slice(1).filter((x) => !x.hidden);
  const alt = alts[0];
  if (!alt || hedgeBudget <= 0) return '';
  if (attr.level === 'high' || (attr.level === 'mid' && alt.score < attr.score * 0.7)) return '';
  const first = hedgeBudget === MAX_HEDGES;
  hedgeBudget--;
  if (attr.level === 'mid') {
    const g = josa(attr.group_label, '은/는');
    return pick([
      `${g} ${alt.label}일 가능성도 있어요(${pct(attr.score)} 대 ${pct(alt.score)}).`,
      `${g} ${josa(alt.label, '과/와')} 헷갈릴 만큼 비슷해요(${pct(attr.score)} 대 ${pct(alt.score)}).`,
    ]);
  }
  const cands = alts.slice(0, 2).map((x) => x.label).join(' 또는 ');
  if (!first) return `${attr.group_label}도 확실하지 않아 ${cands}일 가능성이 있어요.`;
  return pick([
    `다만 ${josa(attr.group_label, '은/는')} 사진만으로 단정하기 어려워 ${cands}일 수도 있어요.`,
    `${josa(attr.group_label, '은/는')} 확실하지 않아 ${cands}일 가능성도 있어요.`,
  ]);
}
const sure = (attr, min = CONFIDENCE.mid) => !!attr && attr.score >= min; // 보정 확률 0.5 이상 ≈ 82% 적중
const TREND_MIN = CONFIDENCE.mid; // 트렌드 키워드는 칩으로 단정해 보이므로 같은 기준을 쓴다
// 여러 속성을 한 문장에 담을 때는 가장 불확실한 속성의 신뢰도로 말투를 정한다
const LV = { high: 2, mid: 1, low: 0 };
const weakest = (...attrs) => attrs.filter(Boolean).reduce((m, x) => (LV[x.level] < LV[m.level] ? x : m));
const shown = (attr) => !!attr && attr.level !== 'low' && !attr.all[0].hidden;
// 세부 문장: 확실하면 단정, 중간이면 부드럽게, 낮으면 문장으로 말하지 않고 막대그래프 후보로만 보여 준다
function say(attr, sureText, softText) {
  if (!attr) return '';
  if (attr.level === 'low') { omitted++; return ''; }
  return pick(attr.level === 'high' ? sureText : softText);
}
// 꼭 말해야 하는 문장(장르 · 컬러)의 서술어: 단정 / 보여요 · 가까워요 / 추정돼요
function ends(attr, noun) {
  if (attr.level === 'high') return ida(noun);
  if (attr.level === 'mid') return pick([`${josa(noun, '으로/로')} 보여요`, `${noun}에 가까워요`]);
  return `${josa(noun, '으로/로')} 추정돼요`;
}
const uniq = (arr) => [...new Set(arr.filter(Boolean))];
const uniqBy = (arr, f) => { const s = new Set(); return arr.filter((x) => (s.has(f(x)) ? false : s.add(f(x)))); };
const trend = (name, why) => ({ name, why });
const trendLine = (t) => pick([
  `SNS에서는 '${t.name}' 스타일로 많이 찾아요.`,
  `요즘 SNS에서 말하는 '${t.name}'에 가까워요.`,
  `'${t.name}'${lastJong(t.name) && lastJong(t.name) !== 8 ? '으로' : '로'} 검색하면 비슷한 스타일을 많이 볼 수 있어요.`,
]);
const akaLine = (label, other) => pick([
  `${josa(label, '은/는')} '${other}'${lastJong(other) ? '이라고도' : '라고도'} 불러요.`,
  `SNS에서는 '${other}'${lastJong(other) && lastJong(other) !== 8 ? '으로' : '로'}도 많이 검색해요.`,
]);

// 장르: 카테고리마다 대표 그룹의 1위 (+ 근소한 2위)
function genreOf(attr, name = attr.label, extra = {}) {
  const second = attr.all.slice(1).find((x) => !x.hidden);
  return { name, group: attr.group_label, score: attr.score, level: attr.level,
    second: second && second.score >= attr.score * 0.6 ? second.label : null, ...extra };
}

// 톤 이름: 톤 그룹 결과 → '차가운 쿨톤' 같은 수식어
function toneOf(a) {
  const t = a.tone;
  if (!t) return null;
  const w = TONE_WORDS[t.label];
  return { ...w, label: t.label, score: t.score, sure: t.score >= CONFIDENCE.mid && t.label !== '뉴트럴' };
}
const toneLine = (label) => pick(label === '쿨톤'
  ? ['쿨톤 계열이라 여름·겨울 쿨톤에게 잘 어울려요.', '전체 색감이 쿨톤이라 여름·겨울 쿨톤에게 추천해요.']
  : ['웜톤 계열이라 봄·가을 웜톤에게 잘 어울려요.', '전체 색감이 따뜻한 웜톤이라 봄·가을 웜톤에게 추천해요.']);

// ---------------------------------------------------------------------------
// 헤어 표현 [확실할 때, 중간일 때]
const BANGS = {
  '앞머리 없음': [['앞머리 없이 이마를 시원하게 드러냈어요.', '앞머리를 내리지 않아 이마선이 깔끔하게 보여요.'], ['앞머리 없이 넘긴 스타일로 보여요.']],
  풀뱅: [['이마를 덮는 풀뱅이 인상을 부드럽고 어려 보이게 해 줘요.', '숱 있게 내린 풀뱅으로 이마를 덮었어요.'], ['앞머리는 이마를 덮는 풀뱅으로 보여요.']],
  처피뱅: [['눈썹 위로 짧게 자른 처피뱅이 개성을 더해요.'], ['앞머리는 눈썹 위로 짧은 처피뱅에 가까워요.']],
  시스루뱅: [['이마가 살짝 비치는 시스루뱅으로 가볍게 내렸어요.', '가볍게 내린 시스루뱅이 답답하지 않은 인상을 줘요.'], ['앞머리는 이마가 비치는 시스루뱅으로 보여요.']],
  U뱅: [['가운데가 짧고 옆으로 갈수록 길어지는 U뱅이 얼굴선을 감싸요.'], ['앞머리는 U자로 떨어지는 U뱅에 가까워요.']],
  사이드뱅: [['앞머리를 한쪽으로 넘긴 사이드뱅이에요.', '옆으로 넘긴 사이드뱅이 성숙한 느낌을 줘요.'], ['앞머리는 옆으로 넘긴 사이드뱅으로 보여요.']],
  커튼뱅: [['가운데서 갈라 내린 커튼뱅이 얼굴을 자연스럽게 감싸요.'], ['앞머리는 가운데서 가른 커튼뱅으로 보여요.']],
  애교머리: [['얼굴 양옆으로 살짝 내린 애교머리가 얼굴선을 부드럽게 잡아 줘요.'], ['얼굴 옆으로 애교머리를 내린 것으로 보여요.']],
  '확인 불가': [['뒷모습 사진이라 앞머리는 보이지 않아요.'], ['앞머리는 사진에서 잘 보이지 않아요.']],
};
const COLOR_TECH = {
  브릿지: ['일부 가닥에 브릿지를 넣어 포인트를 줬어요.', '눈에 띄는 브릿지 컬러로 포인트를 살렸어요.'],
  하이라이트: ['가는 하이라이트를 촘촘히 넣어 입체감을 살렸어요.'],
  발레아쥬: ['붓으로 칠한 듯 자연스럽게 밝아지는 발레아쥬를 넣었어요.'],
  옴브레: ['뿌리는 어둡고 끝으로 갈수록 밝아지는 옴브레 기법이에요.'],
  이너컬러: ['안쪽 머리에만 다른 색을 넣은 이너컬러가 움직일 때마다 보여요.'],
  투톤: ['두 가지 색으로 나눈 투톤 컬러예요.'],
};
const STYLING = {
  포니테일: ['머리를 하나로 묶은 포니테일로 깔끔하게 정리했어요.'],
  똥머리: ['자연스럽게 틀어 올린 똥머리로 편안한 느낌을 냈어요.'],
  슬릭번: ['잔머리 없이 매끈하게 당겨 묶은 슬릭번이에요.'],
  반묶음: ['윗머리만 묶은 반묶음으로 단정함과 여성스러움을 함께 살렸어요.'],
  양갈래: ['양쪽으로 나눠 묶은 양갈래가 발랄해 보여요.'],
  땋은머리: ['머리를 땋아 디테일을 더했어요.'],
  드레드: ['드레드 헤어로 개성을 강하게 드러냈어요.'],
  업스타일: ['머리를 우아하게 올린 업스타일이에요.'],
  '집게핀 헤어': ['집게핀으로 뒷머리를 틀어 올려 꾸안꾸 느낌을 냈어요.'],
  사과머리: ['앞머리를 위로 묶은 사과머리가 귀여운 포인트예요.'],
};
const HAIR_MOOD = {
  청순: '맑고 청순한', 러블리: '사랑스러운', 시크: '세련되고 시크한', 힙: '개성 있고 힙한', 내추럴: '편안하고 자연스러운',
  '단정·오피스': '단정하고 깔끔한', '우아·고급': '우아하고 고급스러운', 레트로: '레트로 감성의',
};
const HAIR_SCENE = {
  '단정·오피스': '출근룩이나 면접처럼 단정해야 하는 자리에도 무난해요.',
  '우아·고급': '하객룩이나 격식 있는 자리에 잘 어울려요.',
  청순: '꾸민 듯 안 꾸민 듯한 데일리룩에 잘 어울려요.',
  러블리: '데이트룩처럼 화사한 차림과 잘 어울려요.',
};
const MEN_CUT = /댄디컷|쉼표머리|가르마펌|애즈펌|가일컷|리프컷|아이비리그컷|투블럭컷|크롭컷|페이드컷|포마드|버즈컷/;

// 네일 표현
const DESIGN_INFO = {
  자석: '빛을 받으면 띠 모양 반사광이 움직이는 자석젤',
  벨벳: '벨벳 천처럼 은은한 반짝임이 손톱 전체에 퍼지는 벨벳 자석',
  오로라: '보는 각도마다 색이 바뀌는 오로라 파우더',
  크롬: '거울처럼 반사되는 크롬 파우더',
  글레이즈드: '도넛 코팅처럼 은은한 펄 광택',
  자개: '자개 조각을 올려 오팔처럼 오묘하게 빛나는 자개',
  시럽: '속이 비치게 맑게 쌓아 올린 시럽 컬러',
  워터드롭: '물방울이 맺힌 듯 볼록한 입체 장식',
  치크: '볼터치처럼 손톱 가운데가 은은하게 물든 치크',
  마이크로프렌치: '손톱 끝에 아주 얇은 선만 그은 마이크로프렌치',
  딥프렌치: '손톱 끝 색을 두껍게 올린 딥프렌치',
  '3D조형': '젤로 빚어 올린 입체 조형',
  크롬하츠: '십자가와 체인 모양의 메탈 파츠',
  레이스: '레이스 원단처럼 섬세한 무늬',
  '호피·애니멀': '호피나 지브라 같은 동물 무늬',
  '니트·트위드': '스웨터 짜임이나 트위드 원단 같은 질감',
  마블: '대리석처럼 물결치는 마블 무늬',
  그라데이션: '색이 자연스럽게 번지는 그라데이션',
};
const NAIL_LAYOUT = {
  원포인트: [['한두 손가락에만 아트를 넣은 원포인트 구성이에요.', '나머지는 깔끔하게 두고 한두 개에만 포인트를 줬어요.'], ['한두 손가락에만 포인트를 준 구성으로 보여요.']],
  퐁당퐁당: [['손가락마다 색을 번갈아 칠한 퐁당퐁당 구성이에요.'], ['손가락마다 색을 번갈아 칠한 것으로 보여요.']],
  오마카세: [['손톱마다 디자인을 다르게 한 오마카세 구성이에요.', '손톱마다 다른 디자인을 섞은 오마카세 스타일이에요.'], ['손톱마다 디자인이 다른 오마카세 구성으로 보여요.']],
};
const NAIL_FINISH = { 유광: '반짝이는 유광', 매트: '보송한 매트', 메탈릭: '거울처럼 반사되는 메탈릭', '투명·쉬어': '속이 비치는 맑은', '펄·쉬머': '은은하게 빛나는 펄' };
const NAIL_MOOD = {
  오피스: '깔끔해서 출근할 때도 부담 없는', '청순·러블리': '사랑스럽고 여리여리한', 화려한: '화려하고 존재감 있는', '시크·힙': '시크하고 힙한',
  쇠맛: '차갑고 메탈릭한 쇠맛', 키치: '통통 튀는 키치한', 여름: '청량한 여름', '겨울·홀리데이': '포근한 겨울 홀리데이', 웨딩: '우아한 웨딩',
};
// 네일 트렌드 키워드: 디자인 라벨 → [키워드, 설명]
const NAIL_TREND = {
  자석: ['자석 네일', '빛에 따라 움직이는 캣아이'], 벨벳: ['벨벳 네일', '벨벳처럼 은은한 자석'], 오로라: ['오로라 네일', '각도마다 색이 바뀌는 오로라'],
  글레이즈드: ['글레이즈드 네일', '도넛처럼 은은한 펄 광택'], 시럽: ['시럽 네일', '맑게 비치는 젤리 컬러'], 치크: ['치크 네일', '볼터치처럼 물든 컬러'],
  워터드롭: ['물방울 네일', '물방울처럼 볼록한 장식'], 자개: ['자개 네일', '오팔처럼 빛나는 자개'], '호피·애니멀': ['호피 네일', '동물 무늬'],
  레이스: ['레이스 네일', '섬세한 레이스 무늬'], '니트·트위드': ['니트 네일', '스웨터 같은 질감'], 마이크로프렌치: ['마이크로 프렌치', '아주 얇은 프렌치'],
  크롬하츠: ['크롬하츠 네일', '메탈 십자가 파츠'],
};

// 메이크업 표현
const MAKEUP_MOOD = {
  데일리: '힘을 뺀 데일리', 청순: '맑고 깨끗한 청순', 음영: '브라운 톤으로 입체감을 준 음영', 과즙: '생기가 도는 과즙', 글램: '화려한 글램',
  스모키: '깊고 강렬한 스모키', 걸크러시: '당당하고 강렬한 걸크러시', 쇠맛: '차갑고 메탈릭한 쇠맛', Y2K: '반짝이는 2000년대 감성의 Y2K',
  갸루: '눈매를 크게 키운 갸루', '고딕·뱀파이어': '어둡고 퇴폐적인 고딕·뱀파이어', 레트로: '클래식한 레트로', 웨딩: '우아한 웨딩',
  아트: '개성 강한 아트', '할로윈·특수분장': '특수효과를 더한 할로윈', '남자 메이크업': '자연스럽게 다듬은 남자',
};
const BASE = { 물광: '물기를 머금은 듯 촉촉한 물광', 윤광: '은은하게 빛나는 윤광', 세미매트: '자연스러운 세미매트', '보송 매트': '보송하게 정돈한 매트' };
const SHADOW = {
  '음영 섀도': '은은한 음영 섀도', 스모키: '어둡게 번진 스모키 섀도', '글리터·펄': '반짝이는 글리터', '핑크·코랄 섀도': '화사한 핑크·코랄 섀도',
  '레드 섀도': '붉은 레드 섀도', '블루 섀도': '시원한 블루 섀도', '퍼플 섀도': '몽환적인 퍼플 섀도', '그린·옐로 섀도': '선명한 그린·옐로 섀도',
  '컬러 섀도': '여러 색을 섞은 컬러 섀도', '애교살 강조': '반짝이는 애교살', '눈앞머리 하이라이트': '눈앞머리 하이라이트',
};
const LINER = {
  캣아이라인: '끝을 올려 뺀 캣아이라인', '강아지 라인': '눈꼬리를 내린 강아지 라인', 언더라인: '아래 점막까지 채운 언더라인',
  '스머지 라인': '번지듯 풀어 준 스머지 라인', '그래픽 라인': '도형처럼 그린 그래픽 라인',
};
const LASH = { '인형 속눈썹': '가닥가닥 살린 인형 속눈썹', '언더 속눈썹': '아래 속눈썹', '컬러 마스카라': '컬러 마스카라' };
const LIP_TEX = {
  글로시: '촉촉한 글로시', 매트: '벨벳 같은 매트', 블러립: '경계를 흐린 블러', 그라데이션립: '안쪽부터 번지는 그라데이션',
  오버립: '입술선을 살짝 넘긴 오버립', '립라인 강조': '라이너로 테두리를 살린 립라인', '프로스티드 립': '펄이 도는 프로스티드',
};
const CHEEK = {
  '홍조 블러셔': [['볼과 콧등까지 붉게 물들인 홍조 블러셔로 생기를 더했어요.'], ['볼과 콧등에 홍조 블러셔를 올린 것으로 보여요.']],
  '숙취 블러셔': [['눈 바로 아래에 블러셔를 올려 발그레한 숙취 메이크업 느낌을 냈어요.'], ['블러셔를 눈 바로 아래에 올린 숙취 블러셔로 보여요.']],
  '코랄 블러셔': [['코랄 블러셔로 볼에 따뜻한 혈색을 더했어요.'], ['볼에는 코랄 블러셔를 올린 것으로 보여요.']],
  '베리 블러셔': [['베리 톤 블러셔로 볼에 깊이 있는 혈색을 냈어요.'], ['볼에는 베리 톤 블러셔를 올린 것으로 보여요.']],
  셰이딩: [['셰이딩으로 광대와 턱선 윤곽을 또렷하게 살렸어요.'], ['셰이딩으로 윤곽을 살린 것으로 보여요.']],
  하이라이터: [['광대와 콧대에 하이라이터를 올려 빛을 모았어요.'], ['광대에 하이라이터를 올린 것으로 보여요.']],
};
const BROW = {
  '일자 눈썹': [['눈썹은 일자로 곧게 그려 순한 인상이에요.'], ['눈썹은 일자 눈썹에 가까워요.']],
  '아치 눈썹': [['눈썹산을 살린 아치 눈썹이라 또렷한 인상이에요.'], ['눈썹은 아치형으로 보여요.']],
  '탈색 눈썹': [['눈썹을 밝게 탈색해 거의 보이지 않게 연출했어요.'], ['눈썹은 밝게 탈색한 것으로 보여요.']],
};
const DETAIL = {
  '큐빅·파츠': [['얼굴에 큐빅 파츠를 붙여 무대 메이크업처럼 반짝임을 더했어요.'], ['얼굴에 큐빅 같은 파츠를 붙인 것으로 보여요.']],
  주근깨: [['주근깨를 그려 넣어 햇볕에 그을린 듯한 느낌을 냈어요.'], ['주근깨를 그려 넣은 것으로 보여요.']],
};
// 메이크업 무드 → 트렌드 키워드
const MAKEUP_TREND = {
  쇠맛: '쇠맛 메이크업', Y2K: 'Y2K 메이크업', 갸루: '갸루 메이크업', '고딕·뱀파이어': '뱀파이어 메이크업', 걸크러시: '걸크러시 메이크업',
  과즙: '과즙 메이크업', 음영: '음영 메이크업', 청순: '청순 메이크업',
};

// 타투 장르 설명 (장르 판정 뒤에 한 줄로 덧붙인다)
const GENRE_INFO = {
  레터링: '글자와 문구를 필기체나 타이포그래피로 새기는 장르',
  파인라인: '가는 바늘로 섬세한 선만 살려 표현하는 장르',
  미니멀: '작고 단순한 선과 면으로 상징만 남기는 장르',
  두들: '낙서한 듯 삐뚤빼뚤한 선으로 가볍게 그리는 장르',
  블랙워크: '검은 잉크로 면을 넓게 채워 강한 대비를 주는 장르',
  블랙아웃: '넓은 면을 검은 잉크로 꽉 채우는 장르',
  블랙앤그레이: '검정과 회색 음영만으로 입체감을 살리는 장르',
  리얼리즘: '사진처럼 사실적인 빛과 질감을 살리는 장르',
  치카노: '부드러운 회색 음영과 필기체 레터링이 특징인 멕시코계 미국 장르',
  올드스쿨: '굵은 외곽선과 빨강·초록·노랑 같은 원색을 쓰는 미국 전통 장르',
  네오트래디셔널: '올드스쿨의 굵은 선에 풍부한 색감과 장식적 디테일을 더한 장르',
  뉴스쿨: '만화처럼 과장된 형태와 강렬한 컬러가 특징인 장르',
  애니: '애니메이션 캐릭터를 깔끔한 선과 평면 채색으로 옮기는 장르',
  이레즈미: '용·잉어·파도·모란 같은 소재를 넓은 부위에 새기는 일본 전통 장르',
  동양화: '먹의 번짐과 붓 터치를 살린 수묵화 느낌의 장르',
  수채화: '물감이 번진 듯한 색 번짐과 붓 터치를 살린 장르',
  일러스트: '펜 드로잉이나 스케치 같은 그림체를 살린 장르',
  도트워크: '수많은 작은 점으로 음영과 무늬를 만드는 장르',
  지오메트릭: '직선 · 원 · 삼각형을 정교하게 맞춰 짜는 기하학 장르',
  오너멘탈: '레이스나 장신구처럼 좌우 대칭의 장식 문양을 새기는 장르',
  트라이벌: '폴리네시아·마오리 같은 부족 문양을 굵은 검은 패턴으로 표현하는 장르',
  네오트라이벌: '가시처럼 뾰족하게 끝나는 굵은 검은 곡선의 Y2K 감성 장르',
  사이버시길리즘: '가시처럼 날카롭고 가느다란 곡선이 흐르는 사이버 감성 장르',
  트래쉬폴카: '검은 사실 묘사에 붉은 잉크 번짐을 콜라주처럼 더하는 장르',
};
const COLOR_PHRASE = {
  블랙: '검은 잉크 선 위주로', 블랙앤그레이: '블랙앤그레이 음영으로', 풀컬러: '선명한 풀컬러로', 포인트컬러: '블랙에 포인트 컬러를 더해',
  레드: '붉은 잉크만으로', 파스텔: '부드러운 파스텔 컬러로',
};
const COVER = /^(반팔|긴팔|등판|전신)$/; // 부위 전체를 덮는 크기
const SIZE_NP = {
  미니: '손톱만 한 미니', 스몰: '손바닥보다 작은 스몰', 미디엄: '손바닥만 한 미디엄', 대형: '큼직한 대형',
  반팔: '어깨부터 팔꿈치까지 덮는 반팔', 긴팔: '팔 전체를 덮는 긴팔', 등판: '등 전체를 채운 등판', 전신: '몸 전체를 잇는 전신',
};

// ---------------------------------------------------------------------------
const COMPOSERS = {
  // ─────────────────────────────── 헤어
  hair(a) {
    const { length, cut, styling, perm, bangs, color, colorTech, mood } = a;
    const tone = toneOf(a);
    const s = [];
    const plain = perm.label === '생머리' || perm.label === '내추럴 곱슬';
    const permNP = perm.label === '생머리' ? '매끈한 생머리' : perm.label === '내추럴 곱슬' ? '자연스러운 곱슬머리' : perm.label;
    // 첫 문장: 기장 + 커트 + 펌. 커트가 불확실하면 커트 없이 말한다
    const withCut = cut.level !== 'low';
    if (!withCut) omitted++;
    const lv = withCut ? weakest(cut, length, perm) : weakest(length, perm);
    const L = length.label;
    if (withCut && plain) {
      s.push(pick([
        `${L} 기장의 ${cut.label}, ${ends(lv, `${permNP} 스타일`)}.`,
        `${josa(permNP, '을/를')} ${L} 기장의 ${josa(cut.label, '으로/로')} 정리한 ${ends(lv, '스타일')}.`,
      ]));
    } else if (withCut) {
      s.push(pick([
        `${L} 기장의 ${cut.label}에 ${josa(perm.label, '을/를')} 더한 ${ends(lv, '스타일')}.`,
        `${josa(cut.label, '으로/로')} 커트하고 ${josa(perm.label, '으로/로')} 질감을 살린 ${ends(lv, `${L} 헤어`)}.`,
      ]));
    } else if (plain) {
      s.push(`${L} 기장의 ${ends(lv, permNP)}.`);
    } else {
      s.push(pick([`${L} 기장에 ${josa(perm.label, '을/를')} 넣은 ${ends(lv, '스타일')}.`, `${josa(perm.label, '으로/로')} 웨이브를 준 ${ends(lv, `${L} 헤어`)}.`]));
    }
    s.push(hedge(cut));
    s.push(hedge(length));
    const other = withCut && aka('cut', cut.label);
    if (other) s.push(akaLine(cut.label, other));

    s.push(BR);
    const B = BANGS[bangs.label];
    s.push(B ? say(bangs, B[0], B[1]) : say(bangs, `앞머리는 ${ida(bangs.label)}.`, `앞머리는 ${josa(bangs.label, '으로/로')} 보여요.`));

    const cLv = tone?.sure ? weakest(color, a.tone) : color;
    if (color.label === '흑발') {
      s.push(cLv.level === 'high' ? pick(['윤기 도는 흑발이에요.', '컬러는 깊은 흑발이에요.']) : `컬러는 ${ends(cLv, '흑발')}.`);
    } else {
      const colorName = tone?.sure ? `${tone.adj} ${tone.label}의 ${color.label}` : color.label;
      s.push(pick([`컬러는 ${ends(cLv, colorName)}.`, `머리색은 ${ends(cLv, colorName)}.`]));
    }
    s.push(hedge(color));
    if (colorTech.label !== '전체 염색') s.push(say(colorTech, COLOR_TECH[colorTech.label] ?? `${colorTech.label} 기법으로 포인트를 줬어요.`, `${colorTech.label} 기법으로 포인트를 준 것으로 보여요.`));
    if (styling.label !== '풀어내린 머리') s.push(say(styling, STYLING[styling.label] ?? `${josa(styling.label, '으로/로')} 연출했어요.`, `${josa(styling.label, '으로/로')} 연출한 것으로 보여요.`));
    s.push(BR);
    const adj = HAIR_MOOD[mood.label] ?? mood.label;
    s.push(say(mood, [`전체적으로 ${adj} 분위기의 헤어예요.`, `${adj} 무드가 잘 살아 있는 스타일이에요.`], `전체적으로 ${adj} 분위기에 가까워요.`));
    if (mood.level === 'high' && HAIR_SCENE[mood.label]) s.push(HAIR_SCENE[mood.label]);

    // SNS 트렌드 이름 (속성 조합)
    const T = [...PRE];
    const is = (attr, re) => re.test(attr.label) && sure(attr);
    if (tone?.sure && !color.family_of && /브라운|블론드|베이지/.test(color.label)) T.push(trend(`${tone.adj} ${color.label}`, '컬러 + 톤'));
    if (is(cut, /^레이어드컷$/) && is(perm, /^C컬펌$/)) T.push(trend('레이어드 C컬', '레이어드컷 + C컬펌'));
    else if (is(cut, /^레이어드컷$/) && is(perm, /S컬|빌드/)) T.push(trend('레이어드펌', '레이어드컷 + 펌'));
    if (is(cut, /^허쉬컷$/) && is(perm, /S컬|히피|젤리|물결/)) T.push(trend('허쉬펌', '허쉬컷 + 펌'));
    if (is(cut, /칼단발|보브컷/) && is(perm, /^생머리$/)) T.push(trend('슬릭 단발', '단발 커트 + 생머리'));
    if (is(cut, /^보브컷$/) && is(bangs, /풀뱅|시스루뱅/)) T.push(trend('프렌치 보브', '보브컷 + 앞머리'));
    if (is(length, /^단발$/) && is(perm, /^C컬펌$/)) T.push(trend('단발 C컬펌', '단발 + C컬'));
    if (is(styling, /슬릭번|포니테일/) && is(perm, /^생머리$/) && is(bangs, /앞머리 없음/)) T.push(trend('클린걸 헤어', '매끈하게 묶은 머리'));
    if (is(color, /^흑발$/) && is(perm, /^생머리$/) && is(length, /^긴머리$/)) T.push(trend('흑발 생머리', '청순 헤어의 대명사'));
    if (is(color, /애쉬그레이|애쉬블론드|블루블랙|비비드 레드/) && is(mood, /시크|힙/)) T.push(trend('쇠맛 헤어', '차갑고 강렬한 컬러'));
    if (is(cut, /크롭컷|페이드컷|포마드|가일컷|버즈컷/)) T.push(trend('테토남 헤어', '짧고 선명한 남자 커트'));
    if (is(cut, /댄디컷|쉼표머리|애즈펌|리프컷/)) T.push(trend('에겐남 헤어', '부드러운 남자 커트'));
    if (is(perm, /빌드펌|히피펌|젤리펌|물결펌/)) T.push(trend(perm.label, '펌'));
    if (is(cut, /히메컷|허쉬컷|칼단발|울프컷|빅시컷|멀릿컷/)) T.push(trend(cut.label, '커트'));
    if (is(colorTech, /^발레아쥬$/)) T.push(trend('컬러 멜팅', '경계 없이 번지는 염색'));
    if (is(colorTech, /^이너컬러$/)) T.push(trend('이너컬러', '컬러 기법'));
    if (T.length) s.push(trendLine(T[0]));

    const headColor = tone?.sure ? `${tone.adj} ${color.label}` : color.label;
    const men = MEN_CUT.test(cut.label);
    // 장르 카드에는 확실하지 않은 커트를 내세우지 않는다 (기장 + 펌으로 대신한다)
    const genre = withCut
      ? genreOf(cut, !plain && !men ? `${cut.label} + ${perm.label}` : cut.label, { sub: `${length.label} · ${headColor}` })
      : genreOf(weakest(length, perm), `${length.label} ${perm.label}`, { group: '기장 · 펌', second: null, sub: headColor });
    return {
      genre,
      headline: `${headColor} · ${length.label} ${cut.label} · ${perm.label}`,
      sentences: s,
      trends: T,
    };
  },

  // ─────────────────────────────── 네일
  nail(a, P) {
    const { part, shape, length, color, design, layout, finish, mood } = a;
    const s = [];
    const pedi = part?.label === '패디' && part.level !== 'low';
    const D = `${design.label} ${pedi ? '패디' : '네일'}`;
    if (pedi) {
      const lv = weakest(design, color, part);
      s.push(pick([`발톱에 ${color.label} 컬러를 올린 ${ends(lv, D)}.`, `${color.label} 컬러로 칠한 ${ends(lv, D)}.`]));
    } else {
      // 첫 문장: 길이 · 쉐입 · 컬러 · 디자인. 길이 · 쉐입이 불확실하면 빼고 말한다
      const useShape = shape.level !== 'low', useLen = length.level !== 'low';
      if (!useShape) omitted++;
      if (!useLen) omitted++;
      const lv = weakest(design, color, useShape && shape, useLen && length);
      const lenNP = { 숏네일: '짧고 깔끔한 숏네일', 미디엄: '적당한 미디엄 길이', 롱네일: '길게 연장한 롱네일' }[length.label] ?? length.label;
      const form = [useLen && lenNP, useShape && `${shape.label} 쉐입`].filter(Boolean).join('에 ');
      const variants = [`${form}, ${color.label} 컬러를 올린 ${ends(lv, D)}.`];
      // 두 문장으로 나누는 표현은 손톱 모양까지 단정할 수 있을 때만 쓴다
      if (lv.level === 'high') variants.push(`${color.label} 컬러로 완성한 ${ends(lv, D)}. 손톱은 ${useShape ? `${form}으로 다듬었어요` : ida(form)}.`);
      s.push(form ? pick(variants) : `${color.label} 컬러를 올린 ${ends(lv, D)}.`);
    }
    s.push(hedge(design));
    s.push(hedge(shape));
    const info = DESIGN_INFO[design.label];
    if (info) s.push(say(design, [`${info} 디자인이 포인트예요.`, `포인트는 ${ida(info)}.`], `${info} 디자인으로 보여요.`));
    const second = design.all[1];
    if (second && second.score >= 0.2 && second.score >= design.score * 0.5) s.push(pick([`${second.label} 요소도 함께 들어가 있어요.`, `${second.label} 느낌도 함께 보여요.`]));
    s.push(BR);
    const Lay = layout && NAIL_LAYOUT[layout.label];
    if (Lay) s.push(say(layout, Lay[0], Lay[1]));
    const fin = NAIL_FINISH[finish.label] ?? finish.label;
    s.push(say(finish, [`마감은 ${fin} 마감이에요.`, `${fin} 마감으로 마무리했어요.`], `마감은 ${fin} 마감으로 보여요.`));
    const adj = NAIL_MOOD[mood.label] ?? mood.label;
    s.push(say(mood, [`전체적으로 ${adj} 무드의 네일이에요.`, `${adj} 분위기가 잘 느껴져요.`], `전체적으로 ${adj} 분위기에 가까워요.`));

    // 트렌드
    const T = [...PRE];
    // 반사광 마감만으로는 부족하고, 크롬 디자인 · 실버 컬러 · 쇠맛 무드 중 하나가 뚜렷해야 한다
    const soemat = Math.max(P('design', '크롬'), P('color', '실버'), P('mood', '쇠맛')) + 0.3 * P('finish', '메탈릭');
    if (soemat >= TREND_MIN) T.push(trend('쇠맛 네일', '차갑고 메탈릭한 크롬·실버'));
    for (const [label, [name, why]] of Object.entries(NAIL_TREND)) if (P('design', label) >= TREND_MIN) T.push(trend(name, why));
    if (P('finish', '투명·쉬어') >= 0.6) T.push(trend('시럽 네일', '맑게 비치는 젤리 컬러'));
    if (sure(layout) && layout.label === '오마카세') T.push(trend('오마카세 네일', '손톱마다 다른 디자인'));
    if (sure(layout) && layout.label === '퐁당퐁당') T.push(trend('퐁당퐁당 네일', '번갈아 칠한 컬러'));
    if (/프렌치|그라데이션|마블|원컬러|글리터|파츠|드로잉|체크|플라워|스트라이프|캐릭터|과일/.test(design.label) && sure(design)) T.push(trend(`${design.label} 네일`, '디자인'));
    const MOOD_T = { 키치: ['키치 네일', '통통 튀는 파츠와 컬러'], 여름: ['여름 네일', '청량한 바캉스 컬러'], '겨울·홀리데이': ['크리스마스 네일', '연말 홀리데이'], 오피스: ['오피스 네일', '출근룩에도 무난한'], 웨딩: ['웨딩 네일', '웨딩 촬영·본식용'] };
    if (sure(mood) && MOOD_T[mood.label]) T.push(trend(...MOOD_T[mood.label]));
    if (pedi) T.push(trend('패디 네일', '발톱 네일'));
    const lead = T.find((t) => t.why !== '디자인' && t.name !== '패디 네일');
    s.push(BR);
    const ro = lead && josa(lead.name, '으로/로').slice(lead.name.length);   // '자석 그라데이션'으로 · '원컬러 네일'로
    if (lead) s.push(pick([`요즘 SNS에서 '${lead.name}'${ro} 불리는 ${lead.why} 스타일에 가까워요.`, `SNS에서는 '${lead.name}'${ro} 많이 찾는 스타일이에요.`]));

    const ct = color.all[0].tone;
    if ((ct === 'cool' || ct === 'warm') && sure(color)) s.push(toneLine(ct === 'cool' ? '쿨톤' : '웜톤'));

    return {
      genre: genreOf(design, D, { sub: pedi ? `${color.label} · 패디` : `${color.label} · ${shape.label} · ${length.label}` }),
      headline: `${pedi ? color.label : `${color.label} ${shape.label} ${length.label}`} · ${D}${lead ? ` (${lead.name})` : ''}`,
      sentences: s,
      trends: T,
    };
  },

  // ─────────────────────────────── 메이크업
  makeup(a, P) {
    const { base, eye, eyeLine, lash, brow, lip, lipTexture, cheek, detail } = a;
    // 성별을 짐작하는 라벨은 틀리면 불쾌하므로, 확실하지 않으면 다음 후보 무드로 말한다
    let mood = a.mood;
    if (mood.label === '남자 메이크업' && mood.level === 'low') {
      const alt = mood.all.find((x) => x.label !== '남자 메이크업' && !x.hidden);
      if (alt) mood = { ...mood, label: alt.label, score: alt.score, level: 'low', all: mood.all.filter((x) => x.label !== '남자 메이크업') };
    }
    const tone = toneOf(a);
    const s = [];
    const mk = (label) => (/메이크업$/.test(label) ? label : `${label} 메이크업`);
    const moodNP = mk(MAKEUP_MOOD[mood.label] ?? mood.label);
    s.push(pick([`${ends(mood, moodNP)}.`, `전체적으로 ${ends(mood, moodNP)}.`]));
    s.push(hedge(mood));
    const bp = BASE[base.label] ?? base.label;
    s.push(say(base, [`피부는 ${bp} 피부로 표현했어요.`, `베이스는 ${bp} 피부로 연출했어요.`], `피부는 ${bp} 피부에 가까워요.`));

    s.push(BR);
    // 눈: 섀도와 아이라인을 한 문장으로 잇는다
    if (eye?.level === 'low') omitted++;
    if (eyeLine?.level === 'low') omitted++;
    const sh = shown(eye) && SHADOW[eye.label] ? eye : null;
    const ln = shown(eyeLine) && LINER[eyeLine.label] ? eyeLine : null;
    if (sh || ln) {
      const lv = weakest(sh, ln);
      const parts = [sh && SHADOW[sh.label], ln && LINER[ln.label]].filter(Boolean);
      const verb = lv.level === 'high' ? '포인트를 줬어요' : '포인트를 준 것으로 보여요';
      s.push(parts.length === 2
        ? pick([`눈은 ${parts[0]}에 ${josa(parts[1], '을/를')} 더해 ${verb}.`, `눈매는 ${josa(parts[0], '과/와')} ${josa(parts[1], '으로/로')} ${verb}.`])
        : `눈은 ${josa(parts[0], '으로/로')} ${verb}.`);
    } else if (shown(eye) && eye.label === '섀도 없음') {
      s.push(say(eye, '눈두덩에는 섀도를 거의 쓰지 않아 깔끔해요.', '섀도는 거의 쓰지 않은 것으로 보여요.'));
    }
    if (sh) s.push(hedge(eye));
    if (shown(lash) && LASH[lash.label]) {
      const ln2 = LASH[lash.label];
      s.push(say(lash, [`속눈썹은 ${josa(ln2, '으로/로')} 강조했어요.`, `${josa(ln2, '으로/로')} 눈매를 더 또렷하게 만들었어요.`], `속눈썹은 ${josa(ln2, '으로/로')} 강조한 것으로 보여요.`));
    } else if (lash?.level === 'low') omitted++;
    if (brow && BROW[brow.label]) s.push(say(brow, BROW[brow.label][0], BROW[brow.label][1]));

    s.push(BR);
    const tp = LIP_TEX[lipTexture.label] ?? lipTexture.label;
    if (lipTexture.level === 'low') s.push(say(lip, [`입술은 ${lip.label} 컬러예요.`, `립은 ${lip.label} 컬러를 골랐어요.`], `입술은 ${lip.label} 컬러로 보여요.`));
    else {
      const w = weakest(lip, lipTexture);
      if (w.level === 'low') omitted++;
      else s.push(pick(w.level === 'high'
        ? [`입술은 ${lip.label} 컬러를 ${tp} 느낌으로 연출했어요.`, `립은 ${lip.label} 컬러에 ${tp} 표현이에요.`]
        : [`입술은 ${lip.label} 컬러를 ${tp} 느낌으로 연출한 것으로 보여요.`]));
    }
    if (lip.level !== 'low') s.push(hedge(lip));
    if (cheek.label !== '미니멀' && CHEEK[cheek.label]) s.push(say(cheek, CHEEK[cheek.label][0], CHEEK[cheek.label][1]));
    if (detail && DETAIL[detail.label]) s.push(say(detail, DETAIL[detail.label][0], DETAIL[detail.label][1]));
    s.push(BR);
    if (tone?.sure && tone.label !== '뉴트럴') s.push(toneLine(tone.label));

    const T = [...PRE];
    for (const [label, name] of Object.entries(MAKEUP_TREND)) if (P('mood', label) >= TREND_MIN) T.push(trend(name, '무드'));
    if (P('cheek', '숙취 블러셔') >= TREND_MIN) T.push(trend('숙취 메이크업', '눈 밑에 올린 블러셔'));
    if (P('base', '물광') >= TREND_MIN) T.push(trend('물광 메이크업', '촉촉한 물광 피부'));
    if (P('base', '윤광') >= TREND_MIN) T.push(trend('윤광 메이크업', '은은한 윤광 피부'));
    if (P('cheek', '홍조 블러셔') >= TREND_MIN) T.push(trend('홍조 메이크업', '코와 볼에 번진 블러셔'));
    if (P('lipTexture', '립라인 강조') >= TREND_MIN) T.push(trend('90년대 립', '라이너로 살린 입술선'));
    if (P('lipTexture', '블러립') >= TREND_MIN) T.push(trend('블러립', '경계를 흐린 입술'));
    if (P('lipTexture', '그라데이션립') >= TREND_MIN) T.push(trend('그라데이션립', '안쪽부터 번지는 입술'));
    if (P('detail', '큐빅·파츠') >= TREND_MIN) T.push(trend('파츠 메이크업', '큐빅으로 꾸민 무대 메이크업'));
    if (P('eye', '블루 섀도') >= TREND_MIN) T.push(trend('블루 섀도', '시원한 블루 아이'));
    if (P('eyeLine', '캣아이라인') >= TREND_MIN) T.push(trend('고양이 눈매', '올려 뺀 아이라인'));
    if (P('eyeLine', '강아지 라인') >= TREND_MIN) T.push(trend('강아지 눈매', '처지게 뺀 아이라인'));
    if (P('eyeLine', '언더라인') >= TREND_MIN) T.push(trend('언더라인', '아래까지 채운 아이라인'));
    if (tone?.sure && tone.label !== '뉴트럴') T.push(trend(`${tone.label} 메이크업`, '퍼스널컬러'));
    // 첫 문장의 무드와 같은 이름은 다시 말하지 않는다
    const lead = T.find((t) => !/톤 메이크업/.test(t.name) && t.name !== MAKEUP_TREND[mood.label]);
    if (lead) s.push(trendLine(lead));

    return {
      genre: genreOf(mood, mk(mood.label), { sub: `${base.label} 피부 · ${lip.label} ${lipTexture.label}` }),
      headline: `${mk(mood.label)} · ${base.label} 피부 · ${lip.label} ${lipTexture.label}`,
      sentences: s,
      trends: T,
    };
  },

  // ─────────────────────────────── 타투 (트렌드 없이 장르 중심)
  tattoo(a) {
    const { style, color, subject, placement, size } = a;
    const s = [];
    const info = GENRE_INFO[style.label];
    s.push(pick([`장르는 ${ends(style, `${style.label} 타투`)}.`, `${ends(style, `${style.label} 장르의 타투`)}.`]));
    if (info) s.push(`${josa(style.label, '은/는')} ${ida(info)}.`);
    const other = style.level !== 'low' && aka('style', style.label);
    if (other && rand() < 0.5) s.push(akaLine(style.label, other));
    const second = style.all[1];
    if (second && second.score >= 0.2 && second.score >= style.score * 0.5) s.push(`${second.label} 요소도 함께 보여요(${pct(style.score)} 대 ${pct(second.score)}).`);
    else s.push(hedge(style));
    s.push(BR);
    const sizeNP = `${SIZE_NP[size.label] ?? size.label} 사이즈`;
    const cp = COLOR_PHRASE[color.label] ?? `${color.label} 컬러로`;
    if (placement.level !== 'low' && size.level !== 'low' && color.level !== 'low') {
      const hi = weakest(placement, size, color).level === 'high';
      const where = COVER.test(size.label) ? josa(sizeNP, '으로/로') : `${placement.label}에 ${josa(sizeNP, '으로/로')}`;
      s.push(pick(hi
        ? [`${where} 새겼고, ${cp} 작업했어요.`, ...(COVER.test(size.label) ? [] : [`${cp} 작업한 ${sizeNP} 타투로, 부위는 ${ida(placement.label)}.`])]
        : [`${where} 새겼고, ${cp} 작업한 것으로 보여요.`]));
    } else {
      s.push(say(placement, [`부위는 ${ida(placement.label)}.`, `${placement.label}에 새긴 타투예요.`], `부위는 ${josa(placement.label, '으로/로')} 보여요.`));
      s.push(say(size, `크기는 ${ida(sizeNP)}.`, `크기는 ${sizeNP} 정도로 보여요.`));
      s.push(say(color, `${cp} 작업했어요.`, `${cp} 작업한 것으로 보여요.`));
    }
    if (placement.level !== 'low') s.push(hedge(placement));
    // 세부 모티브(장미 · 늑대 …)가 읽히면 그것으로 말하고, 아니면 큰 묶음(꽃·식물 …)으로 말한다
    const mo = (subject.motifs || []).filter((m, i) => m.level !== 'low' || (i === 0 && m.score >= 0.25));
    const headMo = (m) => (m && m.level !== 'low' ? m.ko : subject.label);   // 제목에는 확신이 있을 때만 세부 모티브
    const m0 = mo[0], m1 = m0 && subject.motifs[1]?.score >= 0.15 ? subject.motifs[1] : null;   // 두 번째 모티브는 요청서 · 화면과 같은 기준(15%)
    if (m0) {
      const both = m1 ? `${josa(m0.ko, '과/와')} ${m1.ko}` : m0.ko;
      s.push(m0.level === 'high'
        ? pick([`도안의 주인공은 ${ida(m0.ko)}.`, `${josa(both, '을/를')} 모티프로 그린 도안이에요.`, `도안은 ${both} 모티프예요.`])
        : pick([`도안은 ${both} 모티프로 보여요.`, `${josa(m0.ko, '을/를')} 그린 도안으로 보여요.`]));
      if (m1 && m0.level === 'high' && !s[s.length - 1].includes(m1.ko)) s.push(`${m1.ko}도 함께 들어가 있어요.`);
      if (m0.parent && m0.parent !== m0.ko && rand() < 0.5) s.push(`큰 분류로는 ${m0.parent} 도안이에요.`);
    } else {
      s.push(say(subject, [`도안은 ${subject.label} 모티프예요.`, `${josa(subject.label, '을/를')} 모티프로 그렸어요.`], `도안은 ${subject.label} 모티프로 보여요.`));
      if (subject.level !== 'low') s.push(hedge(subject));
    }
    const T = [...PRE];
    if (T.length) { s.push(BR); s.push(trendLine(T[0])); }
    return {
      genre: genreOf(style, `${style.label} 타투`, { sub: `${placement.label} · ${headMo(m0)} · ${color.label}`, info }),
      headline: `${style.label} 타투 · ${placement.label} · ${headMo(m0)} · ${color.label}`,
      sentences: s,
      trends: T,
    };
  },
};
