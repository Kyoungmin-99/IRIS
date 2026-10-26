// 한국어 문장 조립기 — 한국 인스타그램 · 네이버 뷰티 말투 기준
// analyzer 가 뽑은 그룹별 속성(라벨 + 점수 분포)을 받아
//   - 사람이 설명하듯 상세한 문장
//   - 해시태그
//   - SNS 트렌드 키워드 (예: '쇠맛 네일', '레이어드펌', '차가운 애쉬브라운')
// 를 만든다.
// 신뢰도 high : 단정 ("~입니다") / mid : "~로 보입니다" / low : "~로 추정됩니다" + 다른 후보

import { TAXONOMY, TONE_WORDS, CONFIDENCE } from './taxonomy.js';

const MAX_HEDGES = 2; // 설명문 속 불확실성 언급은 최대 2번 (나머지는 막대그래프로 확인)
let hedgeBudget = MAX_HEDGES;

export function compose(category, attributes) {
  hedgeBudget = MAX_HEDGES;
  const a = Object.fromEntries(attributes.map((x) => [x.group, x]));
  const P = (group, label) => a[group]?.all.find((i) => i.label === label)?.score ?? 0;
  const out = COMPOSERS[category](a, P);
  const trends = uniqBy(out.trends.filter(Boolean), (t) => t.name).slice(0, 4);
  const tags = uniq([
    ...trends.map((t) => t.name.replace(/[\s·()]/g, '')),
    ...attributes.flatMap((x) => tagFor(category, x)),
  ]).slice(0, 12);
  const sentences = out.sentences.filter(Boolean);
  return { headline: out.headline, genre: out.genre, sentences, description: sentences.join(' '), tags, trends };
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
// 명사 + 신뢰도별 서술어: said('메이크업', a) → '메이크업입니다' / '메이크업으로 보입니다'
function said(noun, attr, soft = false) {
  if (attr.level === 'high' && !soft) return `${noun}입니다`;
  if (attr.level === 'low') return `${josa(noun, '으로/로')} 추정됩니다`;
  return `${josa(noun, '으로/로')} 보입니다`;
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
  if (attr.level === 'mid') return `${alt.label}일 가능성도 있어요(${pct(attr.score)} 대 ${pct(alt.score)}).`;
  const cands = alts.slice(0, 2).map((x) => x.label).join(' 또는 ');
  // 두 번째 보충 문장은 '다만 …' 이 반복되지 않게 표현을 바꾼다
  if (!first) return `${attr.group_label}도 확실하지 않아 ${cands}일 가능성이 있어요.`;
  return `다만 ${josa(attr.group_label, '은/는')} 사진만으로 단정하기 어려워 ${cands}일 수도 있어요.`;
}
const sure = (attr, min = CONFIDENCE.mid) => attr && attr.score >= min; // 보정 확률 0.5 이상 ≈ 82% 적중
const TREND_MIN = CONFIDENCE.mid; // 트렌드 키워드는 칩으로 단정해 보이므로 같은 기준을 쓴다
const uniq = (arr) => [...new Set(arr.filter(Boolean))];
const uniqBy = (arr, f) => { const s = new Set(); return arr.filter((x) => (s.has(f(x)) ? false : s.add(f(x)))); };
const trend = (name, why) => ({ name, why });

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

// 타투 장르 설명 (장르 판정 뒤에 한 줄로 덧붙인다)
const GENRE_INFO = {
  레터링: '글자와 문구를 필기체나 타이포그래피로 새기는 장르',
  파인라인: '가는 바늘로 섬세한 선만 살려 표현하는 장르',
  미니멀: '작고 단순한 선과 면으로 상징만 남기는 장르',
  블랙워크: '검은 잉크로 면을 넓게 채워 강한 대비를 주는 장르',
  블랙앤그레이: '검정과 회색 음영만으로 입체감을 살리는 장르',
  '컬러 리얼리즘': '사진처럼 사실적인 묘사를 컬러로 표현하는 장르',
  올드스쿨: '굵은 외곽선과 빨강·초록·노랑 같은 원색을 쓰는 미국 전통 장르',
  네오트래디셔널: '올드스쿨의 굵은 선에 풍부한 색감과 장식적 디테일을 더한 장르',
  뉴스쿨: '만화처럼 과장된 형태와 강렬한 컬러가 특징인 장르',
  이레즈미: '용·잉어·파도·모란 같은 소재를 넓은 부위에 새기는 일본 전통 장르',
  수채화: '물감이 번진 듯한 색 번짐과 붓 터치를 살린 장르',
  일러스트: '펜 드로잉이나 스케치 같은 그림체를 살린 장르',
  도트워크: '수많은 점을 찍어 만다라나 기하학 패턴을 만드는 장르',
  트라이벌: '폴리네시아·마오리 같은 부족 문양을 굵은 검은 패턴으로 표현하는 장르',
  사이버시길리즘: '가시처럼 날카롭고 가느다란 곡선이 흐르는 사이버 감성 장르',
};
const COLOR_PHRASE = { 블랙: '검은 잉크 선 위주로', 블랙앤그레이: '블랙앤그레이 음영으로', 풀컬러: '선명한 풀컬러로', 포인트컬러: '블랙에 포인트 컬러를 더해' };

// ---------------------------------------------------------------------------
const COMPOSERS = {
  // ─────────────────────────────── 헤어
  hair(a, P) {
    const { length, cut, styling, perm, bangs, color, colorTech, mood } = a;
    const tone = toneOf(a);
    const s = [];
    const permPhrase = perm.label === '생머리' ? '매끈한 생머리' : perm.label === '내추럴 곱슬' ? '내추럴 곱슬머리' : josa(perm.label, '을/를') + ' 더한';
    if (perm.label === '생머리' || perm.label === '내추럴 곱슬') s.push(`${length.label} 기장의 ${cut.label}, ${said(`${permPhrase} 스타일`, cut)}.`);
    else s.push(`${length.label} 기장의 ${cut.label}에 ${permPhrase} ${said('스타일', cut)}.`);
    s.push(hedge(cut));
    s.push(hedge(length));

    if (bangs.label === '확인 불가') s.push('뒷모습 사진이라 앞머리는 확인되지 않아요.');
    else if (bangs.label === '앞머리 없음') s.push('앞머리 없이 이마를 드러내 깔끔한 인상이에요.');
    else s.push(`앞머리는 ${josa(bangs.label, '으로/로')} 내려 얼굴선을 자연스럽게 감싸 줘요.`);

    const colorName = tone?.sure ? `${tone.adj} ${tone.label}의 ${color.label}` : color.label;
    s.push(`컬러는 ${said(colorName, color)}.`);
    s.push(hedge(color));
    if (colorTech.label !== '전체 염색' && sure(colorTech)) s.push(`${colorTech.label} 기법으로 컬러에 포인트를 줬어요.`);
    if (styling.label !== '풀어내린 머리' && sure(styling)) s.push(`${josa(styling.label, '으로/로')} 연출해 분위기를 살렸어요.`);
    s.push(`전체적으로 ${mood.label} 무드가 느껴지는 헤어예요.`);

    // SNS 트렌드 이름 (속성 조합)
    const T = [];
    if (tone?.sure && /브라운|블론드/.test(color.label)) T.push(trend(`${tone.adj} ${color.label}`, '컬러 + 톤'));
    if (cut.label === '레이어드컷' && /C컬|S컬|빌드/.test(perm.label) && sure(cut) && sure(perm)) T.push(trend('레이어드펌', '레이어드컷 + 펌'));
    if (cut.label === '태슬컷' && perm.label === '생머리' && sure(cut) && sure(perm)) T.push(trend('칼단발', '태슬컷 + 생머리'));
    if (length.label === '단발' && perm.label === 'C컬펌' && sure(length) && sure(perm)) T.push(trend('단발 C컬펌', '단발 + C컬'));
    if (cut.label === '허쉬컷' && perm.label !== '생머리' && sure(cut)) T.push(trend('허쉬컷 펌', '허쉬컷 + 펌'));
    if (color.label === '흑발' && perm.label === '생머리' && length.label === '긴머리' && sure(color) && sure(perm) && sure(length)) T.push(trend('흑발 생머리', '청순 헤어의 대명사'));
    if (/빌드펌|히피펌/.test(perm.label) && sure(perm)) T.push(trend(perm.label, '펌'));
    if (/히메컷|허쉬컷|태슬컷|울프컷/.test(cut.label) && sure(cut)) T.push(trend(cut.label, '커트'));
    if (colorTech.label === '이너컬러' && sure(colorTech)) T.push(trend('이너컬러', '컬러 기법'));
    if (T.length) s.push(`SNS에서는 '${T[0].name}' 스타일로 많이 찾아요.`);

    const headColor = tone?.sure ? `${tone.adj} ${color.label}` : color.label;
    const permed = !/생머리|내추럴 곱슬/.test(perm.label);
    return {
      genre: genreOf(cut, permed ? `${cut.label} + ${perm.label}` : cut.label, { sub: `${length.label} · ${headColor}` }),
      headline: `${headColor} · ${length.label} ${cut.label} · ${perm.label}`,
      sentences: s,
      trends: T,
    };
  },

  // ─────────────────────────────── 네일
  nail(a, P) {
    const { shape, length, color, design, finish, mood } = a;
    const s = [];
    const lenPhrase = { 숏네일: '짧고 깔끔한 숏네일', 미디엄: '적당한 미디엄 길이', 롱네일: '길게 연장한 롱네일' }[length.label];
    s.push(`${lenPhrase}에 ${shape.label} 쉐입, ${color.label} 컬러를 올린 ${said(`${design.label} 네일`, design)}.`);
    s.push(hedge(design));
    s.push(hedge(shape));
    const second = design.all[1];
    if (second && second.score >= 0.2 && second.score >= design.score * 0.5) s.push(`여기에 ${second.label} 느낌도 더해졌어요.`);
    const finishPhrase = {
      유광: '반짝이는 유광', 매트: '보송한 매트', 메탈릭: '거울처럼 반사되는 메탈릭', '투명·쉬어': '속이 비치는 맑은', '펄·쉬머': '은은하게 빛나는 펄',
    }[finish.label];
    s.push(`마감은 ${finishPhrase} 마감이에요.`);
    s.push(`전체적으로 ${mood.label} 무드의 네일이에요.`);

    // 트렌드: 쇠맛 = 크롬/실버/메탈릭/쇠맛 무드 중 하나라도 뚜렷하면
    const T = [];
    // 반사광 마감만으로는 부족하고, 크롬 디자인 · 실버 컬러 · 쇠맛 무드 중 하나가 뚜렷해야 한다
    const soemat = Math.max(P('design', '크롬'), P('color', '실버'), P('mood', '쇠맛')) + 0.3 * P('finish', '메탈릭');
    if (soemat >= TREND_MIN) T.push(trend('쇠맛 네일', '차갑고 메탈릭한 크롬·실버'));
    if (P('design', '자석') >= TREND_MIN) T.push(trend('자석 네일', '빛에 따라 움직이는 캣아이'));
    if (P('design', '오로라') >= TREND_MIN) T.push(trend('오로라 네일', '각도마다 색이 바뀌는 오로라'));
    if (P('design', '글레이즈드') >= TREND_MIN) T.push(trend('글레이즈드 네일', '도넛처럼 은은한 펄 광택'));
    if (P('design', '시럽') >= TREND_MIN || P('finish', '투명·쉬어') >= 0.6) T.push(trend('시럽 네일', '맑게 비치는 젤리 컬러'));
    if (P('design', '치크') >= TREND_MIN) T.push(trend('치크 네일', '볼터치처럼 물든 컬러'));
    if (/프렌치|그라데이션|마블|원컬러|글리터|파츠|드로잉|체크|플라워/.test(design.label) && sure(design)) T.push(trend(`${design.label} 네일`, '디자인'));
    if (mood.label === '오피스' && sure(mood)) T.push(trend('오피스 네일', '출근룩에도 무난한'));
    if (mood.label === '웨딩' && sure(mood)) T.push(trend('웨딩 네일', '웨딩 촬영·본식용'));
    const lead = T.find((t) => /쇠맛|자석|오로라|글레이즈드|시럽|치크/.test(t.name));
    if (lead) s.push(`요즘 SNS에서 '${lead.name}'로 불리는 ${lead.why} 스타일이에요.`);

    const ct = color.all[0].tone;
    if (ct === 'cool' && sure(color)) s.push('컬러가 쿨톤 계열이라 여름·겨울 쿨톤에게 잘 어울려요.');
    if (ct === 'warm' && sure(color)) s.push('컬러가 웜톤 계열이라 봄·가을 웜톤에게 잘 어울려요.');

    return {
      genre: genreOf(design, `${design.label} 네일`, { sub: `${color.label} · ${shape.label} · ${length.label}` }),
      headline: `${color.label} ${shape.label} ${length.label} · ${design.label} 네일${lead ? ` (${lead.name})` : ''}`,
      sentences: s,
      trends: T,
    };
  },

  // ─────────────────────────────── 메이크업
  makeup(a, P) {
    const { mood, base, eye, lip, lipTexture, cheek } = a;
    const tone = toneOf(a);
    const s = [];
    const moodPhrase = {
      데일리: '힘을 뺀 데일리', 청순: '맑고 깨끗한 청순', 음영: '브라운 톤으로 입체감을 준 음영', 과즙: '생기가 도는 과즙', 글램: '화려한 글램',
      스모키: '깊고 강렬한 스모키', 쇠맛: '차갑고 메탈릭한 쇠맛', 레트로: '클래식한 레트로', 웨딩: '우아한 웨딩', 아트: '개성 강한 아트',
    }[mood.label];
    s.push(`${said(`${moodPhrase} 메이크업`, mood)}.`);
    s.push(hedge(mood));
    const basePhrase = { 물광: '물기를 머금은 듯 촉촉한 물광', 윤광: '은은하게 빛나는 윤광', 세미매트: '자연스러운 세미매트', '보송 매트': '보송하게 정돈한 매트' }[base.label];
    s.push(`피부는 ${basePhrase} 피부로 표현했어요.`);
    s.push(`눈은 ${josa(eye.label, '으로/로')} 포인트를 줬어요.`);
    s.push(hedge(eye));
    const texPhrase = { 글로시: '촉촉한 글로시', 매트: '벨벳 같은 매트', 블러립: '경계를 흐린 블러립', 그라데이션립: '안쪽부터 번지는 그라데이션립', 오버립: '입술선을 살짝 넘긴 오버립' }[lipTexture.label];
    s.push(`입술은 ${lip.label} 컬러를 ${josa(texPhrase, '으로/로')} 연출했어요.`);
    s.push(hedge(lip));
    if (cheek.label !== '미니멀') s.push(`볼에는 ${josa(cheek.label, '을/를')} 더했어요.`);
    if (tone?.sure && tone.label !== '뉴트럴') {
      s.push(`전체 컬러가 ${tone.label} 계열이라 ${tone.label === '쿨톤' ? '여름·겨울 쿨톤' : '봄·가을 웜톤'}에게 잘 어울려요.`);
    }

    const T = [];
    if (P('mood', '쇠맛') >= TREND_MIN) T.push(trend('쇠맛 메이크업', '차갑고 메탈릭한 사이버 무드'));
    if (P('base', '물광') >= TREND_MIN) T.push(trend('물광 메이크업', '촉촉한 물광 피부'));
    if (P('base', '윤광') >= TREND_MIN) T.push(trend('윤광 메이크업', '은은한 윤광 피부'));
    if (P('mood', '과즙') >= TREND_MIN) T.push(trend('과즙 메이크업', '생기 있는 볼과 입술'));
    if (P('mood', '음영') >= TREND_MIN) T.push(trend('음영 메이크업', '브라운 음영'));
    if (P('mood', '청순') >= TREND_MIN) T.push(trend('청순 메이크업', '맑은 피부와 핑크 톤'));
    if (P('cheek', '홍조 블러셔') >= TREND_MIN) T.push(trend('홍조 메이크업', '코와 볼에 번진 블러셔'));
    if (P('lipTexture', '블러립') >= TREND_MIN) T.push(trend('블러립', '경계를 흐린 입술'));
    if (P('lipTexture', '그라데이션립') >= TREND_MIN) T.push(trend('그라데이션립', '안쪽부터 번지는 입술'));
    if (P('eye', '캣아이라인') >= TREND_MIN) T.push(trend('고양이 눈매', '올려 뺀 아이라인'));
    if (P('eye', '강아지 눈매') >= TREND_MIN) T.push(trend('강아지 눈매', '처지게 뺀 아이라인'));
    if (tone?.sure && tone.label !== '뉴트럴') T.push(trend(`${tone.label} 메이크업`, '퍼스널컬러'));
    const lead = T[0];
    if (lead && !/톤 메이크업/.test(lead.name)) s.push(`요즘 SNS에서 말하는 '${lead.name}'에 가까워요.`);

    return {
      genre: genreOf(mood, `${mood.label} 메이크업`, { sub: `${base.label} 피부 · ${lip.label} ${lipTexture.label}` }),
      headline: `${mood.label} 메이크업 · ${base.label} 피부 · ${lip.label} ${lipTexture.label}`,
      sentences: s,
      trends: T,
    };
  },

  // ─────────────────────────────── 타투 (트렌드 없이 장르 중심)
  tattoo(a) {
    const { style, color, subject, placement, size } = a;
    const s = [];
    s.push(`장르는 ${said(`${style.label} 타투`, style)}.`);
    if (GENRE_INFO[style.label]) s.push(`${josa(style.label, '은/는')} ${GENRE_INFO[style.label]}예요.`);
    const second = style.all[1];
    if (second && second.score >= 0.2 && second.score >= style.score * 0.5) s.push(`${second.label} 요소도 함께 보여요(${pct(style.score)} 대 ${pct(second.score)}).`);
    else s.push(hedge(style));
    s.push(`${placement.label}에 새긴 ${size.label} 크기이고, ${COLOR_PHRASE[color.label]} 작업했어요.`);
    s.push(hedge(placement));
    s.push(`도안은 ${subject.label} 모티프예요.`);
    s.push(hedge(subject));
    return {
      genre: genreOf(style, `${style.label} 타투`, { sub: `${placement.label} · ${subject.label} · ${color.label}`, info: GENRE_INFO[style.label] }),
      headline: `${style.label} 타투 · ${placement.label} · ${subject.label} · ${color.label}`,
      sentences: s,
      trends: [],
    };
  },
};
