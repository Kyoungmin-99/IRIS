// 속성 사전 (Taxonomy) — 한국 인스타그램 · 네이버 뷰티 용어 기준
//
// - 각 카테고리는 여러 "속성 그룹"으로 구성되고, 그룹마다 독립적으로 분류한다.
// - ko   : 화면에 보이는 이름 (한국 SNS 에서 실제로 쓰는 표현)
// - en   : 모델(CLIP)에 넣는 영어 설명 — 구체적일수록 정확도가 오른다
// - tag  : 해시태그 (없으면 ko + 그룹의 tagSuffix 로 만든다)
// - tone : 'cool' | 'warm' | 'neutral' — 컬러 라벨이 가진 톤 (톤 판정에 사전 정보로 쓴다)
// - templates : 라벨(en)을 감싸는 문장 틀. 여러 개를 평균내면(prompt ensembling) 안정적이다.
// - hidden    : 화면 후보·태그에 노출하지 않는 보조 라벨

export const CATEGORY_ORDER = ['hair', 'nail', 'makeup', 'tattoo'];

// 뷰티 사진이 아닌 경우를 가려내기 위한 문장 (이쪽이 이기면 "뷰티 사진이 아닐 수 있어요" 안내)
export const OTHER_DETECT = [
  'a photo of a landscape, a street or a building',
  'a photo of food or a drink',
  'a photo of a pet or a wild animal',
  'a product photo of a cosmetic bottle, a tube or a package on a table',
  'a cartoon, a drawing, a painting or an illustration',
  'a photo of hardware, tools, screws or metal nails',
  'a screenshot or a document with text',
  'a photo of a car, furniture or an everyday object',
];

export const TAXONOMY = {
  // ───────────────────────────────────────────────────────────── 헤어
  hair: {
    label: '헤어',
    icon: '💇',
    detect: [
      'a photo of a person showing their hairstyle',
      "a photo of someone's haircut and hair color",
      'a hair salon photo where the hair is the main subject',
    ],
    groups: [
      {
        key: 'length', label: '기장',
        templates: ['a photo of a person with {}', 'a hairstyle photo showing {}'],
        labels: [
          { ko: '숏컷', en: 'very short cropped hair above the ears' },
          { ko: '단발', en: 'chin-length bob hair that ends around the jawline' },
          { ko: '중단발', en: 'medium-length hair that ends at the shoulders or collarbone' },
          { ko: '긴머리', en: 'long hair that falls well below the shoulders to the chest or back' },
        ],
      },
      {
        key: 'cut', label: '커트',
        templates: ['a photo of a person with a {}', 'a hairstyle photo of a {}'],
        labels: [
          { ko: '레이어드컷', en: 'layered haircut with soft long layers framing the face' },
          { ko: '허쉬컷', en: 'hush cut, a shaggy layered cut with airy wispy textured ends' },
          { ko: '태슬컷', en: 'tassel cut, a blunt one-length bob with straight sharp ends cut in one line' },
          { ko: '히메컷', en: 'hime cut, long straight hair with blunt bangs and straight cheek-length side locks' },
          { ko: '울프컷', en: 'wolf cut, a choppy mullet-like shag with volume on top and a longer wispy back' },
          { ko: '픽시컷', en: 'pixie cut, a very short cropped feminine haircut' },
          { ko: '댄디컷', en: "dandy cut, a neat short men's haircut with a soft fringe covering the forehead" },
          { ko: '투블럭컷', en: 'two block cut with short shaved sides and longer hair on top' },
          { ko: '리프컷', en: "leaf cut, a men's haircut with side-parted curved bangs and short sides" },
          { ko: '포마드·슬릭백', en: 'slicked back hair or a pompadour styled with pomade' },
          { ko: '버즈컷', en: 'buzz cut, hair clipped very short all over the head' },
          { ko: '모히칸', en: 'mohawk with shaved sides and a strip of spiked hair in the middle' },
        ],
      },
      {
        key: 'styling', label: '연출',
        templates: ['a photo of a person with {}', 'a hairstyle with {}'],
        labels: [
          { ko: '풀어내린 머리', en: 'hair worn down loose', tag: '' },
          { ko: '포니테일', en: 'hair tied back in a ponytail' },
          { ko: '똥머리', en: 'hair tied up in a messy bun on top or at the back of the head' },
          { ko: '반묶음', en: 'half-up half-down hair' },
          { ko: '양갈래', en: 'pigtails, hair tied into two sections on both sides' },
          { ko: '땋은머리', en: 'braided hair with braids or cornrows' },
          { ko: '드레드', en: 'dreadlocks' },
          { ko: '업스타일', en: 'an elegant formal updo for a wedding or event' },
        ],
      },
      {
        key: 'perm', label: '펌·질감', tagSuffix: '',
        templates: ['a photo of a person with {}', 'a close-up photo of {}'],
        labels: [
          { ko: '생머리', en: 'sleek straight hair with no curls or waves' },
          { ko: 'C컬펌', en: 'straight hair with the ends curling inward in a soft c-curl' },
          { ko: 'S컬펌', en: 'loose flowing s-shaped waves through the hair' },
          { ko: '빌드펌', en: 'glamorous big voluminous bouncy curls from the mid-lengths' },
          { ko: '히피펌', en: 'hippie perm, small tight voluminous frizzy curls all over' },
          { ko: '내추럴 곱슬', en: 'natural curly or coily afro-textured hair' },
        ],
      },
      {
        key: 'bangs', label: '앞머리',
        templates: ['a photo of a person with {}', 'a hairstyle with {}'],
        labels: [
          { ko: '앞머리 없음', en: 'no bangs, hair swept away from a visible forehead', tag: '' },
          { ko: '풀뱅', en: 'full thick blunt bangs covering the forehead' },
          { ko: '시스루뱅', en: 'thin wispy see-through bangs' },
          { ko: '사이드뱅', en: 'side-swept bangs brushed to one side' },
          { ko: '커튼뱅', en: 'curtain bangs parted in the middle framing the face' },
          { ko: '확인 불가', en: 'the back of the head seen from behind, the forehead is not visible', hidden: true },
        ],
      },
      {
        key: 'color', label: '컬러',
        templates: ['a photo of a person with {}', 'a hair color photo of {}'],
        labels: [
          { ko: '흑발', en: 'natural jet black hair', tone: 'neutral' },
          { ko: '블루블랙', en: 'blue-black hair, black with a cool blue sheen', tone: 'cool' },
          { ko: '초코브라운', en: 'rich dark chocolate brown hair', tone: 'warm' },
          { ko: '밀크브라운', en: 'soft light milky beige brown hair', tone: 'neutral' },
          { ko: '애쉬브라운', en: 'cool ash brown hair with a greyish muted tone', tone: 'cool' },
          { ko: '카키브라운', en: 'khaki brown hair with an olive greenish tint', tone: 'cool' },
          { ko: '골드브라운', en: 'warm golden caramel brown hair', tone: 'warm' },
          { ko: '오렌지브라운', en: 'orange copper brown hair', tone: 'warm' },
          { ko: '레드와인', en: 'wine red burgundy hair', tone: 'cool' },
          { ko: '애쉬블론드', en: 'cool ash or platinum blonde hair', tone: 'cool' },
          { ko: '골드블론드', en: 'warm golden honey blonde hair', tone: 'warm' },
          { ko: '애쉬그레이', en: 'ash grey or silver hair', tone: 'cool' },
          { ko: '핑크·라벤더', en: 'pastel pink or lavender dyed hair', tone: 'cool', tag: '핑크염색' },
          { ko: '블루·퍼플', en: 'vivid blue or purple dyed hair', tone: 'cool', tag: '블루염색' },
          { ko: '그린·민트', en: 'green or mint dyed hair', tone: 'cool', tag: '민트염색' },
          { ko: '비비드 레드', en: 'vivid bright red dyed hair', tone: 'warm' },
        ],
      },
      {
        key: 'colorTech', label: '컬러 기법',
        templates: ['a photo of a person with {}', 'a hair color photo of {}'],
        labels: [
          { ko: '전체 염색', en: 'hair in one single even color from root to tip', tag: '' },
          { ko: '브릿지', en: 'hair with bold streaks of lighter highlights' },
          { ko: '발레아쥬', en: 'balayage, soft hand-painted lighter color through the lengths' },
          { ko: '옴브레', en: 'ombre hair, dark roots gradually fading to a light color at the ends' },
          { ko: '이너컬러', en: 'hidden inner color, a bright color peeking from the underneath layer' },
          { ko: '투톤', en: 'two-tone hair split into two distinct colors' },
        ],
      },
      {
        key: 'tone', label: '톤', tagSuffix: '염색',
        templates: ['a photo of a person with {}', '{}'],
        labels: [
          { ko: '쿨톤', en: 'cool-toned hair color with ashy grey, blue or violet undertones and no warmth' },
          { ko: '웜톤', en: 'warm-toned hair color with golden, orange, copper or reddish undertones' },
          { ko: '뉴트럴', en: 'natural neutral-toned hair color, neither warm nor ashy', tag: '' },
        ],
      },
      {
        key: 'mood', label: '분위기', tagSuffix: '헤어',
        templates: ['a photo of a {} hairstyle', 'a {} hair look'],
        labels: [
          { ko: '청순', en: 'innocent pure and soft-looking' },
          { ko: '러블리', en: 'cute lovely and girly' },
          { ko: '시크', en: 'chic cool and sophisticated' },
          { ko: '힙', en: 'hip edgy bold street-style' },
          { ko: '내추럴', en: 'natural effortless and casual' },
          { ko: '단정·오피스', en: 'neat tidy and professional', tag: '단정한헤어' },
          { ko: '우아·고급', en: 'elegant glamorous and luxurious', tag: '고급스러운헤어' },
          { ko: '레트로', en: 'retro vintage' },
        ],
      },
    ],
  },

  // ───────────────────────────────────────────────────────────── 네일
  nail: {
    label: '네일',
    icon: '💅',
    detect: [
      'a close-up photo of fingernails with nail polish',
      'a photo of a manicure with nail art',
      'a photo of a hand showing painted nails',
    ],
    groups: [
      {
        key: 'shape', label: '쉐입', tagSuffix: '쉐입',
        templates: ['a close-up photo of {} nails', 'a manicure with {} nail shape'],
        labels: [
          { ko: '라운드', en: 'short rounded' },
          { ko: '오벌', en: 'oval shaped' },
          { ko: '스퀘어', en: 'square shaped flat-tipped' },
          { ko: '라운드스퀘어', en: 'squoval, square with softly rounded corners' },
          { ko: '아몬드', en: 'almond shaped tapered' },
          { ko: '코핀', en: 'coffin ballerina shaped, long tapered with a flat squared tip' },
          { ko: '스틸레토', en: 'sharp pointed stiletto' },
        ],
      },
      {
        key: 'length', label: '길이',
        templates: ['a close-up photo of {}', 'a manicure on {}'],
        labels: [
          { ko: '숏네일', en: 'short natural-length nails' },
          { ko: '미디엄', en: 'medium length nails', tag: '' },
          { ko: '롱네일', en: 'very long extended acrylic nails' },
        ],
      },
      {
        key: 'color', label: '컬러', tagSuffix: '네일',
        templates: ['a close-up photo of {} nails', 'a manicure in {}'],
        labels: [
          { ko: '누드·베이지', en: 'nude beige', tone: 'warm', tag: '누드네일' },
          { ko: '밀키화이트', en: 'milky white sheer', tone: 'neutral' },
          { ko: '연핑크', en: 'soft baby pink', tone: 'cool' },
          { ko: '핫핑크', en: 'bright hot pink', tone: 'cool' },
          { ko: '레드', en: 'classic red', tone: 'neutral' },
          { ko: '버건디', en: 'deep burgundy wine', tone: 'cool' },
          { ko: '블랙', en: 'black', tone: 'neutral' },
          { ko: '실버', en: 'metallic silver', tone: 'cool' },
          { ko: '골드', en: 'metallic gold', tone: 'warm' },
          { ko: '네이비·블루', en: 'navy or royal blue', tone: 'cool', tag: '블루네일' },
          { ko: '하늘색', en: 'light sky blue', tone: 'cool' },
          { ko: '민트·그린', en: 'mint or green', tone: 'cool', tag: '민트네일' },
          { ko: '라벤더·퍼플', en: 'lavender or purple', tone: 'cool', tag: '라벤더네일' },
          { ko: '옐로·오렌지', en: 'yellow or orange', tone: 'warm', tag: '옐로네일' },
          { ko: '브라운·모카', en: 'brown mocha', tone: 'warm', tag: '모카네일' },
        ],
      },
      {
        key: 'design', label: '디자인', tagSuffix: '네일',
        templates: ['a close-up photo of {}', 'nail art with {}'],
        labels: [
          { ko: '원컬러', en: 'plain solid one-color nails with no art' },
          { ko: '프렌치', en: 'french tip nails with a contrasting line on the tips' },
          { ko: '그라데이션', en: 'gradient ombre nails fading between colors' },
          { ko: '글리터', en: 'sparkling chunky glitter nails' },
          { ko: '자석', en: 'magnetic cat eye gel nails with a shimmering band of light' },
          { ko: '오로라', en: 'holographic aurora iridescent nails' },
          { ko: '크롬', en: 'mirror chrome metallic nails' },
          { ko: '글레이즈드', en: 'glazed donut nails, a sheer milky base with a pearly chrome sheen' },
          { ko: '시럽', en: 'syrup jelly nails with translucent glossy sheer color' },
          { ko: '마블', en: 'marble pattern nails with swirls' },
          { ko: '치크', en: 'blush nails with a soft rosy flush in the center of each nail' },
          { ko: '파츠', en: 'nails decorated with rhinestones, gems and 3D charms' },
          { ko: '진주', en: 'nails decorated with small pearls' },
          { ko: '드로잉', en: 'hand-painted nail art with detailed drawings' },
          { ko: '플라워', en: 'floral flower pattern nails' },
          { ko: '체크', en: 'checkered or plaid tweed pattern nails' },
          { ko: '하트·리본', en: 'nails with little hearts or ribbon bows', tag: '리본네일' },
          { ko: '도트', en: 'polka dot nails' },
          { ko: '라인아트', en: 'nails with thin minimalist line art' },
        ],
      },
      {
        key: 'finish', label: '마감', tagSuffix: '네일',
        templates: ['a close-up photo of nails with {}', 'a manicure with {}'],
        labels: [
          { ko: '유광', en: 'a glossy shiny finish', tag: '' },
          { ko: '매트', en: 'a flat matte finish with no shine' },
          { ko: '메탈릭', en: 'a reflective metallic finish' },
          { ko: '투명·쉬어', en: 'a translucent sheer see-through finish', tag: '투명네일' },
          { ko: '펄·쉬머', en: 'a pearly shimmery finish', tag: '펄네일' },
        ],
      },
      {
        key: 'mood', label: '분위기', tagSuffix: '네일',
        templates: ['a photo of {} nails', 'a {} manicure'],
        labels: [
          { ko: '오피스', en: 'simple minimal clean everyday office' },
          { ko: '청순·러블리', en: 'cute lovely girly soft', tag: '러블리네일' },
          { ko: '화려한', en: 'glamorous elaborate bling' },
          { ko: '시크·힙', en: 'chic dark edgy', tag: '시크네일' },
          { ko: '쇠맛', en: 'cold futuristic metallic cyber y2k' },
          { ko: '시즌', en: 'seasonal holiday themed christmas halloween or summer' },
          { ko: '웨딩', en: 'elegant bridal wedding' },
        ],
      },
    ],
  },

  // ───────────────────────────────────────────────────────────── 메이크업
  makeup: {
    label: '메이크업',
    icon: '💄',
    detect: [
      'a close-up photo of a face wearing makeup',
      'a beauty photo showing eye makeup and lipstick',
      'a portrait photo focusing on the makeup look',
    ],
    groups: [
      {
        key: 'mood', label: '무드', tagSuffix: '메이크업',
        templates: ['a photo of a face with {} makeup', 'a {} makeup look'],
        labels: [
          { ko: '데일리', en: 'natural minimal everyday no-makeup style' },
          { ko: '청순', en: 'soft innocent fresh pink-toned' },
          { ko: '음영', en: 'soft muted brown contour shading that sculpts the eyes and face' },
          { ko: '과즙', en: 'fresh juicy fruity makeup with rosy flushed cheeks and glossy tinted lips' },
          { ko: '글램', en: 'glamorous full-coverage evening' },
          { ko: '스모키', en: 'dark dramatic smokey eye' },
          { ko: '쇠맛', en: 'cold metallic futuristic cyber makeup with silver chrome shimmer and sharp lines' },
          { ko: '레트로', en: 'retro vintage classic makeup with red lips and winged liner' },
          { ko: '웨딩', en: 'bridal wedding' },
          { ko: '아트', en: 'bold artistic colorful stage or party' },
        ],
      },
      {
        key: 'base', label: '피부 표현', tagSuffix: '피부',
        templates: ['a close-up photo of a face with {}', 'makeup with {}'],
        labels: [
          { ko: '물광', en: 'very dewy wet-look glass skin' },
          { ko: '윤광', en: 'soft satin healthy glow from within' },
          { ko: '세미매트', en: 'natural semi-matte skin' },
          { ko: '보송 매트', en: 'flat powdery matte skin with no shine' },
        ],
      },
      {
        key: 'eye', label: '아이 메이크업',
        templates: ['a close-up photo of eyes with {}', 'eye makeup with {}'],
        labels: [
          { ko: '브라운 음영 섀도', en: 'subtle neutral brown shading eyeshadow', tag: '음영섀도' },
          { ko: '스모키', en: 'dark smokey black eyeshadow' },
          { ko: '글리터·펄', en: 'sparkly glitter shimmer eyeshadow', tag: '글리터섀도' },
          { ko: '컬러 섀도', en: 'colorful blue, purple or green eyeshadow' },
          { ko: '핑크·코랄 섀도', en: 'pink or coral toned eyeshadow', tag: '코랄섀도' },
          { ko: '캣아이라인', en: 'a sharp lifted cat-eye winged eyeliner' },
          { ko: '강아지 눈매', en: 'soft downturned puppy eyeliner' },
          { ko: '애교살 강조', en: 'highlighted aegyo-sal, a shimmery puffy under-eye area' },
          { ko: '인형 속눈썹', en: 'dramatic long separated doll-like lashes' },
        ],
      },
      {
        key: 'lip', label: '립 컬러', tagSuffix: '립',
        templates: ['a close-up photo of lips with {}', 'makeup with {}'],
        labels: [
          { ko: '레드', en: 'classic red lipstick', tone: 'neutral' },
          { ko: '코랄', en: 'coral peach lipstick', tone: 'warm' },
          { ko: '로즈핑크', en: 'rosy pink lipstick', tone: 'cool' },
          { ko: 'MLBB', en: 'nude natural my-lips-but-better lipstick', tone: 'neutral' },
          { ko: '말린장미', en: 'muted dusty rose lipstick', tone: 'cool' },
          { ko: '버건디·플럼', en: 'deep burgundy plum lipstick', tone: 'cool', tag: '버건디립' },
          { ko: '오렌지', en: 'bright orange lipstick', tone: 'warm' },
          { ko: '브라운', en: '90s brown lipstick', tone: 'warm' },
        ],
      },
      {
        key: 'lipTexture', label: '립 표현',
        templates: ['a close-up photo of {}', 'lips with {}'],
        labels: [
          { ko: '글로시', en: 'shiny glossy wet-look lips', tag: '글로시립' },
          { ko: '매트', en: 'matte velvet lipstick', tag: '매트립' },
          { ko: '블러립', en: 'soft-focus blurred lips with a diffused powdery edge' },
          { ko: '그라데이션립', en: 'korean gradient lips, deeper color in the center fading outward' },
          { ko: '오버립', en: 'overlined full lips drawn slightly beyond the natural lip line' },
        ],
      },
      {
        key: 'cheek', label: '치크·윤곽',
        templates: ['a photo of a face with {}', 'makeup featuring {}'],
        labels: [
          { ko: '홍조 블러셔', en: 'rosy flushed pink blush across the cheeks and nose' },
          { ko: '코랄 블러셔', en: 'warm coral peach blush on the cheeks' },
          { ko: '셰이딩', en: 'sculpted contoured cheekbones and jawline' },
          { ko: '하이라이터', en: 'glowing highlighter on the cheekbones' },
          { ko: '미니멀', en: 'almost no blush or contour', tag: '' },
        ],
      },
      {
        key: 'tone', label: '톤', tagSuffix: '메이크업',
        templates: ['a photo of {}', '{}'],
        labels: [
          { ko: '쿨톤', en: 'cool-toned makeup with pink, rose, berry, plum or silver colors' },
          { ko: '웜톤', en: 'warm-toned makeup with coral, peach, orange, gold or brown colors' },
          { ko: '뉴트럴', en: 'neutral-toned makeup with balanced natural colors', tag: '' },
        ],
      },
    ],
  },

  // ───────────────────────────────────────────────────────────── 타투
  tattoo: {
    label: '타투',
    icon: '🖋️',
    detect: [
      'a photo of a tattoo on skin',
      'a close-up photo of tattooed skin',
      'a photo of a person showing their tattoo',
    ],
    groups: [
      {
        key: 'style', label: '장르', tagSuffix: '타투',
        templates: ['a photo of a {} tattoo', 'a {} style tattoo on skin'],
        labels: [
          { ko: '레터링', en: 'lettering script calligraphy text' },
          { ko: '파인라인', en: 'fine line single-needle delicate thin-line' },
          { ko: '미니멀', en: 'tiny minimalist simple symbol' },
          { ko: '블랙워크', en: 'bold solid black ink blackwork' },
          { ko: '블랙앤그레이', en: 'black and grey shaded realistic' },
          { ko: '컬러 리얼리즘', en: 'full color photorealistic' },
          { ko: '올드스쿨', en: 'american traditional old school with bold outlines and limited colors' },
          { ko: '네오트래디셔널', en: 'neo traditional with bold outlines, rich colors and ornate detail' },
          { ko: '뉴스쿨', en: 'new school cartoon-like exaggerated colorful' },
          { ko: '이레즈미', en: 'japanese irezumi with waves, koi, peonies or dragons' },
          { ko: '수채화', en: 'watercolor splash painterly' },
          { ko: '일러스트', en: 'illustrative sketch-like drawing' },
          { ko: '도트워크', en: 'geometric dotwork mandala' },
          { ko: '트라이벌', en: 'tribal polynesian black pattern' },
          { ko: '사이버시길리즘', en: 'cybersigilism with sharp thin spiky flowing black lines like chrome thorns' },
        ],
      },
      {
        key: 'color', label: '컬러', tagSuffix: '타투',
        templates: ['a photo of a {} tattoo', 'a {} tattoo on skin'],
        labels: [
          { ko: '블랙', en: 'solid black ink only line' },
          { ko: '블랙앤그레이', en: 'black and grey shaded monochrome' },
          { ko: '풀컬러', en: 'fully colored vibrant' },
          { ko: '포인트컬러', en: 'mostly black with a small accent of color' },
        ],
      },
      {
        key: 'subject', label: '도안',
        templates: ['a photo of a tattoo of {}', 'a {} tattoo'],
        labels: [
          { ko: '꽃·식물', en: 'flowers, leaves or plants', tag: '꽃타투' },
          { ko: '동물', en: 'an animal such as a lion, wolf, cat or dog', tag: '동물타투' },
          { ko: '나비·새', en: 'a butterfly or a bird', tag: '나비타투' },
          { ko: '문구·이니셜', en: 'words, letters, initials or numbers', tag: '이니셜타투' },
          { ko: '만다라·기하학', en: 'a mandala or geometric shapes and lines', tag: '만다라타투' },
          { ko: '인물', en: 'a human face or portrait', tag: '인물타투' },
          { ko: '달·별·자연', en: 'the moon, stars, sun, waves or mountains', tag: '달타투' },
          { ko: '용·호랑이·잉어', en: 'an asian dragon, tiger or koi fish', tag: '동양화타투' },
          { ko: '해골·다크', en: 'a skull, demon or dark gothic motif', tag: '해골타투' },
          { ko: '캐릭터', en: 'a cartoon or anime character', tag: '캐릭터타투' },
          { ko: '하트·심볼', en: 'a small heart, cross, infinity or symbol', tag: '하트타투' },
          { ko: '사물', en: 'an object such as a clock, key, anchor, rose-knife or ship', tag: '' },
        ],
      },
      {
        key: 'placement', label: '부위', tagSuffix: '타투',
        templates: ['a photo of a tattoo on the {}', 'a {} tattoo'],
        labels: [
          { ko: '손목', en: 'wrist' },
          { ko: '손가락', en: 'hand or fingers' },
          { ko: '팔안쪽', en: 'inner forearm' },
          { ko: '팔뚝', en: 'outer forearm' },
          { ko: '어깨', en: 'upper arm and shoulder' },
          { ko: '쇄골', en: 'collarbone or chest' },
          { ko: '등', en: 'back or shoulder blade' },
          { ko: '옆구리', en: 'ribs or side of the torso' },
          { ko: '허벅지', en: 'thigh or leg' },
          { ko: '발목', en: 'ankle or foot' },
          { ko: '목·귀뒤', en: 'neck or behind the ear', tag: '귀뒤타투' },
        ],
      },
      {
        key: 'size', label: '크기', tagSuffix: '타투',
        templates: ['a photo of {}', '{} on skin'],
        labels: [
          { ko: '미니', en: 'a tiny tattoo about the size of a coin' },
          { ko: '스몰', en: 'a small tattoo about the size of a palm', tag: '' },
          { ko: '미디엄', en: 'a medium-sized tattoo covering part of a limb', tag: '' },
          { ko: '대형', en: 'a large tattoo covering a whole sleeve, back or big body area' },
        ],
      },
    ],
  },
};

// 그룹별 신뢰도 문구 규칙
// 기준은 보정된 확률(heads/*.json 의 calib)로 잰 실제 적중률에서 정했다 (손라벨 폴드 밖 예측 6천여 개)
//   p ≥ 0.75 → 약 91% 맞음 · p ≥ 0.5 → 약 82% 맞음
export const CONFIDENCE = {
  high: 0.75,  // 이 이상이면 단정 ("~입니다")
  mid: 0.5,    // 이 이상이면 "~로 보입니다" (2위가 가까우면 함께 언급)
};

// 톤 이름 (톤 그룹 결과 → 수식어)
export const TONE_WORDS = {
  쿨톤: { adj: '차가운', short: '쿨' },
  웜톤: { adj: '따뜻한', short: '웜' },
  뉴트럴: { adj: '자연스러운', short: '뉴트럴' },
};
