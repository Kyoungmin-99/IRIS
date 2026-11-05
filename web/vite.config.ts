import { defineConfig } from 'vite';

export default defineConfig({
  // 상대 경로로 빌드해야 GitHub Pages 하위 경로(/IRIS/)에서도 그대로 동작한다.
  base: './',
  // 없는 파일에 index.html 을 돌려주지 않게(SPA 대체 끔): 미리보기 서버가 손톱 모델(SlimSAM) 설정 파일 자리에
  // index.html 을 200 으로 돌려줘 Transformers.js 가 로컬 모델로 착각, 사진 모드 네일이 '손톱을 찾는 중'에서 멈췄다
  // (GitHub Pages 는 404 라 Hugging Face 에서 받아 괜찮았다)
  appType: 'mpa',
  build: {
    target: 'es2022',
    // 두 페이지: 앱(index.html), 손톱 라벨 도구(label.html)
    rollupOptions: { input: { main: 'index.html', label: 'label.html' } },
  },
});
