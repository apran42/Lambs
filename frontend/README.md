# Lambs 프론트엔드

React와 Vite로 구성한 실시간 혼잡도 대시보드입니다.

## 개발 서버 실행

Node.js와 npm을 설치한 뒤 `frontend` 폴더에서 실행합니다.

```bash
npm install
npm run dev
```

터미널에 표시된 주소(기본값 `http://localhost:5173`)를 브라우저에서 엽니다.
백엔드 주소를 환경변수로 변경했다면 Vite 개발 서버를 다시 시작해야 합니다.

## 코드 검사와 빌드

```bash
npm run lint
npm run build
```

- `npm run lint`: ESLint 규칙 검사
- `npm run build`: 배포용 정적 파일 생성

Vite와 React 설정을 변경할 때는 각 도구의 공식 문서를 참고하세요.
