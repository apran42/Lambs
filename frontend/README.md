# Shepherd-AI 프론트엔드

React와 Vite로 구성한 실시간 혼잡도 대시보드입니다.

## Jetson API 연결

`.env.jetson.example`을 `.env.local`로 복사하고 `JETSON_IP`를 실제 Jetson IP로
바꿉니다. Vite는 시작할 때만 환경변수를 읽으므로 값을 바꾼 뒤 개발 서버를 다시
시작해야 합니다.

```bash
cp .env.jetson.example .env.local
npm install
npm run dev -- --host 0.0.0.0
```

Windows에서는 파일 탐색기 또는 다음 명령으로 복사할 수 있습니다.

```powershell
Copy-Item .env.jetson.example .env.local
```

화면 상단의 실행 상태는 다음 세 가지를 구분합니다.

- FastAPI에 연결할 수 없음
- FastAPI는 실행 중이지만 TensorRT Worker 패킷 대기 중
- TensorRT 실시간 스트리밍 중

연결되지 않으면 Jetson에서 다음 주소를 먼저 확인합니다.

```text
http://JETSON_IP:8001/health
http://JETSON_IP:8001/docs
```

## 코드 검사와 빌드

```bash
npm run lint
npm run build
```

- `npm run lint`: ESLint 규칙 검사
- `npm run build`: 배포용 정적 파일 생성
