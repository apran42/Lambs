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

## 구역별 밀집도 경고 로그

로그는 CAM 번호별로 묶고 각 행의 첫 열에 고정 Bird-eye 구역 번호(예: #5)를 표시합니다.
각 셀이 주의/위험에 진입할 때 기록을 만들고, 위험으로 상승하거나
최대 인원·밀도가 높아지면 같은 기록을 갱신합니다. 해당 셀이 여유 상태로
돌아왔을 때만 정상 복귀로 종료합니다. 구역 설정에서 셀이 사라진 경우에는
정상 복귀가 아닌 설정 변경으로 표시합니다.

현재 로그는 브라우저 메모리에만 있으며 새로고침하면 초기화됩니다. 영상과
검출·밀도 스트림은 기존 API를 그대로 사용하고, 로그 영구 보관은 아직 구현하지
않았습니다.

구역별 전이 테스트는 `node --test src/densityLog.test.js`로 실행합니다.

CCTV 영상을 클릭하면 해당 카메라의 확대 팝업이 열립니다. 닫기 버튼,
바깥 영역 클릭 또는 Escape 키로 닫습니다.
