import React, { useEffect, useState, useRef } from 'react';
import { Activity, Clock, Camera, Users, AlertTriangle } from 'lucide-react';
import './App.css';

export default function App() {
  const [data, setData] = useState({ count: 0, status: 'Normal' });
  const [connected, setConnected] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());
  
  // 가상 DOM을 거치지 않고 Canvas에 직접 접근하기 위한 Ref
  const canvasRef = useRef(null);
  // 메모리 가비지 컬렉션 부하를 방지하기 위해 단 하나의 이미지 객체만 박제하여 사용
  const imgRef = useRef(null);
  const boxesRef = useRef([]);

  // 우측 '카메라별 혼잡도 분석'에 보여줄 카메라 목록 (고정값)
  const [cameras, setCameras] = useState([
    { id: 'cam-01', name: 'CAM 01', capacity: 80, current: 0 },
  ]);

  // 상단 시계 업데이트
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // 웹소켓 바이너리 연결 및 순수 Canvas 렌더링 로직
  useEffect(() => {
    if (!imgRef.current) {
      imgRef.current = new Image();
    }

    let ws = null;
    try {
      ws = new WebSocket('ws://localhost:8000/ws/stream');
      ws.binaryType = "arraybuffer"; 
      
      ws.onopen = () => setConnected(true);
      ws.onclose = () => setConnected(false);
      
      ws.onmessage = (event) => {
        try {
          const buffer = event.data;
          const view = new DataView(buffer);
          
          // 1. 하이브리드 바이너리 패킷 압축 해제
          const jsonLength = view.getUint32(0, false);
          const jsonBytes = new Uint8Array(buffer, 4, jsonLength);
          const jsonText = new TextDecoder().decode(jsonBytes);
          const metaData = JSON.parse(jsonText);
          
          // 2. React 상태 업데이트 (UI 갱신용)
          setData({ count: metaData.count, status: metaData.status });
          boxesRef.current = metaData.boxes || [];
          
          // CAM 01의 현재 인원수 실시간 업데이트
          setCameras(prev => {
            const newCameras = [...prev];
            newCameras[0].current = metaData.count;
            return newCameras;
          });

          // 3. 이미지 블롭 생성 및 Canvas 그리기
          const imageBytes = new Uint8Array(buffer, 4 + jsonLength);
          const imageBlob = new Blob([imageBytes], { type: 'image/jpeg' });
          const blobUrl = URL.createObjectURL(imageBlob);
          
          imgRef.current.src = blobUrl;
          imgRef.current.onload = () => {
            const canvas = canvasRef.current;
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            
            // 영상 프레임 그리기
            ctx.clearRect(0, 0, 640, 480);
            ctx.drawImage(imgRef.current, 0, 0, 640, 480);
            
            // 바운딩 박스 덧그리기 (GPU 가속)
            ctx.strokeStyle = "#a855f7"; // 테두리 색상 (보라색)
            ctx.lineWidth = 2;
            
            boxesRef.current.forEach((box) => {
              // 백엔드가 [x1, y1, x2, y2] 형태로 보냄
              const w = box[2] - box[0];
              const h = box[3] - box[1];
              ctx.strokeRect(box[0], box[1], w, h);
              
              // 풋포인트
              ctx.fillStyle = "#22c55e"; // 초록색
              ctx.beginPath();
              ctx.arc(box[0] + (w / 2), box[3], 4, 0, 2 * Math.PI);
              ctx.fill();
            });
            
            // 메모리 누수 방지
            URL.revokeObjectURL(blobUrl);
          };
        } catch (e) {
          console.error("데이터 파싱 오류:", e);
        }
      };
    } catch (err) {
      console.warn("웹소켓 연결 오류", err);
    }

    return () => {
      if (ws) {
        try { ws.close(); } catch(e) {}
      }
    };
  }, []);

  // 서버 미연결 시 정적 캔버스 그리기 (랜덤 데이터 제거)
  useEffect(() => {
    if (connected) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    // 배경 그리기
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, 640, 480);
    
    ctx.strokeStyle = '#333';
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(0, 240); ctx.lineTo(640, 240);
    ctx.moveTo(320, 0); ctx.lineTo(320, 480);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // 대기 중 텍스트 표시
    ctx.fillStyle = '#555';
    ctx.font = '20px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('서버 연결 대기 중...', 320, 240);
    
  }, [connected]);

  // CSS 클래스 매핑 함수 (인라인 스타일 대신 클래스 사용)
  const getCongestionLevel = (current, capacity) => {
    const ratio = current / capacity;
    if (ratio < 0.5) return { text: '여유', badgeClass: 'level-green', barClass: 'progress-green' }; 
    if (ratio < 0.8) return { text: '보통', badgeClass: 'level-yellow', barClass: 'progress-yellow' }; 
    return { text: '혼잡', badgeClass: 'level-red', barClass: 'progress-red' }; 
  };

  return (
    <div className="app-container">
        
        {/* 1. 상단 헤더 */}
        <header className="header">
          <div className="header-title-wrapper">
            <div className="header-icon">
              <Activity size={24} color="white" />
            </div>
            <div>
              <h1 className="header-title">지능형 영상 기반 혼잡도 모니터링</h1>
              <p className="header-subtitle">Edge AI & Binary Hybrid Streaming</p>
            </div>
          </div>
          
          <div className="header-status-wrapper">
            <div className="clock">
              <Clock size={16} />
              {currentTime.toLocaleTimeString('ko-KR')}
            </div>
            {connected ? (
              <div className="status-badge status-connected">
                <div className="status-dot-connected"></div>
                <span className="status-text-connected">서버 연결됨</span>
              </div>
            ) : (
               <div className="status-badge status-disconnected">
                 <div className="status-dot-disconnected"></div>
                 <span className="status-text-disconnected">시뮬레이션 모드 (연결 대기중)</span>
               </div>
            )}
          </div>
        </header>

        {/* 2. 메인 콘텐츠 영역 */}
        <div className="main-content">
          
          {/* 좌측 패널 (CCTV 영상) */}
          <div className="left-panel">
            <div className="card">
              
              {/* 영상 상단 타이틀 및 상태 */}
              <div className="cctv-header">
                <h2 className="cctv-title">
                  <Camera size={20} color="#c084fc" />
                  CAM 01 실시간 모니터링
                </h2>
                
                <div className="cctv-stats">
                  <span className="cctv-count">
                    탐지 인원: <strong>{data.count}명</strong>
                  </span>
                  <span className={`cctv-status ${data.count > 30 ? 'cctv-status-danger' : 'cctv-status-normal'}`}>
                    {data.status}
                  </span>
                </div>
              </div>

              {/* 순정 Canvas 영상 송출 영역 */}
              <div className="canvas-wrapper">
                <canvas 
                  ref={canvasRef} 
                  width={640} 
                  height={480} 
                  className="canvas-element"
                />
              </div>
            </div>
          </div>

          {/* 우측 패널 (통계 및 카메라 리스트) */}
          <div className="right-panel">
            
            {/* 전체 통계 요약 카드 */}
            <div className="card card-padding-lg">
              <h2 className="summary-title">실시간 전체 탐지 인원 (CAM 01)</h2>
              <div className="summary-value-wrapper">
                <span className="summary-value">{data.count}</span>
                <span className="summary-unit">명</span>
              </div>
              {data.count > 30 && (
                 <div className="alert-box">
                   <AlertTriangle size={20} color="#f87171" className="flex-shrink-0" />
                   <p className="alert-text">주의: 메인 스테이지 밀집도 상승 감지됨</p>
                 </div>
              )}
            </div>

            {/* 카메라별 분석 리스트 */}
            <div className="card card-padding-lg card-flex-1">
              <h2 className="list-title">
                <Users size={20} color="#9ca3af" />
                카메라별 혼잡도 분석
              </h2>
              <div className="camera-list">
                {cameras.map((camera) => {
                  const level = getCongestionLevel(camera.current, camera.capacity);
                  const percent = Math.min(100, Math.round((camera.current / camera.capacity) * 100));
                  
                  return (
                    <div key={camera.id} className="camera-item">
                      <div className="camera-item-header">
                        <span className="camera-name">{camera.name}</span>
                        <div className="camera-stats">
                          <span className="camera-count">{camera.current} / {camera.capacity}</span>
                          <span className={`camera-badge ${level.badgeClass}`}>
                            {level.text}
                          </span>
                        </div>
                      </div>
                      {/* 진행률 게이지 바 */}
                      <div className="progress-bg">
                        <div 
                          className={`progress-bar ${level.barClass}`}
                          style={{ width: `${percent}%` }}
                        ></div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

          </div>
        </div>
      </div>
  );
}