import { useEffect, useState, useRef } from 'react';

function App() {
  const [data, setData] = useState({ count: 0, status: 'Normal' });
  const [connected, setConnected] = useState(false);
  
  // 가상 DOM을 거치지 않고 Canvas에 직접 접근하기 위한 Ref
  const canvasRef = useRef(null);
  // 메모리 가비지 컬렉션 부하를 방지하기 위해 단 하나의 이미지 객체만 박제하여 사용
  const imgRef = useRef(null);
  const boxesRef = useRef([]);

  useEffect(() => {
    // 순수 이미지 객체 초기화
    if (!imgRef.current) {
      imgRef.current = new Image();
    }

    const ws = new WebSocket('ws://localhost:8000/ws/stream');
    ws.binaryType = "arraybuffer"; 
    
    ws.onopen = () => setConnected(true);
    
    ws.onmessage = (event) => {
      try {
        const buffer = event.data;
        const view = new DataView(buffer);
        
        // 1. 하이브리드 바이너리 패킷 압축 해제
        const jsonLength = view.getUint32(0, false);
        const jsonBytes = new Uint8Array(buffer, 4, jsonLength);
        const jsonText = new TextDecoder().decode(jsonBytes);
        const metaData = JSON.parse(jsonText);
        
        // 상단 오버레이용 단순 상태 업데이트 (리렌더링 최소화)
        setData({ count: metaData.count, status: metaData.status });
        // 박스 좌표는 리액트 상태가 아닌 Ref에 저장하여 리렌더링 부하 유발 차단
        boxesRef.current = metaData.boxes;

        // 2. 순수 바이트 데이터 추출 후 Blob 변환
        const imageOffset = 4 + jsonLength;
        const imageBytes = new Uint8Array(buffer, imageOffset);
        const blob = new Blob([imageBytes], { type: 'image/jpeg' });

        // 3. Object URL을 만들고 로드되면 곧바로 캔버스에 드로잉 (메인 핵심)
        const currentUrl = URL.createObjectURL(blob);
        
        imgRef.current.onload = () => {
          const canvas = canvasRef.current;
          if (!canvas) return;
          const ctx = canvas.getContext('2d');
          
          // [GPU 가속] 이미지를 이전 프레임 위에 다이렉트로 덮어씌움 (깜빡임 완벽 제로)
          ctx.drawImage(imgRef.current, 0, 0, canvas.width, canvas.height);
          
          // [실시간 바운딩 박스 드로잉] 이미지 위에 바로 사각형 그리기
          ctx.lineWidth = 2;
          ctx.strokeStyle = metaData.count > 10 ? "#f44336" : "#4CAF50";
          
          boxesRef.current.forEach(([x1, y1, x2, y2]) => {
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          });

          // 사용한 메모리 주소는 즉시 폐기 (메모리 누수 및 밀림 차단)
          URL.revokeObjectURL(currentUrl);
        };

        imgRef.current.src = currentUrl;

      } catch (error) {
        console.error("바이너리 파싱 에러: ", error);
      }
    };
    
    ws.onclose = () => setConnected(false);

    return () => {
      ws.close();
    };
  }, []);

  return (
    <div style={{ 
      padding: '20px', 
      backgroundColor: '#1a1a1a', 
      color: 'white', 
      minHeight: '100vh',
      fontFamily: 'Arial, sans-serif'
    }}>
      <header style={{ borderBottom: '2px solid #4CAF50', marginBottom: '20px', paddingBottom: '10px' }}>
        <h1 style={{ margin: 0 }}>Shepherd-AI: 실시간 군중 관제 시스템</h1>
        <span style={{ color: connected ? '#4CAF50' : '#f44336' }}>
          ● {connected ? 'Connected' : 'Disconnected'}
        </span>
      </header>

      <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
        {/* 메인 스트리밍 영역 (레이아웃 및 디자인은 완벽히 보존) */}
        <div style={{ flex: '1 1 640px' }}>
          <div style={{ 
            position: 'relative', 
            border: '2px solid #333', 
            borderRadius: '8px', 
            overflow: 'hidden',
            width: '640px',
            height: '480px'
          }}>
            
            {/* 💡 무거운 react-konva 대신 초고속 HTML5 순정 Canvas를 배치합니다. */}
            <canvas 
              ref={canvasRef} 
              width={640} 
              height={480} 
              style={{ backgroundColor: '#000', display: 'block' }}
            />

            {/* 상태 오버레이 (기존 스타일 100% 동일) */}
            <div style={{
              position: 'absolute',
              top: '10px',
              left: '10px',
              backgroundColor: 'rgba(0,0,0,0.7)',
              padding: '5px 10px',
              borderRadius: '4px',
              fontSize: '18px',
              fontWeight: 'bold',
              color: data.count > 10 ? '#f44336' : '#4CAF50'
            }}>
              Count: {data.count} | Status: {data.status}
            </div>
          </div>
        </div>

        {/* 대시보드 및 통계 영역 */}
        <div style={{ flex: '1 1 400px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div style={{ 
            backgroundColor: '#2d2d2d', 
            padding: '15px', 
            borderRadius: '8px',
            border: '1px solid #444'
          }}>
            <h3 style={{ margin: '0 0 10px 0', color: '#4CAF50' }}>🚀 고성능 GPU 가속 모드 가동</h3>
            <p style={{ fontSize: '14px', color: '#aaa', margin: 0 }}>
              가상 DOM 연산을 배제하고 HTML5 Canvas에 데이터 레이어를 다이렉트 픽셀 맵핑하여 끊김과 지연 현상을 원천 제거했습니다.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;