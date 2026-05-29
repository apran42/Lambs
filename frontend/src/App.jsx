import { useEffect, useState, useRef } from 'react';
import { Stage, Layer, Rect, Image as KonvaImage } from 'react-konva';

// =================================================================
// [최적화] 순수 바이너리(Blob) 주소매핑을 지원하는 영상 컴포넌트
// =================================================================
const VideoBackground = ({ imageBlobUrl }) => {
  const [renderedImage, setRenderedImage] = useState(null);
  const imgRef = useRef(null);

  useEffect(() => {
    if (!imageBlobUrl) return;

    if (!imgRef.current) {
      imgRef.current = new Image();
    }

    imgRef.current.onload = () => {
      setRenderedImage(imgRef.current);
    };

    imgRef.current.src = imageBlobUrl;
  }, [imageBlobUrl]);

  return <KonvaImage image={renderedImage} width={640} height={480} />;
};

function App() {
  // [수정] 누락되었던 AI 추론 데이터 상태 정의 추가
  const [data, setData] = useState({ count: 0, boxes: [], status: 'Normal' });
  const [blobUrl, setBlobUrl] = useState(null);
  const [connected, setConnected] = useState(false);
  const prevBlobUrlRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/stream');
    ws.binaryType = "arraybuffer"; // 바이너리 통신 수신 정의
    
    ws.onopen = () => setConnected(true);
    
    ws.onmessage = async (event) => {
      if (ws.bufferedAmount > 2 * 1024 * 1024) return; 

      try {
        // 백엔드가 전송한 전체 ArrayBuffer 가져오기
        const buffer = event.data;
        const view = new DataView(buffer);
        
        // 1. 처음 4바이트에서 JSON 텍스트의 길이를 정수로 읽어옴
        const jsonLength = view.getUint32(0, false);
        
        // 2. 그 길이만큼 바이트를 잘라내어 텍스트(JSON)로 디코딩
        const jsonBytes = new Uint8Array(buffer, 4, jsonLength);
        const jsonText = new TextDecoder().decode(jsonBytes);
        const metaData = JSON.parse(jsonText);
        
        // AI 추론 데이터 업데이트 (count, boxes, status)
        setData(metaData);

        // 3. 나머지 바이트는 순수 이미지 JPEG 데이터이므로 Blob으로 가공
        const imageOffset = 4 + jsonLength;
        const imageBytes = new Uint8Array(buffer, imageOffset);
        const blob = new Blob([imageBytes], { type: 'image/jpeg' });

        // 고속 메모리 맵핑 주소 생성
        const currentUrl = URL.createObjectURL(blob);
        setBlobUrl(currentUrl);

        // 메모리 누수 방지 (이전 프레임 즉시 해제)
        if (prevBlobUrlRef.current) {
          URL.revokeObjectURL(prevBlobUrlRef.current);
        }
        prevBlobUrlRef.current = currentUrl;

      } catch (error) {
        console.error("하이브리드 바이너리 파싱 에러: ", error);
      }
    };
    
    ws.onclose = () => setConnected(false);

    return () => {
      ws.close();
      if (prevBlobUrlRef.current) URL.revokeObjectURL(prevBlobUrlRef.current);
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
        {/* 메인 스트리밍 영역 */}
        <div style={{ flex: '1 1 640px' }}>
          <div style={{ 
            position: 'relative', 
            border: '2px solid #333', 
            borderRadius: '8px', 
            overflow: 'hidden',
            width: '640px',
            height: '480px'
          }}>
            <Stage width={640} height={480}>
              <Layer>
                {/* 배경 영상 (바이너리 Blob 적용) */}
                <VideoBackground imageBlobUrl={blobUrl} />
                
                {/* Bounding Boxes (정상 렌더링 가능) */}
                {data.boxes.map((box, i) => (
                  <Rect
                    key={i}
                    x={box[0]}
                    y={box[1]}
                    width={box[2] - box[0]}
                    height={box[3] - box[1]}
                    stroke={data.count > 10 ? "#f44336" : "#4CAF50"}
                    strokeWidth={2}
                  />
                ))}
              </Layer>
            </Stage>

            {/* 상태 오버레이 */}
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
            <h3 style={{ margin: '0 0 10px 0', color: '#4CAF50' }}>⚙️ 실시간 고속 스트리밍 모드</h3>
            <p style={{ fontSize: '14px', color: '#aaa', margin: 0 }}>
              하이브리드 바이너리 패킹 데이터 전송 기법이 적용되어 인원수 데이터와 고해상도 프레임이 밀림(0.75배속 현상) 없이 동기화됩니다.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;