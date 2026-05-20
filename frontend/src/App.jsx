import { useEffect, useState } from 'react';
import { Stage, Layer, Rect, Image as KonvaImage } from 'react-konva';
import useImage from 'use-image';

// 실시간 영상을 렌더링하기 위한 커스텀 컴포넌트
const VideoBackground = ({ imageBase64 }) => {
  const [image] = useImage(imageBase64 ? `data:image/jpeg;base64,${imageBase64}` : null);
  return <KonvaImage image={image} width={640} height={480} />;
};

function App() {
  const [data, setData] = useState({ count: 0, boxes: [], image: null, status: 'Normal' });
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/stream');
    
    ws.onopen = () => setConnected(true);
    ws.onmessage = (event) => {
      const received = JSON.parse(event.data);
      setData(received);
    };
    ws.onclose = () => setConnected(false);

    return () => ws.close();
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
                {/* 배경 영상 */}
                <VideoBackground imageBase64={data.image} />
                
                {/* Bounding Boxes */}
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
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;