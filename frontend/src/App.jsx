import React, { useEffect, useState } from 'react';
import { Stage, Layer, Rect, Text } from 'react-konva';

function App() {
  const [data, setData] = useState({ count: 0, boxes: [] });

  useEffect(() => {
    // 백엔드 웹소켓 연결
    const ws = new WebSocket('ws://localhost:8000/ws/stream');
    ws.onmessage = (event) => {
      const received = JSON.parse(event.data);
      setData(received);
    };
    return () => ws.close();
  }, []);

  return (
    <div style={{ padding: '20px' }}>
      <h1>CCTV 실시간 혼잡도: {data.count}명</h1>
      <div style={{ border: '1px solid #ccc', display: 'inline-block' }}>
        {/* Konva Stage: CCTV 영상 위에 겹쳐서 그릴 캔버스 */}
        <Stage width={640} height={480}>
          <Layer>
            {data.boxes.map((box, i) => (
              <Rect
                key={i}
                x={box[0]}
                y={box[1]}
                width={box[2] - box[0]}
                height={box[3] - box[1]}
                stroke="red"
                strokeWidth={2}
              />
            ))}
          </Layer>
        </Stage>
      </div>
    </div>
  );
}

export default App;