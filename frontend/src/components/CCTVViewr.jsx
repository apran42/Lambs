import React, { useEffect, useRef } from 'react';

function CCTVViewer() {
  const canvasRef = useRef(null);
  const wsRef = useRef(null);

  useEffect(() => {
    // 백엔드 웹소켓 연결
    wsRef.current = new WebSocket('ws://localhost:8000/ws/stream');

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      // 1. 메모리 상에 임시 이미지 객체 생성
      const img = new Image();
      img.src = `data:image/jpeg;base64,${data.image}`;
      
      // 2. 이미지가 메모리에 완전히 로드되었을 때만 캔버스에 그리기 (플리커링 제거 핵심!)
      img.onload = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        // 기존 화면을 지우지 않고 새로운 프레임으로 바로 덮어씁니다.
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        
        // 3. 바인딩된 Bounding Box가 있다면 이 위에 바로 그리기
        if (data.boxes) {
          ctx.strokeStyle = '#00FF00'; // 초록색 박스
          ctx.lineWidth = 2;
          data.boxes.forEach(([x1, y1, x2, y2]) => {
            // 백엔드에서 보내준 좌표 스케일에 맞춰 드로잉
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          });
        }
      };
    };

    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  return (
    <div className="video-container">
      <h3>실시간 AI 관제 화면</h3>
      {/* <img> 대신 <canvas>를 배치합니다 */}
      <canvas 
        ref={canvasRef} 
        width={640} 
        height={480} 
        style={{ border: '1px solid #ccc', backgroundColor: '#000' }}
      />
    </div>
  );
}

export default CCTVViewer;