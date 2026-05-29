import React, { useEffect, useRef } from 'react';

function CCTVViewer() {
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const prevBlobUrlRef = useRef(null);

  useEffect(() => {
    // 백엔드 웹소켓 연결
    wsRef.current = new WebSocket('ws://localhost:8000/ws/stream');
    wsRef.current.binaryType = "arraybuffer"; // 바이너리 데이터 수신 설정

    wsRef.current.onmessage = (event) => {
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

        // 3. 나머지 바이트는 순수 이미지 JPEG 데이터이므로 Blob으로 가공
        const imageOffset = 4 + jsonLength;
        const imageBytes = new Uint8Array(buffer, imageOffset);
        const blob = new Blob([imageBytes], { type: 'image/jpeg' });

        // 고속 메모리 맵핑 주소 생성
        const currentUrl = URL.createObjectURL(blob);
        
        // 4. 이미지가 메모리에 완전히 로드되었을 때만 캔버스에 그리기
        const img = new Image();
        img.onload = () => {
          const canvas = canvasRef.current;
          if (!canvas) return;
          
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          
          // 5. 바인딩된 Bounding Box가 있다면 이 위에 바로 그리기
          if (metaData.boxes) {
            ctx.strokeStyle = '#00FF00'; // 초록색 박스
            ctx.lineWidth = 2;
            metaData.boxes.forEach(([x1, y1, x2, y2]) => {
              ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            });
          }
        };
        img.src = currentUrl;

        // 메모리 누수 방지 (이전 프레임 즉시 해제)
        if (prevBlobUrlRef.current) {
          URL.revokeObjectURL(prevBlobUrlRef.current);
        }
        prevBlobUrlRef.current = currentUrl;

      } catch (error) {
        console.error("바이너리 파싱 에러: ", error);
      }
    };

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (prevBlobUrlRef.current) URL.revokeObjectURL(prevBlobUrlRef.current);
    };
  }, []);

  return (
    <div className="video-container">
      <h3>실시간 AI 관제 화면</h3>
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