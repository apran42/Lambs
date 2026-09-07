import { useCallback, useEffect, useRef, useState } from 'react';
import './App.css';

const WS_BASE = import.meta.env.VITE_WS_BASE_URL
  || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.hostname}:8000/ws/stream`;
const API_BASE = import.meta.env.VITE_API_BASE_URL
  || `${window.location.protocol}//${window.location.hostname}:8000`;

const DEFAULT_CAMERAS = [
  { id: 'cam-01', name: 'CAM 01' },
  { id: 'cam-02', name: 'CAM 02' },
];

const LEVEL_TEXT = { Relaxed: '여유', Caution: '주의', Danger: '위험' };
const LEVEL_BADGE = { Relaxed: 'level-green', Caution: 'level-yellow', Danger: 'level-red' };
const LEVEL_STATUS = {
  Relaxed: 'cctv-status-normal',
  Caution: 'cctv-status-caution',
  Danger: 'cctv-status-danger',
};

function CameraFeed({ camera, onMetrics }) {
  const canvasRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [metadata, setMetadata] = useState(null);

  useEffect(() => {
    let socket;
    let reconnectTimer;
    let disposed = false;
    let newestMessage = 0;

    const drawPacket = async (buffer) => {
      if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 4) {
        throw new Error('Invalid stream packet');
      }
      const view = new DataView(buffer);
      const jsonLength = view.getUint32(0, false);
      if (jsonLength > buffer.byteLength - 4) {
        throw new Error('Invalid metadata length');
      }

      const jsonBytes = new Uint8Array(buffer, 4, jsonLength);
      const nextMetadata = JSON.parse(new TextDecoder().decode(jsonBytes));
      const messageId = ++newestMessage;
      setMetadata(nextMetadata);
      onMetrics(camera.id, { ...nextMetadata, connected: true });

      const imageBytes = new Uint8Array(buffer, 4 + jsonLength);
      const bitmap = await createImageBitmap(new Blob([imageBytes], { type: 'image/jpeg' }));
      if (disposed || messageId !== newestMessage) {
        bitmap.close();
        return;
      }

      const canvas = canvasRef.current;
      if (!canvas) {
        bitmap.close();
        return;
      }
      const context = canvas.getContext('2d');
      const sourceWidth = Number(nextMetadata.width) || bitmap.width;
      const sourceHeight = Number(nextMetadata.height) || bitmap.height;
      const scaleX = canvas.width / sourceWidth;
      const scaleY = canvas.height / sourceHeight;
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      bitmap.close();

      const drawPolygon = (points, strokeStyle, lineWidth = 1) => {
        if (!Array.isArray(points) || points.length < 3) return;
        context.beginPath();
        context.moveTo(points[0][0] * scaleX, points[0][1] * scaleY);
        points.slice(1).forEach(([x, y]) => context.lineTo(x * scaleX, y * scaleY));
        context.closePath();
        context.strokeStyle = strokeStyle;
        context.lineWidth = lineWidth;
        context.stroke();
      };

      nextMetadata.grid?.forEach((cell) => {
        drawPolygon(cell.polygon, 'rgba(255, 255, 255, 0.45)');
        const centerX = cell.polygon.reduce((sum, point) => sum + point[0], 0) / cell.polygon.length;
        const centerY = cell.polygon.reduce((sum, point) => sum + point[1], 0) / cell.polygon.length;
        context.fillStyle = 'rgba(255, 255, 255, 0.9)';
        context.font = '12px sans-serif';
        context.textAlign = 'center';
        context.fillText(`#${cell.id} ${cell.count}명`, centerX * scaleX, centerY * scaleY);
      });
      drawPolygon(nextMetadata.roi_points, '#22d3ee', 2);
      if (nextMetadata.local_peak_enabled) {
        drawPolygon(nextMetadata.local_peak?.polygon, '#fb923c', 3);
      }

      nextMetadata.detections?.forEach((detection) => {
        if (!Array.isArray(detection.box) || detection.box.length !== 4) return;
        const [x1, y1, x2, y2] = detection.box;
        const left = x1 * scaleX;
        const top = y1 * scaleY;
        const width = (x2 - x1) * scaleX;
        const height = (y2 - y1) * scaleY;
        context.strokeStyle = detection.in_roi ? '#a855f7' : '#6b7280';
        context.fillStyle = detection.in_roi ? '#22c55e' : '#6b7280';
        context.lineWidth = 2;
        context.strokeRect(left, top, width, height);
        context.beginPath();
        context.arc(left + width / 2, top + height, 4, 0, 2 * Math.PI);
        context.fill();
      });
    };

    const connect = () => {
      socket = new WebSocket(`${WS_BASE}/${camera.id}`);
      socket.binaryType = 'arraybuffer';
      socket.onopen = () => setConnected(true);
      socket.onmessage = (event) => {
        drawPacket(event.data).catch((error) => console.error(`${camera.id} packet error:`, error));
      };
      socket.onerror = () => socket.close();
      socket.onclose = () => {
        setConnected(false);
        onMetrics(camera.id, { connected: false });
        if (!disposed) reconnectTimer = window.setTimeout(connect, 1500);
      };
    };

    connect();
    return () => {
      disposed = true;
      newestMessage += 1;
      window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [camera.id, onMetrics]);

  useEffect(() => {
    if (connected) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext('2d');
    context.fillStyle = '#111';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = '#777';
    context.font = '20px sans-serif';
    context.textAlign = 'center';
    context.fillText('서버 연결 대기 중...', canvas.width / 2, canvas.height / 2);
  }, [connected]);

  const level = metadata?.risk_level || 'Unavailable';
  return (
    <div className="card camera-feed-card">
      <div className="cctv-header">
        <h2 className="cctv-title">{camera.name}</h2>
        <div className="cctv-stats">
          <span className="cctv-count">ROI <strong>{metadata?.roi_count || 0}명</strong></span>
          <span className={`cctv-status ${LEVEL_STATUS[level] || 'cctv-status-normal'}`}>
            {LEVEL_TEXT[level] || '분석 대기'}
          </span>
        </div>
      </div>
      <div className="canvas-wrapper">
        <canvas ref={canvasRef} width={640} height={480} className="canvas-element" />
      </div>
      <div className="feed-metrics">
        <span>영상 {Number(metadata?.capture_fps || 0).toFixed(1)} FPS</span>
        <span>AI {Number(metadata?.analysis_fps || 0).toFixed(1)} FPS</span>
        <span>고정 그리드 최대 {Number(metadata?.max_grid_density_people_per_m2 || 0).toFixed(2)} 명/㎡</span>
        <span>
          5분 예측 {metadata?.forecast_5m?.predicted_roi_count ?? '-'}명
          {metadata?.forecast_5m && ` (${metadata.forecast_5m.confidence_label})`}
        </span>
        <span>분석 지연 {metadata?.analysis_lag_frames ?? '-'} frames</span>
      </div>
    </div>
  );
}

export default function App() {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [cameraMetrics, setCameraMetrics] = useState({});
  const [cameras, setCameras] = useState(DEFAULT_CAMERAS);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    fetch(`${API_BASE}/api/cameras`)
      .then((response) => {
        if (!response.ok) throw new Error(`Camera API returned ${response.status}`);
        return response.json();
      })
      .then((payload) => {
        if (Array.isArray(payload.cameras) && payload.cameras.length) {
          setCameras(payload.cameras.map((camera) => ({
            id: camera.camera_id,
            name: camera.name || camera.camera_id,
          })));
        }
      })
      .catch((error) => console.warn('카메라 목록을 불러오지 못했습니다:', error));
  }, []);

  const updateMetrics = useCallback((cameraId, next) => {
    setCameraMetrics((previous) => ({
      ...previous,
      [cameraId]: { ...previous[cameraId], ...next },
    }));
  }, []);

  const connectedCount = cameras.filter((camera) => cameraMetrics[camera.id]?.connected).length;

  return (
    <div className="app-container">
      <header className="header">
        <div className="header-title-wrapper">
          <div className="header-icon">AI</div>
          <div>
            <h1 className="header-title">다중 영상 혼잡도 모니터링</h1>
            <p className="header-subtitle">원본 시점 스트리밍 · Bird-eye 좌표 기반 ㎡ 밀도</p>
          </div>
        </div>
        <div className="header-status-wrapper">
          <div className="clock">{currentTime.toLocaleTimeString('ko-KR')}</div>
          <div className={`status-badge ${connectedCount ? 'status-connected' : 'status-disconnected'}`}>
            <div className={connectedCount ? 'status-dot-connected' : 'status-dot-disconnected'} />
            <span className={connectedCount ? 'status-text-connected' : 'status-text-disconnected'}>
              {connectedCount}/{cameras.length} 영상 연결
            </span>
          </div>
        </div>
      </header>

      <div className="multi-camera-layout">
        <div className="camera-grid">
          {cameras.map((camera) => (
            <CameraFeed key={camera.id} camera={camera} onMetrics={updateMetrics} />
          ))}
        </div>

        <div className="right-panel">
          <div className="card card-padding-lg">
            <h2 className="list-title">카메라별 Bird-eye 밀도</h2>
            <div className="camera-list">
              {cameras.map((camera) => {
                const metrics = cameraMetrics[camera.id] || {};
                const level = metrics.risk_level || 'Unavailable';
                const density = Number(metrics.applied_peak_density_people_per_m2 || 0);
                const forecast = metrics.forecast_5m;
                const dangerThreshold = Number(metrics.thresholds?.danger_min || 5);
                return (
                  <div key={camera.id} className="camera-item">
                    <div className="camera-item-header">
                      <span className="camera-name">{camera.name}</span>
                      <div className="camera-stats">
                        <span className="camera-count">{density.toFixed(1)} 명/㎡</span>
                        <span className={`camera-badge ${LEVEL_BADGE[level] || 'level-green'}`}>
                          {LEVEL_TEXT[level] || '대기'}
                        </span>
                      </div>
                    </div>
                    <div className="forecast-line">
                      5분 뒤 ROI {forecast?.predicted_roi_count ?? '-'}명 · {' '}
                      {forecast?.ready ? `신뢰도 ${Math.round(forecast.confidence * 100)}%` : '추세 학습 중'}
                    </div>
                    <div className="progress-bg">
                      <div
                        className={`progress-bar ${level === 'Danger' ? 'progress-red' : level === 'Caution' ? 'progress-yellow' : 'progress-green'}`}
                        style={{ width: `${Math.min(100, density / dangerThreshold * 100)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
          <div className="card card-padding-lg">
            <h2 className="summary-title">면적 정확도 조건</h2>
            <p className="calibration-note">
              각 카메라의 청록색 ROI 네 점과 촬영 평면의 실측 가로·세로를 입력해야
              Bird-eye 공간의 1×1 영역을 실제 1㎡로 해석할 수 있습니다.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
