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

function forecastFor(metadata) {
  return metadata?.forecast || metadata?.forecast_5m || null;
}

function forecastHorizonLabel(forecast) {
  const seconds = Number(forecast?.horizon_seconds || 60);
  return seconds % 60 === 0 ? `${seconds / 60}분` : `${seconds}초`;
}

function forecastMethodLabel(forecast) {
  if (!forecast) return '예측 대기 중';
  if (forecast.method === 'persistence-baseline') return '현재 인원 유지 가정';
  if (forecast.method === 'damped-linear-trend') return '추세식(실험)';
  return '추세 준비 중';
}

function CameraFeed({ camera, onMetrics, onDensityLog, workerCamera, inferenceAvailable }) {
  const canvasRef = useRef(null);
  const previousLevelRef = useRef('Unavailable');
  const activeLogIdRef = useRef(null);
  const alertTimerRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [metadata, setMetadata] = useState(null);
  const [renderFps, setRenderFps] = useState(0);
  const [showDensityAlert, setShowDensityAlert] = useState(false);
  const [visibleAlertLevel, setVisibleAlertLevel] = useState(null);
  const level = metadata?.risk_level || 'Unavailable';
  const roiCount = Number(metadata?.roi_count || 0);
  const density = Number(metadata?.applied_peak_density_people_per_m2 || 0);

  useEffect(() => {
    const previousLevel = previousLevelRef.current;
    const isAlertLevel = level === 'Caution' || level === 'Danger';
    const wasAlertLevel = previousLevel === 'Caution' || previousLevel === 'Danger';

    if (isAlertLevel) {
      if (level !== previousLevel) {
        window.clearTimeout(alertTimerRef.current);
        setShowDensityAlert(true);
        setVisibleAlertLevel(level);
        alertTimerRef.current = window.setTimeout(() => setShowDensityAlert(false), 5000);
      }
      if (!activeLogIdRef.current) {
        const id = `${camera.id}-${Date.now()}`;
        activeLogIdRef.current = id;
        onDensityLog('detected', {
          id,
          cameraName: camera.name,
          level,
          density,
          detectedAt: new Date().toLocaleTimeString('ko-KR'),
          clearedAt: '',
        });
      } else if (level === 'Danger' && previousLevel !== 'Danger') {
        onDensityLog('updated', { id: activeLogIdRef.current, level, density });
      }
    } else if (wasAlertLevel && activeLogIdRef.current) {
      const id = activeLogIdRef.current;
      activeLogIdRef.current = null;
      onDensityLog('cleared', { id, clearedAt: new Date().toLocaleTimeString('ko-KR') });
    }

    previousLevelRef.current = level;
  }, [camera.id, camera.name, density, level, onDensityLog]);

  useEffect(() => () => window.clearTimeout(alertTimerRef.current), []);

  useEffect(() => {
    let socket;
    let reconnectTimer;
    let disposed = false;
    let pendingPacket = null;
    let rendering = false;
    const renderTimes = [];

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
      setMetadata(nextMetadata);
      onMetrics(camera.id, { ...nextMetadata, connected: true });

      const imageBytes = new Uint8Array(buffer, 4 + jsonLength);
      const bitmap = await createImageBitmap(new Blob([imageBytes], { type: 'image/jpeg' }));
      if (disposed) {
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

      const renderedAt = performance.now();
      renderTimes.push(renderedAt);
      while (renderTimes.length > 1 && renderTimes[0] < renderedAt - 2000) {
        renderTimes.shift();
      }
      const renderSpan = renderTimes.at(-1) - renderTimes[0];
      const clientRenderFps = renderSpan > 0
        ? (renderTimes.length - 1) * 1000 / renderSpan
        : 0;
      setRenderFps(clientRenderFps);
      onMetrics(camera.id, { client_render_fps: clientRenderFps });
    };

    const renderLatestPacket = async () => {
      if (rendering) return;
      rendering = true;
      try {
        while (!disposed && pendingPacket) {
          const packet = pendingPacket;
          pendingPacket = null;
          await drawPacket(packet);
        }
      } catch (error) {
        console.error(`${camera.id} packet error:`, error);
      } finally {
        rendering = false;
        if (!disposed && pendingPacket) renderLatestPacket();
      }
    };

    const connect = () => {
      socket = new WebSocket(`${WS_BASE}/${camera.id}`);
      socket.binaryType = 'arraybuffer';
      socket.onopen = () => setConnected(true);
      socket.onmessage = (event) => {
        pendingPacket = event.data;
        renderLatestPacket();
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
      pendingPacket = null;
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

  const forecast = forecastFor(metadata);
  const waitingMessage = workerCamera?.last_error
    || (!inferenceAvailable ? 'TensorRT Worker 패킷 대기 중...' : '영상 스트림 연결 대기 중...');
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
        {!connected && <div className="stream-waiting-message">{waitingMessage}</div>}
      </div>
      {showDensityAlert && visibleAlertLevel && (
        <div className={`density-alert ${visibleAlertLevel === 'Danger' ? 'density-alert-danger' : 'density-alert-caution'}`} role="alert" aria-live="assertive">
          <span className="density-alert-icon" aria-hidden="true">!</span>
          <span>
            {visibleAlertLevel === 'Danger' ? '밀집 위험 경고' : '밀집도 주의'}
            {' · '}{camera.name}에 {roiCount}명이 감지되었습니다. 현장을 확인해 주세요.
          </span>
        </div>
      )}
      <div className="feed-metrics">
        <span>영상 {Number(metadata?.capture_fps || 0).toFixed(1)} FPS</span>
        <span>AI {Number(metadata?.analysis_fps || 0).toFixed(1)} FPS</span>
        <span>브라우저 {renderFps.toFixed(1)} FPS</span>
        <span>고정 그리드 최대 {Number(metadata?.max_grid_density_people_per_m2 || 0).toFixed(2)} 명/㎡</span>
        <span>
          {forecastHorizonLabel(forecast)} 예측 {forecast?.predicted_roi_count ?? '-'}명
          {forecast && ` · ${forecastMethodLabel(forecast)}`}
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
  const [serverHealth, setServerHealth] = useState(null);
  const [serverError, setServerError] = useState(null);
  const [densityLogs, setDensityLogs] = useState([]);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    let disposed = false;
    const loadHealth = async () => {
      try {
        const response = await fetch(`${API_BASE}/health`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        if (disposed) return;
        setServerHealth(payload);
        setServerError(null);
        if (Array.isArray(payload.cameras) && payload.cameras.length) {
          setCameras(payload.cameras.map((camera) => ({
            id: camera.camera_id,
            name: camera.name || camera.camera_id,
          })));
        }
      } catch (error) {
        if (disposed) return;
        setServerError(error instanceof Error ? error.message : String(error));
      }
    };
    loadHealth();
    const timer = window.setInterval(loadHealth, 2000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, []);

  const updateMetrics = useCallback((cameraId, next) => {
    setCameraMetrics((previous) => ({
      ...previous,
      [cameraId]: { ...previous[cameraId], ...next },
    }));
  }, []);

  const updateDensityLog = useCallback((action, entry) => {
    setDensityLogs((previous) => {
      if (action === 'detected') return [...previous, entry].slice(-50);
      return previous.map((item) => item.id === entry.id ? { ...item, ...entry } : item);
    });
  }, []);

  const connectedCount = cameras.filter((camera) => cameraMetrics[camera.id]?.connected).length;
  const inferenceAvailable = Boolean(serverHealth?.inference?.available);
  const apiConnected = Boolean(serverHealth) && !serverError;

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

      <div className={`runtime-banner ${apiConnected && inferenceAvailable ? 'runtime-ready' : 'runtime-warning'}`}>
        <strong>
          {!apiConnected
            ? 'FastAPI 연결 실패'
            : inferenceAvailable
              ? 'Jetson TensorRT 연결됨'
              : 'FastAPI 연결됨 · TensorRT Worker 대기 중'}
        </strong>
        <span>
          {serverError
            ? `${API_BASE} · ${serverError}`
            : serverHealth?.inference?.reason || `API ${API_BASE}`}
        </span>
      </div>

      <div className="multi-camera-layout">
        <div className="camera-grid">
          {cameras.map((camera) => (
            <CameraFeed
              key={camera.id}
              camera={camera}
              onMetrics={updateMetrics}
              onDensityLog={updateDensityLog}
              inferenceAvailable={inferenceAvailable}
              workerCamera={serverHealth?.cameras?.find((item) => item.camera_id === camera.id)}
            />
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
                const forecast = forecastFor(metrics);
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
                      {forecastHorizonLabel(forecast)} 뒤 ROI {forecast?.predicted_roi_count ?? '-'}명 · {' '}
                      {forecastMethodLabel(forecast)}
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

      <section className="card density-log shared-density-log" aria-label="전체 카메라 밀집도 로그">
        <h2 className="list-title">전체 밀집도 로그</h2>
        <div className="density-log-columns" aria-hidden="true">
          <span>카메라</span><span>단계</span><span>밀집도</span><span>감지 시간</span><span>해제 시간</span>
        </div>
        {densityLogs.length === 0 ? (
          <p className="density-log-empty">밀집도 경고 기록이 없습니다.</p>
        ) : (
          <ul className="density-log-list" aria-live="polite">
            {densityLogs.map((entry) => (
              <li className="density-log-entry" key={entry.id}>
                <span className="density-log-camera">{entry.cameraName}</span>
                <span className={`density-log-level ${entry.level === 'Danger' ? 'log-danger' : 'log-caution'}`}>
                  {entry.level === 'Danger' ? '위험' : '주의'}
                </span>
                <span className="density-log-density">{entry.density.toFixed(2)} 명/㎡</span>
                <time>{entry.detectedAt}</time>
                <time>{entry.clearedAt || '—'}</time>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
