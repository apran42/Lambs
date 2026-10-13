const ALERT_SEVERITY = { Caution: 1, Danger: 2 };

// Keep one incident per camera/grid cell until that cell explicitly returns to Relaxed.
export function updateZoneAlerts(activeZones, cells, camera, timestamp, createId) {
  const nextActive = new Map(activeZones);
  const changes = [];
  const raisedZones = [];
  const seen = new Set();

  for (const cell of cells || []) {
    const zoneId = Number(cell.id);
    if (!Number.isInteger(zoneId) || zoneId < 1 || seen.has(zoneId)) continue;
    seen.add(zoneId);

    const previous = nextActive.get(zoneId);
    const level = cell.level;
    const rawDensity = Number(cell.density_people_per_m2);
    const rawCount = Number(cell.count);
    const density = Number.isFinite(rawDensity) ? Math.max(0, rawDensity) : 0;
    const count = Number.isFinite(rawCount) ? Math.max(0, rawCount) : 0;

    if (level === 'Relaxed') {
      if (previous) {
        changes.push({
          action: 'cleared',
          entry: { id: previous.id, clearedAt: timestamp, closedReason: '정상 복귀' },
        });
        nextActive.delete(zoneId);
      }
      continue;
    }
    if (!ALERT_SEVERITY[level]) continue;

    if (!previous) {
      const id = createId(zoneId);
      nextActive.set(zoneId, { id, level, peakDensity: density, maxCount: count });
      changes.push({
        action: 'detected',
        entry: {
          id,
          cameraId: camera.id,
          cameraName: camera.name,
          zoneId,
          level,
          peakDensity: density,
          maxCount: count,
          detectedAt: timestamp,
          clearedAt: '',
          closedReason: '',
        },
      });
      raisedZones.push({ zoneId, level });
      continue;
    }

    const escalated = ALERT_SEVERITY[level] > ALERT_SEVERITY[previous.level];
    const nextLevel = escalated ? level : previous.level;
    const peakDensity = Math.max(previous.peakDensity, density);
    const maxCount = Math.max(previous.maxCount, count);
    if (escalated || peakDensity > previous.peakDensity || maxCount > previous.maxCount) {
      nextActive.set(zoneId, { ...previous, level: nextLevel, peakDensity, maxCount });
      changes.push({
        action: 'updated',
        entry: { id: previous.id, level: nextLevel, peakDensity, maxCount },
      });
      if (escalated) raisedZones.push({ zoneId, level });
    }
  }

  // A removed cell is no longer observable; do not label it safely cleared.
  for (const [zoneId, previous] of nextActive) {
    if (!seen.has(zoneId)) {
      changes.push({
        action: 'cleared',
        entry: { id: previous.id, clearedAt: timestamp, closedReason: '구역 설정 변경' },
      });
      nextActive.delete(zoneId);
    }
  }

  return { activeZones: nextActive, changes, raisedZones };
}

export function groupDensityLogsByCamera(logs) {
  const groups = new Map();
  for (const entry of [...logs].reverse()) {
    if (!groups.has(entry.cameraId)) {
      groups.set(entry.cameraId, {
        cameraId: entry.cameraId,
        cameraName: entry.cameraName,
        entries: [],
      });
    }
    groups.get(entry.cameraId).entries.push(entry);
  }
  return [...groups.values()].sort((a, b) => a.cameraId.localeCompare(b.cameraId));
}
