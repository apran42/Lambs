import test from 'node:test';
import assert from 'node:assert/strict';

import { groupDensityLogsByCamera, updateZoneAlerts } from './densityLog.js';

const camera = { id: 'cam-01', name: 'CAM 01' };
const cell = (id, level, count) => ({
  id, level, count, density_people_per_m2: count,
});
const createId = (zoneId) => `cam-01-zone-${zoneId}`;

test('simultaneous alerts remain separate for each grid zone', () => {
  const first = updateZoneAlerts(
    new Map(),
    [cell(1, 'Caution', 3), cell(2, 'Danger', 5), cell(3, 'Relaxed', 0)],
    camera,
    '10:00',
    createId,
  );
  assert.equal(first.activeZones.size, 2);
  assert.deepEqual(first.changes.map((change) => change.entry.zoneId), [1, 2]);
  assert.deepEqual(first.raisedZones.map((zone) => zone.zoneId), [1, 2]);

  const second = updateZoneAlerts(
    first.activeZones,
    [cell(1, 'Relaxed', 0), cell(2, 'Danger', 5), cell(3, 'Relaxed', 0)],
    camera,
    '10:01',
    createId,
  );
  assert.equal(second.activeZones.has(1), false);
  assert.equal(second.activeZones.has(2), true);
  assert.deepEqual(second.changes, [{
    action: 'cleared',
    entry: { id: 'cam-01-zone-1', clearedAt: '10:01', closedReason: '정상 복귀' },
  }]);
});

test('an escalating zone updates the same incident and preserves its peak', () => {
  const first = updateZoneAlerts(
    new Map(), [cell(5, 'Caution', 3)], camera, '10:00', createId,
  );
  const repeated = updateZoneAlerts(
    first.activeZones, [cell(5, 'Caution', 3)], camera, '10:01', createId,
  );
  assert.equal(repeated.changes.length, 0);

  const escalated = updateZoneAlerts(
    repeated.activeZones, [cell(5, 'Danger', 5)], camera, '10:02', createId,
  );
  assert.equal(escalated.changes[0].action, 'updated');
  assert.equal(escalated.changes[0].entry.id, 'cam-01-zone-5');
  assert.equal(escalated.changes[0].entry.peakDensity, 5);
  assert.equal(escalated.raisedZones[0].level, 'Danger');

  const eased = updateZoneAlerts(
    escalated.activeZones, [cell(5, 'Caution', 3)], camera, '10:03', createId,
  );
  assert.equal(eased.changes.length, 0);
  assert.equal(eased.activeZones.get(5).level, 'Danger');
});

test('an unobservable zone is closed as a configuration change, not as safe', () => {
  const first = updateZoneAlerts(
    new Map(), [cell(9, 'Danger', 6)], camera, '10:00', createId,
  );
  const changed = updateZoneAlerts(
    first.activeZones, [cell(1, 'Relaxed', 0)], camera, '10:01', createId,
  );
  assert.equal(changed.activeZones.size, 0);
  assert.equal(changed.changes[0].entry.closedReason, '구역 설정 변경');
});

test('unknown cell status does not clear an active alert', () => {
  const first = updateZoneAlerts(
    new Map(), [cell(4, 'Caution', 3)], camera, '10:00', createId,
  );
  const unknown = updateZoneAlerts(
    first.activeZones, [cell(4, 'Unavailable', 0)], camera, '10:01', createId,
  );
  assert.equal(unknown.activeZones.has(4), true);
  assert.equal(unknown.changes.length, 0);
});

test('logs are grouped by CAM while each row retains its zone', () => {
  const logs = [
    { id: 'a', cameraId: 'cam-02', cameraName: 'CAM 02', zoneId: 2 },
    { id: 'b', cameraId: 'cam-01', cameraName: 'CAM 01', zoneId: 1 },
    { id: 'c', cameraId: 'cam-01', cameraName: 'CAM 01', zoneId: 5 },
    { id: 'd', cameraId: 'cam-02', cameraName: 'CAM 02', zoneId: 7 },
  ];
  const groups = groupDensityLogsByCamera(logs);
  assert.deepEqual(groups.map((group) => group.cameraId), ['cam-01', 'cam-02']);
  assert.deepEqual(groups[0].entries.map((entry) => entry.zoneId), [5, 1]);
  assert.deepEqual(groups[1].entries.map((entry) => entry.zoneId), [7, 2]);
});
