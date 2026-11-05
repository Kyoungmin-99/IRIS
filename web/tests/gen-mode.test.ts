import { test } from 'node:test';
import assert from 'node:assert/strict';
import { genAutoCheck } from '../src/beauty/gen-mode.ts';

test('생성 서버 자동 확인: 로컬·?gen 일 때만(배포 사이트는 127.0.0.1 을 두드리지 않음)', () => {
  assert.equal(genAutoCheck('localhost', ''), true);
  assert.equal(genAutoCheck('127.0.0.1', ''), true);
  assert.equal(genAutoCheck('192.168.0.5', ''), true);
  assert.equal(genAutoCheck('rudals906377.github.io', '?mode=photo'), false);
  assert.equal(genAutoCheck('rudals906377.github.io', '?mode=photo&gen=1'), true);
});
