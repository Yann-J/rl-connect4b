import test from "node:test";
import assert from "node:assert/strict";

import { normalizeMaskedPolicy, pickChildByPuct } from "./mcts.js";

test("normalizeMaskedPolicy zeroes illegal columns and renormalizes", () => {
  const policy = normalizeMaskedPolicy(new Float32Array([0.1, 0.2, 0.5, 0.2, 0, 0, 0]), [1, 3]);
  assert.equal(policy[0], 0);
  assert.equal(policy[2], 0);
  assert.equal(policy[1] + policy[3], 1);
});

test("normalizeMaskedPolicy falls back to uniform on invalid policy", () => {
  const policy = normalizeMaskedPolicy(new Float32Array([0, 0, 0, 0, 0, 0, 0]), [2, 4, 6]);
  assert.ok(Math.abs(policy[2] - 1 / 3) < 1e-6);
  assert.ok(Math.abs(policy[4] - 1 / 3) < 1e-6);
  assert.ok(Math.abs(policy[6] - 1 / 3) < 1e-6);
});

test("pickChildByPuct prefers highest q plus exploration bonus", () => {
  const node = {
    visitCount: 16,
    children: new Map([
      [0, { prior: 0.1, visitCount: 6, valueSum: 1.8 }],
      [1, { prior: 0.2, visitCount: 8, valueSum: 4.8 }],
    ]),
  };
  const [bestMove] = pickChildByPuct(node, 1.5);
  assert.equal(bestMove, 1);
});
