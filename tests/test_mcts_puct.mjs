import assert from "node:assert/strict";
import test from "node:test";
import { pickChildByPuct } from "../web/mcts.js";

test("PUCT uses negated child Q so a child that is good for the opponent is not preferred", () => {
  const goodForOpponent = {
    visitCount: 10,
    valueSum: 10,
    prior: 0.5,
    children: new Map(),
  };
  const goodForUs = {
    visitCount: 10,
    valueSum: -10,
    prior: 0.5,
    children: new Map(),
  };
  const root = {
    visitCount: 20,
    children: new Map([
      [0, goodForOpponent],
      [3, goodForUs],
    ]),
  };
  const picked = pickChildByPuct(root, 1.5);
  assert.ok(picked);
  assert.equal(picked[0], 3);
});
