#!/usr/bin/env node
import { runSearch, legalColumnsForBoard, COLS } from "../web/mcts.js";

function constantEvaluate(value) {
  return async (board, _toPlay) => {
    const legal = legalColumnsForBoard(board);
    const policy = new Float32Array(COLS);
    if (legal.length === 0) {
      return { legal, policy, value: 0 };
    }
    const mass = 1 / legal.length;
    for (const col of legal) policy[col] = mass;
    return { legal, policy, value };
  };
}

const raw = await new Promise((resolve, reject) => {
  const chunks = [];
  process.stdin.setEncoding("utf8");
  process.stdin.on("data", (chunk) => chunks.push(chunk));
  process.stdin.on("end", () => resolve(chunks.join("")));
  process.stdin.on("error", reject);
});

const spec = JSON.parse(raw);
const result = await runSearch(
  constantEvaluate(spec.value),
  spec.board,
  spec.toPlay,
  spec.sims,
);
process.stdout.write(
  JSON.stringify({
    move: result.move,
    visitPolicy: Array.from(result.visitPolicy),
  }),
);
