export const ROWS = 6;
export const COLS = 7;
export const EMPTY = 0;
export const C_PUCT = 1.5;

export function cloneBoard(board) {
  return board.map((row) => row.slice());
}

export function legalColumnsForBoard(board) {
  const legal = [];
  for (let col = 0; col < COLS; col += 1) {
    if (board[0][col] === EMPTY) legal.push(col);
  }
  return legal;
}

export function dropPieceOnBoard(board, column, piece) {
  for (let row = ROWS - 1; row >= 0; row -= 1) {
    if (board[row][column] === EMPTY) {
      board[row][column] = piece;
      return row;
    }
  }
  return -1;
}

export function hasConnect4OnBoard(board, piece) {
  for (let row = 0; row < ROWS; row += 1) {
    for (let col = 0; col < COLS; col += 1) {
      if (board[row][col] !== piece) continue;
      if (
        col + 3 < COLS &&
        board[row][col + 1] === piece &&
        board[row][col + 2] === piece &&
        board[row][col + 3] === piece
      )
        return true;
      if (
        row + 3 < ROWS &&
        board[row + 1][col] === piece &&
        board[row + 2][col] === piece &&
        board[row + 3][col] === piece
      )
        return true;
      if (
        row + 3 < ROWS &&
        col + 3 < COLS &&
        board[row + 1][col + 1] === piece &&
        board[row + 2][col + 2] === piece &&
        board[row + 3][col + 3] === piece
      )
        return true;
      if (
        row - 3 >= 0 &&
        col + 3 < COLS &&
        board[row - 1][col + 1] === piece &&
        board[row - 2][col + 2] === piece &&
        board[row - 3][col + 3] === piece
      )
        return true;
    }
  }
  return false;
}

function isTerminalBoard(board) {
  if (hasConnect4OnBoard(board, 1) || hasConnect4OnBoard(board, 2)) return true;
  return legalColumnsForBoard(board).length === 0;
}

export function normalizeMaskedPolicy(rawPolicy, legal) {
  const masked = new Float32Array(COLS);
  let total = 0;
  for (const col of legal) {
    const prob = Number.isFinite(rawPolicy[col])
      ? Math.max(0, rawPolicy[col])
      : 0;
    masked[col] = prob;
    total += prob;
  }
  if (total <= 0) {
    const fallback = 1 / legal.length;
    for (const col of legal) masked[col] = fallback;
    return masked;
  }
  for (const col of legal) masked[col] /= total;
  return masked;
}

function resolveInputName(sessionInputNames, preferredNames, fallbackIndex = 0) {
  for (const name of preferredNames) {
    if (sessionInputNames.includes(name)) return name;
  }
  return sessionInputNames[fallbackIndex] || preferredNames[0];
}

function buildInferenceFeeds(session, obsTensor, legalMaskTensor) {
  const inputNames = session.inputNames || [];
  const obsName = resolveInputName(inputNames, ["obs", "x"], 0);
  const maskName = resolveInputName(
    inputNames,
    ["legal_mask", "action_mask", "mask"],
    1,
  );
  return {
    [obsName]: obsTensor,
    [maskName]: legalMaskTensor,
  };
}

function buildObsForPlayer(board, toPlay) {
  const obs = new Float32Array(2 * ROWS * COLS);
  const opp = toPlay === 1 ? 2 : 1;
  let idx = 0;
  for (let ch = 0; ch < 2; ch += 1) {
    for (let row = 0; row < ROWS; row += 1) {
      for (let col = 0; col < COLS; col += 1) {
        const value = board[row][col];
        obs[idx] =
          ch === 0 ? (value === toPlay ? 1 : 0) : value === opp ? 1 : 0;
        idx += 1;
      }
    }
  }
  return obs;
}

async function inferPolicyValue(session, ort, board, toPlay) {
  const legal = legalColumnsForBoard(board);
  if (legal.length === 0) {
    return { legal, policy: new Float32Array(COLS), value: 0 };
  }
  const obs = new ort.Tensor("float32", buildObsForPlayer(board, toPlay), [
    1,
    2,
    ROWS,
    COLS,
  ]);
  const mask = new Uint8Array(COLS);
  for (const col of legal) mask[col] = 1;
  const legalMask = new ort.Tensor("bool", mask, [1, COLS]);
  const outputs = await session.run(buildInferenceFeeds(session, obs, legalMask));
  const rawPolicy = outputs.policy?.data ?? outputs.logits?.data;
  const rawValue = outputs.value?.data?.[0] ?? 0;
  return {
    legal,
    policy: normalizeMaskedPolicy(rawPolicy, legal),
    value: Number(rawValue),
  };
}

function createNode(board, toPlay, prior = 0) {
  return {
    board,
    toPlay,
    prior,
    visitCount: 0,
    valueSum: 0,
    children: new Map(),
    expanded: false,
  };
}

export function pickChildByPuct(node, cPuct = C_PUCT) {
  const parentSqrt = Math.sqrt(Math.max(1, node.visitCount));
  let bestScore = Number.NEGATIVE_INFINITY;
  let best = null;
  for (const [move, child] of node.children.entries()) {
    const q = child.visitCount > 0 ? -(child.valueSum / child.visitCount) : 0;
    const u = cPuct * child.prior * (parentSqrt / (1 + child.visitCount));
    const score = q + u;
    if (score > bestScore) {
      bestScore = score;
      best = [move, child];
    }
  }
  return best;
}

function expandNode(node, policy, legal) {
  if (node.expanded) return;
  for (const move of legal) {
    const nextBoard = cloneBoard(node.board);
    dropPieceOnBoard(nextBoard, move, node.toPlay);
    const nextPlayer = node.toPlay === 1 ? 2 : 1;
    node.children.set(move, createNode(nextBoard, nextPlayer, policy[move]));
  }
  node.expanded = true;
}

function backprop(path, leafValue) {
  let value = leafValue;
  for (let i = path.length - 1; i >= 0; i -= 1) {
    const node = path[i];
    node.visitCount += 1;
    node.valueSum += value;
    value = -value;
  }
}

export async function runSearch(evaluate, board, toPlay, sims = 80) {
  const root = createNode(cloneBoard(board), toPlay, 1);
  const rootEval = await evaluate(root.board, root.toPlay);
  expandNode(root, rootEval.policy, rootEval.legal);

  const nSims = Math.max(1, sims);
  for (let sim = 0; sim < nSims; sim += 1) {
    const path = [root];
    let node = root;
    while (node.expanded && node.children.size > 0) {
      const picked = pickChildByPuct(node);
      if (!picked) break;
      node = picked[1];
      path.push(node);
      if (isTerminalBoard(node.board)) break;
    }

    let leafValue = 0;
    if (hasConnect4OnBoard(node.board, node.toPlay === 1 ? 2 : 1)) {
      leafValue = -1;
    } else if (legalColumnsForBoard(node.board).length === 0) {
      leafValue = 0;
    } else {
      const evalResult = await evaluate(node.board, node.toPlay);
      expandNode(node, evalResult.policy, evalResult.legal);
      leafValue = evalResult.value;
    }
    backprop(path, leafValue);
  }

  const visitPolicy = new Float32Array(COLS);
  let bestMove = -1;
  let bestVisits = -1;
  let totalVisits = 0;
  for (const [move, child] of root.children.entries()) {
    visitPolicy[move] = child.visitCount;
    totalVisits += child.visitCount;
    if (child.visitCount > bestVisits) {
      bestVisits = child.visitCount;
      bestMove = move;
    }
  }
  if (totalVisits > 0) {
    for (let col = 0; col < COLS; col += 1) visitPolicy[col] /= totalVisits;
  }
  return { move: bestMove, visitPolicy };
}

export async function runMcts(session, ort, board, aiPiece, sims = 80) {
  const result = await runSearch(
    (nextBoard, toPlay) => inferPolicyValue(session, ort, nextBoard, toPlay),
    board,
    aiPiece,
    sims,
  );
  return result.move;
}
