import {
  COLS,
  ROWS,
  legalColumnsForBoard,
  normalizeMaskedPolicy,
  runMcts,
} from "./mcts.js";

const EMPTY = 0;
const P1 = 1;
const P2 = 2;

const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");
const legendEl = document.getElementById("legend");
const newHumanFirstBtn = document.getElementById("new-human-first");
const newAiFirstBtn = document.getElementById("new-ai-first");
const openSettingsBtn = document.getElementById("open-settings");
const settingsModalEl = document.getElementById("settings-modal");
const modelNameInput = document.getElementById("model-name-input");
const mctsSimsInput = document.getElementById("mcts-sims-input");
const cancelSettingsBtn = document.getElementById("cancel-settings");
const saveSettingsBtn = document.getElementById("save-settings");
const modeSelectEl = document.getElementById("ai-mode");

let board = createEmptyBoard();
let done = false;
let winner = null;
let currentPlayer = P1;
let humanPiece = P1;
let aiPiece = P2;
let session = null;
let isAnimatingMove = false;
let endgameAnimationPlayed = false;
let downloadedModelBytes = 0;
let modelLoadError = null;
const MODEL_STORAGE_KEY = "connect4-model-name";
const DEFAULT_MODEL_NAME = "policy.onnx";
const MODE_STORAGE_KEY = "connect4-ai-mode";
const FAST_MODE = "fast";
const STRONG_MODE = "strong";
const MCTS_SIMS_STORAGE_KEY = "connect4-mcts-sims";
const DEFAULT_MCTS_SIMS = 80;
const MIN_MCTS_SIMS = 1;
const MAX_MCTS_SIMS = 5000;
let modelName = loadModelNameSetting();
let aiMode = loadAiModeSetting();
let mctsSimulations = loadMctsSimulationsSetting();

function normalizeModelName(rawName) {
  const trimmed = String(rawName || "").trim();
  return trimmed.length > 0 ? trimmed : DEFAULT_MODEL_NAME;
}

function loadModelNameSetting() {
  const stored = window.localStorage.getItem(MODEL_STORAGE_KEY);
  return normalizeModelName(stored);
}

function saveModelNameSetting(nextName) {
  const normalized = normalizeModelName(nextName);
  window.localStorage.setItem(MODEL_STORAGE_KEY, normalized);
  return normalized;
}

function createEmptyBoard() {
  return Array.from({ length: ROWS }, () => Array(COLS).fill(EMPTY));
}

function legalColumns() {
  return legalColumnsForBoard(board);
}

function dropPiece(column, piece) {
  for (let row = ROWS - 1; row >= 0; row -= 1) {
    if (board[row][column] === EMPTY) {
      board[row][column] = piece;
      return row;
    }
  }
  return -1;
}

function hasConnect4(piece) {
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

function buildObsForAi() {
  const obs = new Float32Array(2 * ROWS * COLS);
  let idx = 0;
  for (let ch = 0; ch < 2; ch += 1) {
    for (let row = 0; row < ROWS; row += 1) {
      for (let col = 0; col < COLS; col += 1) {
        const value = board[row][col];
        if (ch === 0) obs[idx] = value === aiPiece ? 1 : 0;
        else obs[idx] = value === humanPiece ? 1 : 0;
        idx += 1;
      }
    }
  }
  return obs;
}

async function chooseAiColumn() {
  const legal = legalColumns();
  if (legal.length === 0) return -1;

  if (aiMode === STRONG_MODE) {
    return runMcts(session, ort, board, aiPiece, mctsSimulations);
  }

  const obs = new ort.Tensor("float32", buildObsForAi(), [1, 2, ROWS, COLS]);
  const maskData = new Uint8Array(COLS);
  for (const col of legal) maskData[col] = 1;
  const legalMask = new ort.Tensor("bool", maskData, [1, COLS]);
  const outputs = await session.run({ x: obs, legal_mask: legalMask });
  const rawPolicy = outputs.policy?.data ?? outputs.logits?.data;
  const policy = normalizeMaskedPolicy(rawPolicy, legal);

  let bestCol = legal[0];
  let bestProb = Number.NEGATIVE_INFINITY;
  for (const col of legal) {
    if (policy[col] > bestProb) {
      bestProb = policy[col];
      bestCol = col;
    }
  }
  return bestCol;
}

function updateStatus(text) {
  statusEl.textContent = text;
}

function updateLegend() {
  const modeLabel =
    aiMode === STRONG_MODE
      ? `MCTS strong (${mctsSimulations} sims)`
      : "policy fast";
  legendEl.textContent = `🟡 = you  🔴 = ${modeLabel} (${modelName})`;
}

function normalizeAiMode(mode) {
  return mode === STRONG_MODE ? STRONG_MODE : FAST_MODE;
}

function loadAiModeSetting() {
  return normalizeAiMode(window.localStorage.getItem(MODE_STORAGE_KEY));
}

function saveAiModeSetting(nextMode) {
  const normalized = normalizeAiMode(nextMode);
  window.localStorage.setItem(MODE_STORAGE_KEY, normalized);
  return normalized;
}

function normalizeMctsSimulations(rawValue) {
  const parsed = Number.parseInt(String(rawValue ?? ""), 10);
  if (!Number.isFinite(parsed)) return DEFAULT_MCTS_SIMS;
  return Math.max(MIN_MCTS_SIMS, Math.min(MAX_MCTS_SIMS, parsed));
}

function loadMctsSimulationsSetting() {
  return normalizeMctsSimulations(
    window.localStorage.getItem(MCTS_SIMS_STORAGE_KEY),
  );
}

function saveMctsSimulationsSetting(nextValue) {
  const normalized = normalizeMctsSimulations(nextValue);
  window.localStorage.setItem(MCTS_SIMS_STORAGE_KEY, String(normalized));
  return normalized;
}

function formatDownloadedBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIdx = 0;
  while (value >= 1024 && unitIdx < units.length - 1) {
    value /= 1024;
    unitIdx += 1;
  }
  const rounded =
    value >= 100 || unitIdx === 0
      ? Math.round(value).toString()
      : value.toFixed(1);
  return `${rounded} ${units[unitIdx]}`;
}

function gameStateText() {
  if (modelLoadError) {
    return modelLoadError;
  }
  if (done) {
    if (winner === humanPiece) return "You win! 🎉";
    if (winner === aiPiece) return "AI wins... 😭";
    return "Draw... 🤝";
  }
  if (!session) {
    if (downloadedModelBytes > 0) {
      return `Loading ONNX model... (${formatDownloadedBytes(downloadedModelBytes)})`;
    }
    return "Loading ONNX model...";
  }
  return currentPlayer === humanPiece
    ? "Your turn... ⏳"
    : `AI is thinking (${aiMode})... 🤔`;
}

function render(animatedMove = null) {
  boardEl.innerHTML = "";
  const legal = new Set(legalColumns());
  for (let row = 0; row < ROWS; row += 1) {
    for (let col = 0; col < COLS; col += 1) {
      const value = board[row][col];
      const btn = document.createElement("button");
      btn.className = "cell";
      btn.dataset.row = String(row);
      btn.dataset.col = String(col);
      if (value === humanPiece) btn.classList.add("human");
      if (value === aiPiece) btn.classList.add("ai");
      if (
        animatedMove &&
        row === animatedMove.row &&
        col === animatedMove.col &&
        value === animatedMove.piece
      ) {
        btn.classList.add("dropping");
      }
      const humanTurn =
        !done && currentPlayer === humanPiece && !isAnimatingMove;
      if (humanTurn && legal.has(col)) {
        btn.classList.add("playable");
        btn.disabled = false;
        btn.title = `Drop in column ${col + 1}`;
        btn.addEventListener("click", () => humanMove(col));
      } else {
        btn.disabled = true;
      }
      boardEl.appendChild(btn);
    }
  }
  updateStatus(gameStateText());
}

function applyMove(column, piece) {
  const row = dropPiece(column, piece);
  if (row < 0) return null;
  if (hasConnect4(piece)) {
    done = true;
    winner = piece;
  } else if (legalColumns().length === 0) {
    done = true;
    winner = null;
  } else {
    currentPlayer = piece === P1 ? P2 : P1;
  }
  return row;
}

function animateDrop(row, col) {
  const prefersReducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  ).matches;
  if (prefersReducedMotion) return Promise.resolve();

  const targetCell = boardEl.querySelector(
    `.cell[data-row="${row}"][data-col="${col}"]`,
  );
  if (!targetCell) return Promise.resolve();

  const boardStyle = getComputedStyle(boardEl);
  const gap = Number.parseFloat(boardStyle.gap || "0") || 0;
  const cellSize = targetCell.getBoundingClientRect().height;
  const startOffset = (row + 1) * (cellSize + gap);
  const duration = Math.min(500, 260 + row * 35);

  const animation = targetCell.animate(
    [
      {
        transform: `translateY(-${startOffset}px)`,
        offset: 0,
        easing: "cubic-bezier(0.15, 0.9, 0.3, 1)",
      },
      { transform: "translateY(0)", offset: 0.8, easing: "ease-out" },
      { transform: "translateY(-7px)", offset: 0.92, easing: "ease-in-out" },
      { transform: "translateY(0)", offset: 1 },
    ],
    {
      duration: duration + 140,
      easing: "linear",
      fill: "both",
    },
  );

  return animation.finished.catch(() => undefined);
}

function animateFireworks() {
  const prefersReducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  ).matches;
  if (prefersReducedMotion) return Promise.resolve();

  const particles = [];
  boardEl.classList.add("board-win");
  const burstWaves = [
    { count: 32, spread: 220, duration: 980, delayMax: 120 },
    { count: 26, spread: 200, duration: 900, delayMax: 220 },
    { count: 20, spread: 180, duration: 840, delayMax: 320 },
  ];
  for (const wave of burstWaves) {
    for (let i = 0; i < wave.count; i += 1) {
      const particle = document.createElement("div");
      particle.className = "firework-particle";
      particle.style.left = `${8 + Math.random() * 84}%`;
      particle.style.top = `${4 + Math.random() * 58}%`;
      particle.style.background =
        i % 3 === 0 ? "#ffffff" : i % 2 === 0 ? "#f4ce46" : "#ffd86a";
      particle.style.setProperty(
        "--dx",
        `${-wave.spread + Math.random() * wave.spread * 2}px`,
      );
      particle.style.setProperty(
        "--dy",
        `${-wave.spread + Math.random() * wave.spread * 1.8}px`,
      );
      particle.style.setProperty(
        "--delay",
        `${Math.random() * wave.delayMax}ms`,
      );
      particle.style.setProperty("--duration", `${wave.duration}ms`);
      particle.style.setProperty("--size", `${5 + Math.random() * 8}px`);
      boardEl.appendChild(particle);
      particles.push(particle);
    }
  }

  return new Promise((resolve) => {
    window.setTimeout(() => {
      boardEl.classList.remove("board-win");
      for (const particle of particles) particle.remove();
      resolve();
    }, 1450);
  });
}

function animateLoss() {
  const prefersReducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  ).matches;
  if (prefersReducedMotion) return Promise.resolve();

  const debris = [];
  const count = 26;
  for (let i = 0; i < count; i += 1) {
    const particle = document.createElement("div");
    particle.className = "loss-particle";
    particle.style.left = `${10 + Math.random() * 80}%`;
    particle.style.top = `${5 + Math.random() * 25}%`;
    particle.style.setProperty("--dx", `${-80 + Math.random() * 160}px`);
    particle.style.setProperty("--dy", `${90 + Math.random() * 120}px`);
    particle.style.setProperty("--delay", `${Math.random() * 120}ms`);
    particle.style.setProperty("--size", `${4 + Math.random() * 7}px`);
    boardEl.appendChild(particle);
    debris.push(particle);
  }

  boardEl.classList.add("board-loss", "board-loss-impact");
  return new Promise((resolve) => {
    window.setTimeout(() => {
      boardEl.classList.remove("board-loss", "board-loss-impact");
      for (const particle of debris) particle.remove();
      resolve();
    }, 1050);
  });
}

async function animateEndgame() {
  if (!done || endgameAnimationPlayed) return;
  endgameAnimationPlayed = true;
  if (winner === humanPiece) {
    await animateFireworks();
  } else if (winner === aiPiece) {
    await animateLoss();
  }
}

async function playMove(column, piece) {
  const row = applyMove(column, piece);
  if (row === null) return false;
  isAnimatingMove = true;
  render({ row, col: column, piece });
  await animateDrop(row, column);
  isAnimatingMove = false;
  render();
  await animateEndgame();
  return true;
}

async function maybeAiMove() {
  if (done || currentPlayer !== aiPiece || !session) return;
  render();
  const col = await chooseAiColumn();
  if (col >= 0) await playMove(col, aiPiece);
}

async function humanMove(column) {
  if (done || currentPlayer !== humanPiece || isAnimatingMove) return;
  if (!legalColumns().includes(column)) return;
  await playMove(column, humanPiece);
  await maybeAiMove();
}

async function newGame(humanStarts) {
  board = createEmptyBoard();
  done = false;
  winner = null;
  endgameAnimationPlayed = false;
  humanPiece = humanStarts ? P1 : P2;
  aiPiece = humanStarts ? P2 : P1;
  currentPlayer = P1;
  render();
  await maybeAiMove();
}

async function loadModel() {
  try {
    session = null;
    downloadedModelBytes = 0;
    modelLoadError = null;
    updateStatus(gameStateText());
    const modelUrl = `./${modelName}`;
    const response = await fetch(modelUrl);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    downloadedModelBytes =
      Number.parseInt(response.headers.get("content-length") || "0", 10) || 0;
    updateStatus(gameStateText());
    session = await ort.InferenceSession.create(modelUrl);
    await newGame(true);
  } catch (error) {
    modelLoadError = `Failed to load ${modelName}: ${error.message}`;
    updateStatus(gameStateText());
    render();
  }
}

function openSettingsModal() {
  modeSelectEl.value = aiMode;
  modelNameInput.value = modelName;
  mctsSimsInput.value = String(mctsSimulations);
  settingsModalEl.classList.remove("hidden");
  modeSelectEl.focus();
}

function closeSettingsModal() {
  settingsModalEl.classList.add("hidden");
}

async function handleSaveSettings() {
  const nextAiMode = saveAiModeSetting(modeSelectEl.value);
  const nextModelName = saveModelNameSetting(modelNameInput.value);
  const nextMctsSimulations = saveMctsSimulationsSetting(mctsSimsInput.value);
  const modeChanged = nextAiMode !== aiMode;
  const modelChanged = nextModelName !== modelName;
  const simsChanged = nextMctsSimulations !== mctsSimulations;

  aiMode = nextAiMode;
  modelName = nextModelName;
  mctsSimulations = nextMctsSimulations;
  updateLegend();
  render();
  closeSettingsModal();
  if (modelChanged) {
    await loadModel();
    return;
  }
  if (modeChanged || simsChanged) {
    await maybeAiMove();
  }
}

newHumanFirstBtn.addEventListener("click", () => newGame(true));
newAiFirstBtn.addEventListener("click", () => newGame(false));
openSettingsBtn.addEventListener("click", openSettingsModal);
cancelSettingsBtn.addEventListener("click", closeSettingsModal);
saveSettingsBtn.addEventListener("click", () => {
  handleSaveSettings();
});
modelNameInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    handleSaveSettings();
  } else if (event.key === "Escape") {
    event.preventDefault();
    closeSettingsModal();
  }
});
mctsSimsInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    handleSaveSettings();
  } else if (event.key === "Escape") {
    event.preventDefault();
    closeSettingsModal();
  }
});
settingsModalEl.addEventListener("click", (event) => {
  if (event.target === settingsModalEl) closeSettingsModal();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !settingsModalEl.classList.contains("hidden")) {
    closeSettingsModal();
  }
});

updateLegend();
modeSelectEl.value = aiMode;
render();
loadModel();
