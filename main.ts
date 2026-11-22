// main.ts -- Sokoban solver for the make.fun competition
// "Build the best Sokoban solver, 21 November 2026".
//
// I/O contract: reads levels from argv[2] or MAKEFUN_INPUT_DIR (default /input),
// writes solutions to argv[3] or MAKEFUN_OUTPUT_DIR (default /output).
// Never removes the output directory itself — only creates per-level
// subdirectories inside it.
//
// Plain JavaScript (no TypeScript-only syntax) so Node 22 runs it as-is.
//
// How it works
// -----------
// The generator builds every level as a one-square-wide hallway with the
// player in it, plus one to four rooms off the hallway (3-4 boxes each).
// A box pushed out of its room can never come back, so each room is an
// independent puzzle and the level optimum is the sum of the room optima.
// This program therefore
//   1. splits the level into rooms by cutting the hallway line,
//   2. solves each room push-optimally with A* over push states, using a
//      min-matching lower bound on box-to-goal distances, dead-square
//      analysis and 2x2-block deadlock pruning, with a transposition table
//      keyed by a compact numeric state code,
//   3. requires every room but the last to leave the player able to walk
//      back out through its door, and solves last the room where that exit
//      costs the most pushes,
//   4. walks the player between rooms through the hallway and replays the
//      whole move string to verify it before writing it out.
// If a search exhausts its budget, weighted A* and then greedy best-first
// search try to find any solution (a suboptimal solve still scores).
// If the level does not match the hallway-plus-rooms shape at all, a
// whole-level greedy search is tried as a last resort.
//
// Determinism: no clock, no randomness, no environment reads. Every search
// is bounded by a fixed number of node expansions, never by elapsed time,
// so two runs on the same input write identical bytes.

import fs from "node:fs";
import path from "node:path";

const DR = [-1, 1, 0, 0];
const DC = [0, 0, -1, 1];
const LETTER = ["u", "d", "l", "r"];
const OPP = [1, 0, 3, 2];
const INF = 1000000000;

// Work budgets, counted in node expansions (deterministic).
const ASTAR_BUDGET = 250000;
const WEIGHTED_BUDGET = 150000;
const GREEDY_BUDGET = 150000;
const FALLBACK_ASTAR_BUDGET = 150000;
const FALLBACK_WEIGHTED_BUDGET = 100000;
const FALLBACK_BUDGET = 60000;
const ROOM_TOTAL_BUDGET = 800000;
const GLOBAL_BUDGET = 2000000;

// ---------------------------------------------------------------- heap ---

// Binary min-heap of (key, priority, g) in parallel arrays.
class Heap {
  constructor() {
    this.k = [];
    this.p = [];
    this.g = [];
  }
  get size() {
    return this.k.length;
  }
  push(key, pri, g) {
    const k = this.k;
    const p = this.p;
    const gg = this.g;
    let i = k.length;
    k.push(key);
    p.push(pri);
    gg.push(g);
    while (i > 0) {
      const par = (i - 1) >> 1;
      if (p[par] <= p[i]) break;
      let t = k[par];
      k[par] = k[i];
      k[i] = t;
      t = p[par];
      p[par] = p[i];
      p[i] = t;
      t = gg[par];
      gg[par] = gg[i];
      gg[i] = t;
      i = par;
    }
  }
  pop() {
    const k = this.k;
    const p = this.p;
    const gg = this.g;
    const n = k.length - 1;
    const tk = k[0];
    const tp = p[0];
    const tg = gg[0];
    if (n > 0) {
      k[0] = k[n];
      p[0] = p[n];
      gg[0] = gg[n];
    }
    k.length = n;
    p.length = n;
    gg.length = n;
    let i = 0;
    for (;;) {
      const l = i * 2 + 1;
      const r = l + 1;
      let m = i;
      if (l < n && p[l] < p[m]) m = l;
      if (r < n && p[r] < p[m]) m = r;
      if (m === i) break;
      let t = k[m];
      k[m] = k[i];
      k[i] = t;
      t = p[m];
      p[m] = p[i];
      p[i] = t;
      t = gg[m];
      gg[m] = gg[i];
      gg[i] = t;
      i = m;
    }
    return [tk, tp, tg];
  }
}

// ------------------------------------------------------------- parsing ---

function parseLevel(text) {
  const lines = text.split(/\r?\n/);
  while (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
  const h = lines.length;
  let w = 0;
  for (const line of lines) if (line.length > w) w = line.length;
  const wall = new Uint8Array(w * h);
  const goal = new Uint8Array(w * h);
  const boxes = [];
  let player = -1;
  for (let r = 0; r < h; r++) {
    const line = lines[r];
    for (let c = 0; c < w; c++) {
      const ch = c < line.length ? line[c] : "#";
      const i = r * w + c;
      switch (ch) {
        case "#":
          wall[i] = 1;
          break;
        case ".":
          goal[i] = 1;
          break;
        case "$":
          boxes.push(i);
          break;
        case "*":
          goal[i] = 1;
          boxes.push(i);
          break;
        case "@":
          player = i;
          break;
        case "+":
          goal[i] = 1;
          player = i;
          break;
        default:
          break; // ' ' and anything else: open floor
      }
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goal, boxes, player };
}

// -------------------------------------------------------- decomposition ---

// Split the level into rooms by cutting the hallway line (one row or one
// column). Returns an array of rooms or null when the level does not match
// the hallway-plus-rooms shape.
function tryDecompose(level, isRow, line) {
  const { w, h, wall, goal, boxes } = level;
  const onLine = (i) => (isRow ? Math.floor(i / w) === line : i % w === line);
  const blocked = (i) => wall[i] === 1 || onLine(i);
  const seen = new Uint8Array(w * h);
  const boxSet = new Set(boxes);
  let totalGoals = 0;
  for (let i = 0; i < w * h; i++) if (goal[i]) totalGoals++;
  const rooms = [];
  let coveredBoxes = 0;
  let coveredGoals = 0;
  for (let s = 0; s < w * h; s++) {
    if (blocked(s) || seen[s]) continue;
    const cells = [];
    const stack = [s];
    seen[s] = 1;
    while (stack.length > 0) {
      const cur = stack.pop();
      cells.push(cur);
      const r = Math.floor(cur / w);
      const c = cur % w;
      for (let d = 0; d < 4; d++) {
        const nr = r + DR[d];
        const nc = c + DC[d];
        if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
        const ni = nr * w + nc;
        if (blocked(ni) || seen[ni]) continue;
        seen[ni] = 1;
        stack.push(ni);
      }
    }
    cells.sort((a, b) => a - b);
    const rboxes = [];
    const rgoals = [];
    for (const cell of cells) {
      if (boxSet.has(cell)) rboxes.push(cell);
      if (goal[cell]) rgoals.push(cell);
    }
    if (rboxes.length === 0) {
      if (rgoals.length > 0) return null; // goals with no boxes nearby
      continue;
    }
    if (rboxes.length !== rgoals.length) return null;
    // The generator builds rooms of 3-4 boxes; a larger "room" means the
    // cut found a spurious split, not a real room. Let the whole-level
    // fallback handle the level instead.
    if (rboxes.length > 4) return null;
    let door = -1;
    let entry = -1;
    for (const cell of cells) {
      const r = Math.floor(cell / w);
      const c = cell % w;
      for (let d = 0; d < 4; d++) {
        const nr = r + DR[d];
        const nc = c + DC[d];
        if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
        const ni = nr * w + nc;
        if (onLine(ni) && !wall[ni]) {
          door = cell;
          entry = ni;
          break;
        }
      }
      if (door >= 0) break;
    }
    if (door < 0) return null; // room does not open onto the hallway
    coveredBoxes += rboxes.length;
    coveredGoals += rgoals.length;
    rooms.push({ cells, door, entry, boxes: rboxes, goals: rgoals });
  }
  if (rooms.length === 0) return null;
  if (coveredBoxes !== boxes.length) return null;
  if (coveredGoals !== totalGoals) return null;
  return rooms;
}

// --------------------------------------------------------------- room -----

// Build the search tables for one room (or, for the whole-level fallback,
// for the whole level). comp.cells are the room's floor cells (global
// indices, sorted); entryGlobal is the hallway cell outside the door, or
// -1 for the whole-level fallback; startGlobal is where the player starts
// (the entry cell for a room, the player cell for the fallback).
function makeRoom(level, comp, entryGlobal, startGlobal) {
  const { w, h } = level;
  const cells = comp.cells;
  const n = cells.length;
  const localOf = new Int32Array(w * h).fill(-1);
  for (let i = 0; i < n; i++) localOf[cells[i]] = i;
  const hasEntry = entryGlobal >= 0;
  const entryLocal = hasEntry ? n : -1;
  const domain = n + (hasEntry ? 1 : 0);
  const globalOf = cells.slice();
  if (hasEntry) globalOf.push(entryGlobal);
  const domainMask = new Uint8Array(w * h);
  for (const g of cells) domainMask[g] = 1;
  if (hasEntry) domainMask[entryGlobal] = 1;

  // dest[(cell * 4) + dir]: destination local index, or -1.
  const dest = new Int32Array(domain * 4).fill(-1);
  for (let lc = 0; lc < domain; lc++) {
    const g = globalOf[lc];
    const r = Math.floor(g / w);
    const c = g % w;
    for (let d = 0; d < 4; d++) {
      const nr = r + DR[d];
      const nc = c + DC[d];
      if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
      const ng = nr * w + nc;
      if (localOf[ng] >= 0) dest[lc * 4 + d] = localOf[ng];
      else if (hasEntry && ng === entryGlobal) dest[lc * 4 + d] = entryLocal;
    }
  }

  const goals = comp.goals.map((g) => localOf[g]).sort((a, b) => a - b);
  const boxes = comp.boxes.map((g) => localOf[g]).sort((a, b) => a - b);
  const b = boxes.length;
  const goalSet = new Uint8Array(domain);
  for (const g of goals) goalSet[g] = 1;

  // BFS distances from each goal over room cells, boxes ignored.
  const dist = new Int32Array(n * b).fill(INF);
  for (let gi = 0; gi < b; gi++) {
    const g0 = goals[gi];
    const q = [g0];
    dist[g0 * b + gi] = 0;
    let head = 0;
    while (head < q.length) {
      const cur = q[head++];
      const nd = dist[cur * b + gi] + 1;
      for (let d = 0; d < 4; d++) {
        const nb = dest[cur * 4 + d];
        if (nb < 0 || nb >= n) continue;
        if (dist[nb * b + gi] === INF) {
          dist[nb * b + gi] = nd;
          q.push(nb);
        }
      }
    }
  }

  // Dead squares: room cells from which no goal is reachable by a box,
  // even with the player free to stand anywhere in the domain. Found by a
  // reverse-push flood from the goals: a box at y could have come from
  // x = y - d with the player at y - 2d.
  const canReach = new Uint8Array(n);
  const rq = [];
  for (const g0 of goals) {
    canReach[g0] = 1;
    rq.push(g0);
  }
  let rhead = 0;
  while (rhead < rq.length) {
    const y = rq[rhead++];
    const gy = globalOf[y];
    const yr = Math.floor(gy / w);
    const yc = gy % w;
    for (let d = 0; d < 4; d++) {
      const xr = yr - DR[d];
      const xc = yc - DC[d];
      const pr = yr - 2 * DR[d];
      const pc = yc - 2 * DC[d];
      if (xr < 0 || xr >= h || xc < 0 || xc >= w) continue;
      const xl = localOf[xr * w + xc];
      if (xl < 0 || canReach[xl]) continue;
      if (pr < 0 || pr >= h || pc < 0 || pc >= w) continue;
      if (!domainMask[pr * w + pc]) continue;
      canReach[xl] = 1;
      rq.push(xl);
    }
  }
  const dead = new Uint8Array(n);
  for (let i = 0; i < n; i++) dead[i] = canReach[i] ? 0 : 1;

  // 2x2 blocks: four cells, each a room cell or a wall. When every room
  // cell of such a block holds a box, every box in it is frozen: each of
  // its pushes needs a target or a standing cell inside the block. A block
  // holding a box that is not on a goal is therefore a deadlock.
  const blocks = [];
  const blocksByCell = Array.from({ length: n }, () => []);
  let minR = h;
  let maxR = -1;
  let minC = w;
  let maxC = -1;
  for (const g of cells) {
    const r = Math.floor(g / w);
    const c = g % w;
    if (r < minR) minR = r;
    if (r > maxR) maxR = r;
    if (c < minC) minC = c;
    if (c > maxC) maxC = c;
  }
  for (let r = minR - 1; r <= maxR; r++) {
    for (let c = minC - 1; c <= maxC; c++) {
      const rc = [];
      let skip = false;
      for (let dr = 0; dr < 2 && !skip; dr++) {
        for (let dc = 0; dc < 2; dc++) {
          const rr = r + dr;
          const cc = c + dc;
          if (rr < 0 || rr >= h || cc < 0 || cc >= w) continue; // outside: wall
          const g = rr * w + cc;
          if (hasEntry && g === entryGlobal) {
            skip = true; // the entry cell is open floor, never a wall
            break;
          }
          const l = localOf[g];
          if (l >= 0) rc.push(l);
          else if (!level.wall[g]) {
            skip = true; // open floor outside the room: not a wall
            break;
          }
        }
      }
      if (skip || rc.length === 0) continue;
      const bi = blocks.length;
      blocks.push(rc);
      for (const l of rc) blocksByCell[l].push(bi);
    }
  }

  // Goal permutations for the min-matching heuristic (exact up to 6 boxes;
  // past that the heuristic falls back to a sum of per-box minima).
  let perms = null;
  if (b <= 6) {
    perms = [];
    const arr = [];
    for (let i = 0; i < b; i++) arr.push(i);
    const gen = (k) => {
      if (k === 1) {
        perms.push(arr.slice());
        return;
      }
      for (let i = 0; i < k; i++) {
        gen(k - 1);
        const j = k % 2 === 1 ? 0 : i;
        const t = arr[j];
        arr[j] = arr[k - 1];
        arr[k - 1] = t;
      }
    };
    gen(b);
  }

  const numeric = n <= 63 && Math.pow(n, b) * 64 < 4294967296;
  const room = {
    level,
    w,
    h,
    n,
    b,
    domain,
    entryLocal,
    entryGlobal,
    startGlobal,
    globalOf,
    localOf,
    domainMask,
    dest,
    goals,
    boxes,
    goalSet,
    dist,
    dead,
    blocks,
    blocksByCell,
    perms,
    numeric,
    startPlayer: hasEntry ? entryLocal : localOf[startGlobal],
    hCache: new Map(),
  };
  room.boxesKey = numeric
    ? (boxesArr) => {
        let k = 0;
        for (let i = 0; i < b; i++) k = k * n + boxesArr[i];
        return k;
      }
    : (boxesArr) => boxesArr.join(",");
  room.encode = numeric
    ? (boxesArr, region) => room.boxesKey(boxesArr) * 64 + region
    : (boxesArr, region) => boxesArr.join(",") + "|" + region;
  room.decode = numeric
    ? (key) => {
        const region = key % 64;
        let bk = (key - region) / 64;
        const boxesArr = new Array(b);
        for (let i = b - 1; i >= 0; i--) {
          boxesArr[i] = bk % n;
          bk = (bk - boxesArr[i]) / n;
        }
        return { boxes: boxesArr, region };
      }
    : (key) => {
        const bar = key.indexOf("|");
        const boxesArr = key.slice(0, bar).split(",").map(Number);
        return { boxes: boxesArr, region: Number(key.slice(bar + 1)) };
      };
  return room;
}

// ------------------------------------------------------------- search -----

// Flood the player's reachable region from `start` (local cell), avoiding
// boxes. Fills `reach`, returns the region's smallest cell index (the
// canonical representative of the player's position).
function floodInto(room, start, boxSet, reach, stack) {
  reach.fill(0);
  let top = 0;
  stack[top++] = start;
  reach[start] = 1;
  let min = start;
  while (top > 0) {
    const cur = stack[--top];
    if (cur < min) min = cur;
    const base = cur * 4;
    for (let d = 0; d < 4; d++) {
      const nb = room.dest[base + d];
      if (nb < 0 || reach[nb] || boxSet[nb]) continue;
      reach[nb] = 1;
      stack[top++] = nb;
    }
  }
  return min;
}

// Admissible heuristic: the cheapest assignment of boxes to goals under
// the precomputed box-to-goal distances (exact min-matching up to 6 boxes,
// else a sum of per-box minima). Cached per box configuration.
function heuristic(room, boxes) {
  const bk = room.boxesKey(boxes);
  const cached = room.hCache.get(bk);
  if (cached !== undefined) return cached;
  const { b, dist, perms } = room;
  let h;
  if (perms) {
    h = INF;
    for (const perm of perms) {
      let s = 0;
      for (let i = 0; i < b; i++) {
        const d = dist[boxes[i] * b + perm[i]];
        if (d >= INF) {
          s = INF;
          break;
        }
        s += d;
      }
      if (s < h) h = s;
      if (h === 0) break;
    }
  } else {
    h = 0;
    for (let i = 0; i < b; i++) {
      let m = INF;
      for (let gi = 0; gi < b; gi++) {
        const d = dist[boxes[i] * b + gi];
        if (d < m) m = d;
      }
      if (m >= INF) {
        h = INF;
        break;
      }
      h += m;
    }
  }
  room.hCache.set(bk, h);
  return h;
}

function deadlock2x2(room, boxes, movedCell) {
  const { blocks, blocksByCell, goalSet } = room;
  const list = blocksByCell[movedCell];
  for (let i = 0; i < list.length; i++) {
    const cells = blocks[list[i]];
    let allBox = true;
    let anyNotGoal = false;
    for (let j = 0; j < cells.length; j++) {
      const c = cells[j];
      if (boxes.indexOf(c) < 0) {
        allBox = false;
        break;
      }
      if (!goalSet[c]) anyNotGoal = true;
    }
    if (allBox && anyNotGoal) return true;
  }
  return false;
}

function prio(g, h, opts) {
  const f = opts.greedy ? h : g + opts.weight * h;
  return f * 1024 - g;
}

// A* (weight 1), weighted A* or greedy best-first over push states.
// opts: { requireExit, weight, greedy }.
// Returns { status: "solved"|"exhausted"|"budget"|"error", ... }.
function searchRoom(room, opts, budget, shared) {
  const { n, domain, entryLocal, dest, goalSet, b, dead } = room;
  const boxSet = new Uint8Array(domain);
  const boxSet2 = new Uint8Array(domain);
  const reach = new Uint8Array(domain);
  const reach2 = new Uint8Array(domain);
  const stack = new Int32Array(domain + 4);

  const startBoxes = room.boxes.slice();
  for (const bx of startBoxes) boxSet[bx] = 1;
  const startRegion = floodInto(room, room.startPlayer, boxSet, reach, stack);
  const h0 = heuristic(room, startBoxes);
  if (h0 >= INF) return { status: "exhausted", expansions: 0 };
  const startKey = room.encode(startBoxes, startRegion);

  const gScore = new Map();
  const parent = new Map();
  const heap = new Heap();
  gScore.set(startKey, 0);
  parent.set(startKey, null);
  heap.push(startKey, prio(0, h0, opts), 0);

  let expansions = 0;
  while (heap.size > 0) {
    const [key, , g] = heap.pop();
    if (g !== gScore.get(key)) continue; // stale heap entry
    expansions++;
    shared.used++;
    if (expansions > budget || shared.used > shared.cap) {
      return { status: "budget", expansions };
    }
    const { boxes, region } = room.decode(key);
    boxSet.fill(0);
    for (let i = 0; i < b; i++) boxSet[boxes[i]] = 1;
    let solved = true;
    for (let i = 0; i < b; i++) {
      if (!goalSet[boxes[i]]) {
        solved = false;
        break;
      }
    }
    floodInto(room, region, boxSet, reach, stack);
    if (solved) {
      if (!opts.requireExit || reach[entryLocal]) {
        return finishSearch(room, parent, key, startKey, opts.requireExit, expansions);
      }
      // Solved but the player cannot get out: keep searching, another
      // route may solve it with an exit.
    }
    for (let bi = 0; bi < b; bi++) {
      const box = boxes[bi];
      const base = box * 4;
      for (let d = 0; d < 4; d++) {
        const target = dest[base + d];
        if (target < 0 || target >= n) continue; // boxes stay in the room
        if (boxSet[target]) continue;
        const pushPos = dest[base + OPP[d]];
        if (pushPos < 0 || !reach[pushPos]) continue;
        if (dead[target]) continue;
        const nb = boxes.slice();
        nb[bi] = target;
        nb.sort((x, y) => x - y);
        if (deadlock2x2(room, nb, target)) continue;
        const h = heuristic(room, nb);
        if (h >= INF) continue;
        boxSet2.fill(0);
        for (let i = 0; i < b; i++) boxSet2[nb[i]] = 1;
        const newRegion = floodInto(room, box, boxSet2, reach2, stack);
        const newKey = room.encode(nb, newRegion);
        const g2 = g + 1;
        const prev = gScore.get(newKey);
        if (prev !== undefined && prev <= g2) continue;
        gScore.set(newKey, g2);
        parent.set(newKey, room.numeric ? key * 256 + box * 4 + d : [key, box, d]);
        heap.push(newKey, prio(g2, h, opts), g2);
      }
    }
  }
  return { status: "exhausted", expansions };
}

function finishSearch(room, parent, goalKey, startKey, requireExit, expansions) {
  const pushes = [];
  let key = goalKey;
  while (key !== startKey) {
    const p = parent.get(key);
    let pk;
    let box;
    let dir;
    if (room.numeric) {
      dir = p % 4;
      box = Math.floor(p / 4) % 64;
      pk = Math.floor(p / 256);
    } else {
      pk = p[0];
      box = p[1];
      dir = p[2];
    }
    pushes.push([box, dir]);
    key = pk;
  }
  pushes.reverse();
  const built = buildMoves(room, pushes, requireExit);
  if (!built) return { status: "error", expansions };
  return {
    status: "solved",
    moves: built.moves,
    pushes: pushes.length,
    finalPlayer: built.finalPlayer,
    expansions,
  };
}

// Turn a list of pushes into the full LURD move string, filling in the
// player's walks between pushes with deterministic BFS paths.
function buildMoves(room, pushes, requireExit) {
  const { w } = room;
  const boxes = new Set(room.boxes.map((l) => room.globalOf[l]));
  let player = room.startGlobal;
  const parts = [];
  for (const [boxLocal, dir] of pushes) {
    const boxG = room.globalOf[boxLocal];
    const br = Math.floor(boxG / w);
    const bc = boxG % w;
    const pushPos = (br - DR[dir]) * w + (bc - DC[dir]);
    const path = bfsPath(room.level, player, pushPos, boxes, room.domainMask);
    if (path === null) return null;
    parts.push(path);
    parts.push(LETTER[dir]);
    boxes.delete(boxG);
    boxes.add((br + DR[dir]) * w + (bc + DC[dir]));
    player = boxG;
  }
  if (requireExit) {
    const path = bfsPath(room.level, player, room.entryGlobal, boxes, room.domainMask);
    if (path === null) return null;
    parts.push(path);
    player = room.entryGlobal;
  }
  return { moves: parts.join(""), finalPlayer: player };
}

// Deterministic BFS path (fixed neighbour order) from `from` to `to` over
// non-wall cells that are in `mask` (null: anywhere) and hold no box.
// Returns the move string, or null when unreachable.
function bfsPath(level, from, to, boxes, mask) {
  if (from === to) return "";
  const { w, h, wall } = level;
  const prev = new Int32Array(w * h).fill(-1);
  const prevDir = new Int8Array(w * h);
  const q = [from];
  prev[from] = from;
  let head = 0;
  while (head < q.length) {
    const cur = q[head++];
    if (cur === to) break;
    const r = Math.floor(cur / w);
    const c = cur % w;
    for (let d = 0; d < 4; d++) {
      const nr = r + DR[d];
      const nc = c + DC[d];
      if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
      const ni = nr * w + nc;
      if (prev[ni] !== -1 || wall[ni]) continue;
      if (mask && !mask[ni]) continue;
      if (boxes.has(ni)) continue;
      prev[ni] = cur;
      prevDir[ni] = d;
      q.push(ni);
    }
  }
  if (prev[to] === -1) return null;
  const letters = [];
  let cur = to;
  while (cur !== from) {
    letters.push(LETTER[prevDir[cur]]);
    cur = prev[cur];
  }
  letters.reverse();
  return letters.join("");
}

// ------------------------------------------------------------- verify -----

// Replay a move string on the level. Returns { ok, pushes }.
function verify(level, moves) {
  const { w, h, wall, goal } = level;
  const boxes = new Set(level.boxes);
  let player = level.player;
  let pushes = 0;
  for (let i = 0; i < moves.length; i++) {
    const d = "udlr".indexOf(moves[i]);
    if (d < 0) return { ok: false, pushes: 0 };
    const r = Math.floor(player / w);
    const c = player % w;
    const nr = r + DR[d];
    const nc = c + DC[d];
    if (nr < 0 || nr >= h || nc < 0 || nc >= w) return { ok: false, pushes: 0 };
    const ni = nr * w + nc;
    if (wall[ni]) return { ok: false, pushes: 0 };
    if (boxes.has(ni)) {
      const br = nr + DR[d];
      const bc = nc + DC[d];
      if (br < 0 || br >= h || bc < 0 || bc >= w) return { ok: false, pushes: 0 };
      const bi = br * w + bc;
      if (wall[bi] || boxes.has(bi)) return { ok: false, pushes: 0 };
      boxes.delete(ni);
      boxes.add(bi);
      pushes++;
    }
    player = ni;
  }
  for (const b of boxes) {
    if (!goal[b]) return { ok: false, pushes };
  }
  return { ok: true, pushes };
}

// -------------------------------------------------------------- solving ---

function attemptRoom(room, opts, nominalBudget, acct, shared) {
  const budget = Math.min(nominalBudget, acct.left);
  if (budget <= 0) return null;
  const res = searchRoom(room, opts, budget, shared);
  acct.left -= res.expansions;
  if (res.status !== "solved") return null;
  return { moves: res.moves, pushes: res.pushes, finalPlayer: res.finalPlayer };
}

function solveRooms(level, rooms, shared) {
  const solved = [];
  for (let i = 0; i < rooms.length; i++) {
    const comp = rooms[i];
    const room = makeRoom(level, comp, comp.entry, comp.entry);
    const acct = { left: ROOM_TOTAL_BUDGET };
    const exitSol =
      attemptRoom(room, { requireExit: true, weight: 1, greedy: false }, ASTAR_BUDGET, acct, shared) ||
      attemptRoom(room, { requireExit: true, weight: 2, greedy: false }, WEIGHTED_BUDGET, acct, shared) ||
      attemptRoom(room, { requireExit: true, weight: 1, greedy: true }, GREEDY_BUDGET, acct, shared);
    const freeSol =
      attemptRoom(room, { requireExit: false, weight: 1, greedy: false }, ASTAR_BUDGET, acct, shared) ||
      (!exitSol
        ? attemptRoom(room, { requireExit: false, weight: 2, greedy: false }, WEIGHTED_BUDGET, acct, shared)
        : null) ||
      (!exitSol
        ? attemptRoom(room, { requireExit: false, weight: 1, greedy: true }, GREEDY_BUDGET, acct, shared)
        : null);
    solved.push({ room, exitSol, freeSol });
  }

  // Choose the last room: a room with no exit solution must go last; else
  // the room where dropping the exit requirement saves the most pushes.
  let last = -1;
  let mustLast = -1;
  let mustCount = 0;
  for (let i = 0; i < solved.length; i++) {
    const s = solved[i];
    if (!s.exitSol && !s.freeSol) return null; // room unsolved
    if (!s.exitSol && s.freeSol) {
      mustCount++;
      mustLast = i;
    }
  }
  if (mustCount > 1) return null;
  if (mustCount === 1) {
    last = mustLast;
  } else {
    let bestSaving = -1;
    for (let i = 0; i < solved.length; i++) {
      const s = solved[i];
      const saving = s.freeSol ? s.exitSol.pushes - s.freeSol.pushes : 0;
      if (saving > bestSaving) {
        bestSaving = saving;
        last = i;
      }
    }
  }

  const boxes = new Set(level.boxes);
  let player = level.player;
  const parts = [];
  let totalPushes = 0;
  const order = [];
  for (let i = 0; i < solved.length; i++) if (i !== last) order.push(i);
  order.push(last);
  for (const idx of order) {
    const s = solved[idx];
    const comp = rooms[idx];
    const sol = idx === last ? s.freeSol || s.exitSol : s.exitSol;
    const walk = bfsPath(level, player, comp.entry, boxes, null);
    if (walk === null) return null;
    parts.push(walk);
    parts.push(sol.moves);
    totalPushes += sol.pushes;
    for (const b of comp.boxes) boxes.delete(b);
    for (const g of comp.goals) boxes.add(g);
    player = sol.finalPlayer;
  }
  const moves = parts.join("");
  const check = verify(level, moves);
  if (!check.ok || check.pushes !== totalPushes) return null;
  return { moves, pushes: totalPushes };
}

// Last resort for levels without the hallway-plus-rooms shape: a push
// search over the whole level, optimal A* first, then weighted A* and
// greedy best-first if the budget runs out.
function solveWholeLevel(level, shared) {
  const cells = [];
  const goals = [];
  for (let i = 0; i < level.w * level.h; i++) {
    if (level.wall[i]) continue;
    cells.push(i);
    if (level.goal[i]) goals.push(i);
  }
  if (level.boxes.length !== goals.length) return null;
  const room = makeRoom(level, { cells, boxes: level.boxes.slice(), goals }, -1, level.player);
  const acct = { left: FALLBACK_ASTAR_BUDGET + FALLBACK_WEIGHTED_BUDGET + FALLBACK_BUDGET };
  const sol =
    attemptRoom(room, { requireExit: false, weight: 1, greedy: false }, FALLBACK_ASTAR_BUDGET, acct, shared) ||
    attemptRoom(room, { requireExit: false, weight: 2, greedy: false }, FALLBACK_WEIGHTED_BUDGET, acct, shared) ||
    attemptRoom(room, { requireExit: false, weight: 1, greedy: true }, FALLBACK_BUDGET, acct, shared);
  if (!sol) return null;
  const check = verify(level, sol.moves);
  if (!check.ok) return null;
  return { moves: sol.moves, pushes: check.pushes };
}

function solveLevel(text, shared) {
  const level = parseLevel(text);
  if (level.player < 0) return { moves: "", pushes: 0, status: "no-player" };
  if (level.boxes.length === 0) return { moves: "", pushes: 0, status: "already-solved" };
  let solved0 = true;
  for (const b of level.boxes) {
    if (!level.goal[b]) {
      solved0 = false;
      break;
    }
  }
  if (solved0) return { moves: "", pushes: 0, status: "already-solved" };
  const pr = Math.floor(level.player / level.w);
  const pc = level.player % level.w;
  const rooms = tryDecompose(level, true, pr) || tryDecompose(level, false, pc);
  if (rooms) {
    const res = solveRooms(level, rooms, shared);
    if (res) return { moves: res.moves, pushes: res.pushes, status: "solved" };
  }
  const whole = solveWholeLevel(level, shared);
  if (whole) return { moves: whole.moves, pushes: whole.pushes, status: "solved-fallback" };
  return { moves: "", pushes: 0, status: "unsolved" };
}

// ---------------------------------------------------------------- main ----

function main() {
  // The harness mounts the instance files at /input and collects answers
  // from /output; its process driver stands those mounts in with the
  // MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR variables, so read them first.
  // Plain positional arguments override both, for local testing only.
  const inputDir = process.argv[2] || process.env.MAKEFUN_INPUT_DIR || "/input";
  const outputDir = process.argv[3] || process.env.MAKEFUN_OUTPUT_DIR || "/output";
  let ids = [];
  try {
    ids = fs
      .readdirSync(inputDir, { withFileTypes: true })
      .filter((e) => e.isDirectory() && /^i[0-9]+$/.test(e.name))
      .map((e) => e.name)
      .sort();
  } catch {
    ids = [];
  }
  if (ids.length === 0) ids = ["i001", "i002", "i003", "i004", "i005"];
  // Split the run-wide expansion budget evenly over the instances, so one
  // pathological instance cannot starve the rest.
  const perInstanceCap = Math.max(1, Math.floor(GLOBAL_BUDGET / ids.length));
  let totalUsed = 0;
  for (const id of ids) {
    const shared = { used: 0, cap: perInstanceCap };
    let moves = "";
    let status = "missing";
    let pushes = 0;
    try {
      const text = fs.readFileSync(path.join(inputDir, id, "level.txt"), "utf8");
      const res = solveLevel(text, shared);
      moves = res.moves;
      status = res.status;
      pushes = res.pushes;
    } catch {
      status = "error";
    }
    totalUsed += shared.used;
    try {
      const outDir = path.join(outputDir, id);
      fs.mkdirSync(outDir, { recursive: true });
      fs.writeFileSync(path.join(outDir, "solution.txt"), moves, "utf8");
    } catch {
      status = "write-error";
    }
    console.error(`${id}: ${status} pushes=${pushes} moves=${moves.length} expansions=${shared.used} total=${totalUsed}`);
  }
}

main();
