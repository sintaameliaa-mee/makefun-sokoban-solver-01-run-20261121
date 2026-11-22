# Sokoban Solver — make.fun sokoban-solver-01-run-20261121

[![sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/sintaameliaa-mee/makefun-sokoban-solver-01-run-20261121/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/sintaameliaa-mee/makefun-sokoban-solver-01-run-20261121/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)

A push-optimal Sokoban solver in a single file, `main.ts` (plain JavaScript,
run by Node 22 as `node /bot/main.ts`). It reads every instance of a set from
`/input/<id>/level.txt` and writes `/output/<id>/solution.txt`.

How it works: each level is a hallway plus one to four independent rooms, so
the solver splits the level into rooms, solves each room push-optimally with
A* over push states (min-matching box-to-goal heuristic, dead-square analysis,
2x2-block deadlock pruning, compact numeric state codes), requires every room
but the last to leave the player able to walk back out through its door, and
replays the full move string to verify it before writing it out. Every search
is bounded by a fixed expansion budget — never by the clock — so two runs on
the same input write identical bytes.
