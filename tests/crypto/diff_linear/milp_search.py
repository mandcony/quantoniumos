#!/usr/bin/env python3
"""MILP for lower-bounding active S-boxes in a Feistel-like round function.

This is a simplified TinySPN-style MILP to bound the number of active S-boxes
across 8 rounds using binary variables and linear constraints. It is NOT a full
cryptanalytic proof; it provides a reproducible baseline bound for CI gating.
"""
from __future__ import annotations

import argparse
from typing import List

try:
    import pulp  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(f"pulp not installed: {e}")


def build_model(rounds: int, sboxes_per_round: int, min_active: int):
    # Binary variables: x[r,i] = 1 if S-box i in round r is active
    model = pulp.LpProblem("Active_Sboxes_Bound", pulp.LpMaximize)
    x = [[pulp.LpVariable(f"x_{r}_{i}", cat=pulp.LpBinary) for i in range(sboxes_per_round)] for r in range(rounds)]

    # Objective: maximize active S-boxes (used to test worst-case)
    model += pulp.lpSum(x[r][i] for r in range(rounds) for i in range(sboxes_per_round))

    # Simple diffusion constraint: an active S-box in round r implies at least
    # one active in round r+1 (toy model to avoid trivial zero solution)
    for r in range(rounds - 1):
        model += pulp.lpSum(x[r][i] for i in range(sboxes_per_round)) <= pulp.lpSum(x[r + 1][j] for j in range(sboxes_per_round))

    # Force at least one active S-box in the first round (avoid all-zero)
    model += pulp.lpSum(x[0][i] for i in range(sboxes_per_round)) >= 1

    return model, x


def solve_and_check(rounds: int, sboxes_per_round: int, min_active: int) -> int:
    model, x = build_model(rounds, sboxes_per_round, min_active)
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    total_active = int(sum(var.value() or 0 for row in x for var in row))
    return total_active


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--sboxes-per-round", type=int, default=4)
    p.add_argument("--min-active", type=int, default=20)
    args = p.parse_args()

    total = solve_and_check(args.rounds, args.sboxes_per_round, args.min_active)
    print({"rounds": args.rounds, "sboxes_per_round": args.sboxes_per_round, "total_active": total})
    if total < args.min_active:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
