#!/usr/bin/env python3
"""MILP model to lower-bound active S-boxes across rounds (approximate but structured).

State per round (16 bytes through S-box layer):
  state0 -> MDS1 -> state1 -> SBOX -> state2 -> MDS2 -> state3 -> next round state0

Constraints:
  - SBOX is bijective at byte granularity: activity in == activity out
  - MDS1/MDS2 use branch-number B=5 (AES-like 4x4 MDS):
      For each 4-byte column: sum_in + sum_out >= 5 if any input active
  - Feistel bridging: if any byte active after MDS2, at least one byte is active entering next round

Objective:
  - Minimize total number of active S-boxes across all rounds (sum of state1)
  - The minimum provides a lower bound; CI enforces this bound ≥ threshold
"""
from __future__ import annotations

import argparse
from typing import List

try:
    import pulp  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(f"pulp not installed: {e}")


def build_model(rounds: int) -> tuple[pulp.LpProblem, List[List[pulp.LpVariable]]]:
    # Binary byte-activity variables for each stage
    model = pulp.LpProblem("Active_Sboxes_LowerBound", pulp.LpMinimize)
    # 16 bytes per round through S-box stage
    state0 = [[pulp.LpVariable(f"s0_{r}_{i}", cat=pulp.LpBinary) for i in range(16)] for r in range(rounds)]
    state1 = [[pulp.LpVariable(f"s1_{r}_{i}", cat=pulp.LpBinary) for i in range(16)] for r in range(rounds)]
    state2 = [[pulp.LpVariable(f"s2_{r}_{i}", cat=pulp.LpBinary) for i in range(16)] for r in range(rounds)]
    state3 = [[pulp.LpVariable(f"s3_{r}_{i}", cat=pulp.LpBinary) for i in range(16)] for r in range(rounds)]

    # Objective: minimize total active S-boxes (equals active state1 bytes)
    model += pulp.lpSum(state1[r][i] for r in range(rounds) for i in range(16))

    # Column indices for 4x4 MDS (AES-style layout: columns of 4 bytes)
    cols = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]

    for r in range(rounds):
        # MDS1: state0 -> state1 with branch number 5
        for c, col in enumerate(cols):
            t = pulp.LpVariable(f"t1_{r}_{c}", cat=pulp.LpBinary)
            sum_in = pulp.lpSum(state0[r][i] for i in col)
            sum_out = pulp.lpSum(state1[r][i] for i in col)
            model += sum_in - 4 * t <= 0
            model += sum_in - t >= 0
            model += sum_in + sum_out >= 5 * t

        # S-box bijection: activity preserved bytewise
        for i in range(16):
            model += state2[r][i] - state1[r][i] == 0

        # MDS2: state2 -> state3 with branch number 5
        for c, col in enumerate(cols):
            t = pulp.LpVariable(f"t2_{r}_{c}", cat=pulp.LpBinary)
            sum_in = pulp.lpSum(state2[r][i] for i in col)
            sum_out = pulp.lpSum(state3[r][i] for i in col)
            model += sum_in - 4 * t <= 0
            model += sum_in - t >= 0
            model += sum_in + sum_out >= 5 * t

        # Bridge to next round: if any active after MDS2, at least one active next input
        if r < rounds - 1:
            u = pulp.LpVariable(f"u_{r}", cat=pulp.LpBinary)
            sum_out = pulp.lpSum(state3[r][i] for i in range(16))
            model += sum_out - 16 * u <= 0
            model += sum_out - u >= 0
            model += pulp.lpSum(state0[r + 1][i] for i in range(16)) >= u

    # Initial condition: at least one active byte to avoid trivial zero
    model += pulp.lpSum(state0[0][i] for i in range(16)) >= 1

    # Return model and the S-box activity variables for reporting (state1)
    return model, state1


def solve_and_check(rounds: int, min_active: int) -> int:
    model, state1 = build_model(rounds)
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    total_active = int(sum((v.value() or 0) for r in state1 for v in r))
    return total_active


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--min-active", type=int, default=32)
    args = p.parse_args()

    total = solve_and_check(args.rounds, args.min_active)
    print({"rounds": args.rounds, "sboxes_per_round": args.sboxes_per_round, "total_active": total})
    if total < args.min_active:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
