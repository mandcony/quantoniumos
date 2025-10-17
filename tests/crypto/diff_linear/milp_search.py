#!/usr/bin/env python3
"""MILP stub for differential/linear trail bounds.
Currently returns success if invoked; replace with real MILP constraints.
"""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--min-active", type=int, default=32)
    args = p.parse_args()
    print(f"MILP stub: rounds={args.rounds} min_active={args.min_active}")


if __name__ == "__main__":
    main()
