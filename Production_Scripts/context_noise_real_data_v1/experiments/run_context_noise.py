"""Run the real-data context-noise experiment sweep.

This focused entry point keeps the context-noise sweep separate from the
multi-experiment runner in ``run_experiments.py`` while reusing the shared RMAB
instance generation, policy suite, CSV writing, and plotting helpers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from run_experiments import (
    _maybe_load_dfs,
    make_instance,
    plot_context_noise,
    run_policy_suite,
    write_group_summary,
    write_rows,
)


DEFAULT_CONTEXT_NOISE_VARIANTS = [
    "dense",
    "offline",
    "gated_offline",
    "support_offline",
]

DEFAULT_POLICIES = [
    "state_thompson",
    "local_ucb_tw",
    "global_ucb_tw",
    "exp4",
    "tw",
    "tm_tw",
    "tm_tw_refined",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/research/rmab_vm_outputs/context_noise_real_data_v1")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to directory containing datacenter_*_with_metrics.csv files.",
    )
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--state-grid", type=int, nargs="+", default=[8, 20, 50, 100])
    parser.add_argument("--noise-grid", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    parser.add_argument("--sparsity", type=int, default=2)
    parser.add_argument("--transition-dominance", type=float, default=0.45)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Transition-learning variants. Defaults exclude low-rank variants.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=DEFAULT_POLICIES,
        help="Policy filter for the sweep.",
    )
    parser.add_argument("--trust-scale-mults", type=float, nargs="+", default=[1.0])
    parser.add_argument("--gate-scale-mults", type=float, nargs="+", default=[1.0])
    parser.add_argument("--gate-modes", choices=["deterministic", "beta"], nargs="+", default=["deterministic"])
    parser.add_argument("--beta-gate-concentration", type=float, default=20.0)
    parser.add_argument("--trust-floors", type=float, nargs="+", default=[0.10])
    parser.add_argument("--trust-cap", type=float, default=0.95)
    parser.add_argument("--flush-every", type=int, default=0)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def run_context_noise_experiment(args: argparse.Namespace) -> None:
    rows: list[dict] = []
    variants = args.variants or DEFAULT_CONTEXT_NOISE_VARIANTS
    datacenter_dfs = _maybe_load_dfs(args)
    for state_count in args.state_grid:
        for noise_level in args.noise_grid:
            for trust_scale_mult in args.trust_scale_mults:
                for gate_scale_mult in args.gate_scale_mults:
                    for gate_mode in args.gate_modes:
                        for trust_floor in args.trust_floors:
                            for seed in range(args.seeds):
                                instance = make_instance(
                                    seed=args.seed + seed + 31 * state_count,
                                    n_states=state_count,
                                    sparsity=args.sparsity,
                                    transition_dominance=args.transition_dominance,
                                    datacenter_dfs=datacenter_dfs,
                                )
                                result = run_policy_suite(
                                    instance,
                                    seed=args.seed + seed,
                                    rounds=args.rounds,
                                    noise_level=noise_level,
                                    include_masked=True,
                                    variants=variants,
                                    policies_filter=set(args.policies) if args.policies else None,
                                    trust_scale_mult=trust_scale_mult,
                                    gate_scale_mult=gate_scale_mult,
                                    gate_mode=gate_mode,
                                    beta_gate_concentration=args.beta_gate_concentration,
                                    trust_floor=trust_floor,
                                    trust_cap=args.trust_cap,
                                )
                                for row in result.rows:
                                    row["experiment_id"] = "context_noise_refinement"
                                    rows.append(row)
                                if args.flush_every > 0 and len(rows) % args.flush_every == 0:
                                    out_dir = Path(args.output)
                                    write_rows(out_dir / "context_noise_results.partial.csv", rows)
                                if args.progress:
                                    print(
                                        "finished",
                                        f"S={state_count}",
                                        f"noise={noise_level}",
                                        f"trust={trust_scale_mult}",
                                        f"gate={gate_scale_mult}",
                                        f"mode={gate_mode}",
                                        f"floor={trust_floor}",
                                        f"seed={seed}",
                                        flush=True,
                                    )

    out_dir = Path(args.output)
    write_rows(out_dir / "context_noise_results.csv", rows)
    write_group_summary(
        rows,
        [
            "S",
            "context_noise_level",
            "trust_scale_mult",
            "gate_scale_mult",
            "gate_mode",
            "trust_floor",
            "policy_label",
        ],
        out_dir / "context_noise_summary.csv",
    )
    plot_context_noise(rows, out_dir)


def main() -> None:
    run_context_noise_experiment(parse_args())


if __name__ == "__main__":
    main()
