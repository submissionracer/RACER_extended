"""Run late-peak ERCOT refinement stages and plot their reward waterfall."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

WATERFALL_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "st_tw_tmtw_comparison"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_local_global_tw_oracle_tw_n40 as experiment  # noqa: E402


STAGES = [
    {
        "stage": "TW",
        "policy": "TW only",
        "use_job_scheduling_graph_prior": True,
        "job_graph_path_weight_mode": "equal_uniform",
    },
    {
        "stage": "Adaptive trust",
        "policy": "Local + Global + TW",
        "use_job_scheduling_graph_prior": True,
        "job_graph_path_weight_mode": "equal_uniform",
    },
    {
        "stage": "Graph-based prior",
        "policy": "Local + Global + TW",
        "use_job_scheduling_graph_prior": True,
        "job_graph_path_weight_mode": "reduction",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WATERFALL_DIR,
    )
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--n-jobs", type=int, default=40)
    parser.add_argument("--delay-weight", type=float, default=1.0)
    parser.add_argument("--reward-prior-mean", type=float, default=-0.1)
    parser.add_argument("--adaptive-mix-warmup-rounds", type=int, default=40)
    parser.add_argument("--adaptive-mix-trust-max", type=float, default=0.95)
    parser.add_argument("--adaptive-score-mode", choices=["thompson", "greedy"], default="greedy")
    parser.add_argument("--mix-whittle-guard", action="store_true")
    parser.add_argument("--job-graph-lookahead-batches", type=int, default=3)
    parser.add_argument("--job-graph-corehour-cap", type=float, default=95.0)
    parser.add_argument("--ercot-time", default="19:00")
    parser.add_argument(
        "--datacenter-dir",
        default="datasets/mit_gpu_datacenter_with_metrics",
        help="Directory containing datacenter_*_with_metrics.csv workload files.",
    )
    return parser.parse_args()


def make_stage_args(base_args: argparse.Namespace, stage: dict[str, object]) -> argparse.Namespace:
    args = argparse.Namespace(**vars(base_args))
    args.output_dir = str(base_args.output_dir / str(stage["stage"]).lower().replace("\n", "_").replace(" ", "_"))
    args.budget_map = "5:2"
    args.use_job_scheduling_graph_prior = bool(stage["use_job_scheduling_graph_prior"])
    args.job_graph_path_weight_mode = str(stage["job_graph_path_weight_mode"])
    args.job_graph_equal_path_weight = False
    args.mix_warmup_rounds = base_args.adaptive_mix_warmup_rounds
    args.mix_trust_max = base_args.adaptive_mix_trust_max
    args.mix_reward_score_mode = base_args.adaptive_score_mode
    args.mix_whittle_guard = base_args.mix_whittle_guard
    args.mix_reward_conf_scale = experiment.Config().mix_reward_conf_scale
    args.mix_trans_conf_scale = experiment.Config().mix_trans_conf_scale
    args.vi_max_iters = 60
    args.binary_iters = 6
    args.replan_interval = 50
    args.exp4_initial_weight_mode = "uniform"
    args.exp4_initial_weight_scale = 0.0
    return args


def run_stage(base_args: argparse.Namespace, stage: dict[str, object]) -> dict[str, object]:
    stage_args = make_stage_args(base_args, stage)
    stage_dir = Path(stage_args.output_dir).resolve()
    stage_dir.mkdir(parents=True, exist_ok=True)

    summary_df, round_df, ercot_meta = experiment.run_ercot_timepoint(
        stage_args,
        label="late_afternoon_peak",
        time_col=base_args.ercot_time,
    )
    summary_df.to_csv(stage_dir / "local_global_tw_oracle_tw_n40_summary.csv", index=False)
    round_df.to_csv(stage_dir / "local_global_tw_oracle_tw_n40_round_rewards.csv", index=False)
    ercot_meta.to_csv(stage_dir / "ercot_lmp_metadata.csv", index=False)
    experiment.plot_results(summary_df, str(stage_dir))
    experiment.plot_running_average_rewards(round_df, str(stage_dir))
    experiment.write_comparison_table(summary_df, str(stage_dir))

    policy_summary = summary_df[summary_df["policy"] == stage["policy"]]
    oracle_summary = summary_df[summary_df["policy"] == "Oracle Whittle"]
    per_seed = policy_summary.set_index("seed")["avg_reward"].to_dict()
    return {
        "stage": stage["stage"],
        "policy": stage["policy"],
        "avg_reward": float(policy_summary["avg_reward"].mean()),
        "sem_reward": float(policy_summary["avg_reward"].sem()),
        "oracle_reward": float(oracle_summary["avg_reward"].mean()),
        "per_seed_reward": {int(k): float(v) for k, v in per_seed.items()},
        "output_dir": str(stage_dir.relative_to(REPO_ROOT)),
        "job_graph_prior_enabled": stage["use_job_scheduling_graph_prior"],
        "job_graph_path_weight_mode": stage["job_graph_path_weight_mode"],
    }


def plot_waterfall(stage_df: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update(
        {
            "axes.titlesize": 30,
            "axes.labelsize": 30,
            "xtick.labelsize": 26,
            "ytick.labelsize": 28,
        }
    )
    labels = stage_df["stage"].tolist()
    rewards = stage_df["avg_reward"].to_numpy(dtype=float)
    errors = stage_df["sem_reward"].fillna(0.0).to_numpy(dtype=float)
    deltas = np.diff(rewards, prepend=rewards[0])

    has_oracle = "oracle_reward" in stage_df.columns and stage_df["oracle_reward"].notna().all()
    if has_oracle:
        oracle = stage_df["oracle_reward"].to_numpy(dtype=float)
        oracle_ref = float(np.mean(oracle))

    has_impr = {"improvement", "improvement_sem"}.issubset(stage_df.columns)
    if has_impr:
        impr = stage_df["improvement"].to_numpy(dtype=float)
        impr_sem = stage_df["improvement_sem"].fillna(0.0).to_numpy(dtype=float)

    base_color = "#4C78A8"
    gain_color = "#2F8F46"
    loss_color = "#C43C39"
    final_color = "#6A4C93"

    fig, ax = plt.subplots(figsize=(16.0, 10.0), constrained_layout=True)
    colors = [base_color] + [gain_color if value >= 0 else loss_color for value in deltas[1:]]
    colors[-1] = final_color

    def _mean_dot(idx: int) -> None:
        ax.plot(
            idx,
            rewards[idx],
            marker="o",
            markersize=12,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=2.0,
            linestyle="none",
            zorder=5,
        )

    ax.bar(0, rewards[0], color=colors[0], width=0.58)
    _mean_dot(0)

    for i in range(1, len(rewards)):
        previous = rewards[i - 1]
        delta = rewards[i] - previous
        bottom = min(previous, rewards[i])
        ax.bar(i, abs(delta), bottom=bottom, color=colors[i], width=0.58)
        ax.plot([i - 1 + 0.29, i - 0.29], [previous, previous], color="0.45", linewidth=1.0)
        _mean_dot(i)
        if has_impr:
            impr_text = f"avg improvement\n{impr[i]:+.3f} +/- {impr_sem[i]:.3f}"
        else:
            impr_text = f"avg improvement\n{delta:+.3f}"
        ax.text(
            i,
            rewards[i],
            impr_text,
            ha="center",
            va="bottom" if delta >= 0 else "top",
            fontsize=21,
        )

    legend_handles = [Patch(facecolor=base_color, label="Baseline stage (absolute mean reward)")]
    if np.any(deltas[1:-1] >= 0):
        legend_handles.append(Patch(facecolor=gain_color, label="Improvement vs. previous stage"))
    if np.any(deltas[1:-1] < 0):
        legend_handles.append(Patch(facecolor=loss_color, label="Regression vs. previous stage"))
    legend_handles.append(Patch(facecolor=final_color, label="Final stage"))
    ax.legend(handles=legend_handles, loc="upper left", fontsize=19, framealpha=0.9)

    ax.set_xticks(np.arange(len(labels)), labels=[lab.replace(" ", "\n") for lab in labels])
    ax.tick_params(axis="x", rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")
    ax.set_ylabel(r"Average late-peak reward (\$/MWh)", fontsize=30)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    span = float(np.ptp(rewards))
    y_lo = min(rewards) - max(span * 0.35, 0.25)
    y_hi = max(rewards) + max(span * 0.55, 0.45)
    if has_oracle:
        y_hi = max(y_hi, oracle_ref + max(span * 0.35, 0.25))
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(-0.75, len(labels) - 1 + 0.75)

    if has_oracle:
        secax = ax.secondary_yaxis(
            "right",
            functions=(lambda r: 100.0 * r / oracle_ref, lambda p: p * oracle_ref / 100.0),
        )
        secax.set_ylabel("% of Oracle-Whittle reward", fontsize=30)
        secax.tick_params(labelsize=28)
        secax.yaxis.set_major_formatter(FormatStrFormatter("%.1f%%"))
        ax.axhline(oracle_ref, linestyle=":", color="0.35", linewidth=1.6, zorder=1)
        ax.text(
            0.0,
            oracle_ref,
            "Oracle-Whittle (100%)",
            ha="left",
            va="bottom",
            fontsize=18,
            color="0.35",
        )

    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def _add_improvement_stats(rows: list[dict[str, object]]) -> None:
    """Paired stage-over-stage improvement (mean and SEM across shared seeds)."""
    for i, row in enumerate(rows):
        if i == 0:
            row["improvement"] = 0.0
            row["improvement_sem"] = 0.0
            continue
        prev = rows[i - 1].get("per_seed_reward", {})
        cur = row.get("per_seed_reward", {})
        seeds = sorted(set(prev) & set(cur))
        diffs = np.array([cur[s] - prev[s] for s in seeds], dtype=float)
        row["improvement"] = float(diffs.mean()) if diffs.size else float("nan")
        row["improvement_sem"] = (
            float(diffs.std(ddof=1) / np.sqrt(diffs.size)) if diffs.size > 1 else 0.0
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_stage(args, stage) for stage in STAGES]
    _add_improvement_stats(rows)
    stage_df = pd.DataFrame(rows).drop(columns=["per_seed_reward"])
    csv_path = args.output_dir / "late_peak_refinement_waterfall.csv"
    png_path = args.output_dir / "late_peak_refinement_waterfall.png"
    stage_df.to_csv(csv_path, index=False)
    plot_waterfall(stage_df, png_path)
    print(stage_df.to_string(index=False))
    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
