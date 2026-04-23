"""
Analyzer module for Azul MARL experiment results.

Reads experiment CSV files produced by SimulationManager and generates
statistical analysis and publication-quality visualisations covering:
  1. Win rate matrix with confidence intervals
  2. Score distributions (histograms, box plots, summary statistics)
  3. First-player advantage (binomial tests)
  4. Luck vs skill analysis (variance decomposition, Cohen's d)
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def _simplify_agent_name(raw: str) -> str:
    """Shorten verbose agent descriptors for display.

    Examples:
        'RandomAgent(seed=42)'            -> 'RandomAgent'
        'MCTSAgent(sims=200, depth=60)'   -> 'MCTSAgent(200)'
        'MinimaxAgent(depth=1)'           -> 'MinimaxAgent'
        'RandomAgentHierarchical(seed=99)'-> 'RandomAgentHier.'
    """
    # Extract base name
    base = raw.split("(")[0]
    # Abbreviate long names
    if base == "RandomAgentHierarchical":
        base = "RandomAgentHier."

    # For MCTS, keep sim count
    m = re.search(r"sims=(\d+)", raw)
    if m:
        return f"{base}({m.group(1)})"
    # For Minimax, keep depth if > 1
    m = re.search(r"depth=(\d+)", raw)
    if m and int(m.group(1)) > 1:
        return f"{base}(d={m.group(1)})"
    return base


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = wins / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


class Analyzer:
    """Statistical analyzer for Azul MARL experiment results."""

    def __init__(
        self,
        results_dir: str | Path = "results",
        figures_dir: str | Path | None = None,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir) if figures_dir else self.results_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.data = self._load_all_csvs()
        # Build simplified agent names
        self.data["agent1_short"] = self.data["agent1_type"].map(_simplify_agent_name)
        self.data["agent2_short"] = self.data["agent2_type"].map(_simplify_agent_name)

    # ------------------------------------------------------------------ #
    #  Data loading                                                       #
    # ------------------------------------------------------------------ #

    def _load_all_csvs(self) -> pd.DataFrame:
        """Load and concatenate every CSV in *results_dir*."""
        frames: list[pd.DataFrame] = []
        for path in sorted(self.results_dir.glob("*.csv")):
            df = pd.read_csv(path)
            df["source_file"] = path.stem
            frames.append(df)
        if not frames:
            raise FileNotFoundError(
                f"No CSV files found in {self.results_dir}"
            )
        return pd.concat(frames, ignore_index=True)

    def _matchup_label(self, row: pd.Series) -> str:
        return f"{row['agent1_short']} vs {row['agent2_short']}"

    # ------------------------------------------------------------------ #
    #  1. Win‑rate matrix                                                 #
    # ------------------------------------------------------------------ #

    def win_rate_matrix(self) -> pd.DataFrame:
        """Compute and display win‑rate matrix with Wilson CIs.

        Returns a DataFrame with the matrix values.
        """
        records: list[dict] = []

        for (a1, a2), grp in self.data.groupby(
            ["agent1_short", "agent2_short"]
        ):
            n = len(grp)
            a1_wins = (grp["winner"] == "agent1").sum()
            a2_wins = (grp["winner"] == "agent2").sum()
            ties = (grp["winner"] == "tie").sum()
            wr1 = a1_wins / n if n else 0
            ci1 = _wilson_ci(a1_wins, n)
            records.append(
                dict(
                    agent1=a1,
                    agent2=a2,
                    n_games=n,
                    agent1_wins=a1_wins,
                    agent2_wins=a2_wins,
                    ties=ties,
                    agent1_wr=wr1,
                    ci_low=ci1[0],
                    ci_high=ci1[1],
                )
            )

        result_df = pd.DataFrame(records)

        # ---- Print table ----
        print("\n" + "=" * 70)
        print("WIN RATE MATRIX (agent1 win rate)")
        print("=" * 70)
        for _, r in result_df.iterrows():
            print(
                f"  {r['agent1']:>22s} vs {r['agent2']:<22s}  "
                f"WR={r['agent1_wr']:.1%}  "
                f"95%CI=[{r['ci_low']:.1%}, {r['ci_high']:.1%}]  "
                f"(W{int(r['agent1_wins'])}/L{int(r['agent2_wins'])}/T{int(r['ties'])} "
                f"n={int(r['n_games'])})"
            )
        print()

        # ---- Heatmap ----
        agents = sorted(
            set(result_df["agent1"].tolist() + result_df["agent2"].tolist())
        )
        matrix = pd.DataFrame(np.nan, index=agents, columns=agents)
        annot = pd.DataFrame("", index=agents, columns=agents)
        for _, r in result_df.iterrows():
            wr = r["agent1_wr"]
            matrix.loc[r["agent1"], r["agent2"]] = wr
            annot.loc[r["agent1"], r["agent2"]] = (
                f"{wr:.0%}\n({int(r['n_games'])}g)"
            )

        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = plt.cm.RdYlGn
        im = ax.imshow(
            matrix.values.astype(float), cmap=cmap, vmin=0, vmax=1, aspect="auto"
        )
        ax.set_xticks(range(len(agents)))
        ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(agents)))
        ax.set_yticklabels(agents, fontsize=9)
        ax.set_xlabel("Agent 2 (opponent)", fontsize=10)
        ax.set_ylabel("Agent 1 (row player)", fontsize=10)
        ax.set_title("Win Rate Matrix (Agent 1 win rate)", fontsize=12, pad=12)

        # Annotate cells
        for i in range(len(agents)):
            for j in range(len(agents)):
                txt = annot.iloc[i, j]
                if txt:
                    val = matrix.iloc[i, j]
                    color = "white" if (val > 0.75 or val < 0.25) else "black"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

        fig.colorbar(im, ax=ax, shrink=0.8, label="Win rate")
        fig.tight_layout()
        fig.savefig(self.figures_dir / "win_rate_heatmap.png", dpi=150)
        plt.close(fig)
        print(f"  [saved] {self.figures_dir / 'win_rate_heatmap.png'}")

        return result_df

    # ------------------------------------------------------------------ #
    #  2. Score distributions                                             #
    # ------------------------------------------------------------------ #

    def score_distributions(self) -> pd.DataFrame:
        """Histograms, box plots, and summary statistics of scores."""
        matchups = self.data.groupby("source_file")

        # ---- Summary stats ----
        stats_rows: list[dict] = []
        for src, grp in matchups:
            a1 = grp["agent1_short"].iloc[0]
            a2 = grp["agent2_short"].iloc[0]
            for agent_label, col in [(a1, "agent1_score"), (a2, "agent2_score")]:
                scores = grp[col]
                stats_rows.append(
                    dict(
                        matchup=src,
                        agent=agent_label,
                        mean=scores.mean(),
                        median=scores.median(),
                        std=scores.std(),
                        min=scores.min(),
                        max=scores.max(),
                        n=len(scores),
                    )
                )
        stats_df = pd.DataFrame(stats_rows)

        print("\n" + "=" * 70)
        print("SCORE DISTRIBUTIONS — Summary Statistics")
        print("=" * 70)
        print(stats_df.to_string(index=False, float_format="{:.2f}".format))
        print()

        # ---- Overlaid histograms per matchup, laid out in a portrait-friendly grid ----
        matchup_groups = list(self.data.groupby("source_file"))
        n_matchups = len(matchup_groups)
        ncols = min(3, n_matchups)
        nrows = math.ceil(n_matchups / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False
        )
        for idx, (src, grp) in enumerate(matchup_groups):
            ax = axes[idx // ncols, idx % ncols]
            a1 = grp["agent1_short"].iloc[0]
            a2 = grp["agent2_short"].iloc[0]

            all_scores = pd.concat([grp["agent1_score"], grp["agent2_score"]])
            lo, hi = all_scores.min(), all_scores.max()
            bins = np.linspace(lo - 0.5, hi + 0.5, min(30, hi - lo + 2))

            ax.hist(
                grp["agent1_score"], bins=bins, alpha=0.6, label=a1, edgecolor="black", linewidth=0.5
            )
            ax.hist(
                grp["agent2_score"], bins=bins, alpha=0.6, label=a2, edgecolor="black", linewidth=0.5
            )
            ax.set_title(f"{a1} vs {a2}", fontsize=10)
            ax.set_xlabel("Score", fontsize=9)
            ax.set_ylabel("Frequency", fontsize=9)
            ax.legend(fontsize=8)
        for idx in range(n_matchups, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)
        fig.suptitle("Score Distributions by Matchup", fontsize=12, y=1.0)
        fig.tight_layout()
        fig.savefig(self.figures_dir / "score_histograms.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [saved] {self.figures_dir / 'score_histograms.png'}")

        # ---- Box plots, oriented horizontally so matchup labels run down the y-axis ----
        box_data: list[dict] = []
        for src, grp in self.data.groupby("source_file"):
            a1 = grp["agent1_short"].iloc[0]
            a2 = grp["agent2_short"].iloc[0]
            box_data.append(
                dict(label=f"{a1} (vs {a2})", scores=grp["agent1_score"].values)
            )
            box_data.append(
                dict(label=f"{a2} (vs {a1})", scores=grp["agent2_score"].values)
            )

        fig, ax = plt.subplots(figsize=(8, max(5, 0.35 * len(box_data))))
        bp = ax.boxplot(
            [d["scores"] for d in box_data],
            tick_labels=[d["label"] for d in box_data],
            patch_artist=True,
            vert=False,
        )
        colors = plt.cm.tab10(np.linspace(0, 1, len(box_data)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xlabel("Score", fontsize=10)
        ax.set_title("Score Distributions (Box Plots)", fontsize=12)
        ax.tick_params(axis="y", labelsize=8)
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(self.figures_dir / "score_boxplots.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [saved] {self.figures_dir / 'score_boxplots.png'}")

        return stats_df

    # ------------------------------------------------------------------ #
    #  3. First‑player advantage                                          #
    # ------------------------------------------------------------------ #

    def first_player_advantage(self) -> pd.DataFrame:
        """Binomial test for first‑player advantage per matchup."""
        records: list[dict] = []

        for src, grp in self.data.groupby("source_file"):
            a1 = grp["agent1_short"].iloc[0]
            a2 = grp["agent2_short"].iloc[0]
            label = f"{a1} vs {a2}"

            # Determine if first player won, excluding ties
            no_tie = grp[grp["winner"] != "tie"].copy()
            n = len(no_tie)
            if n == 0:
                continue

            first_wins = (
                (no_tie["starting_player"] == no_tie["winner"])
            ).sum()

            first_wr = first_wins / n
            # Two-sided binomial test: H0: p = 0.5
            btest = stats.binomtest(first_wins, n, 0.5, alternative="two-sided")
            p_value = btest.pvalue

            # Effect size (difference from 0.5)
            effect = first_wr - 0.5

            records.append(
                dict(
                    matchup=label,
                    source=src,
                    n_decisive=n,
                    first_wins=first_wins,
                    first_wr=first_wr,
                    p_value=p_value,
                    effect=effect,
                )
            )

        result_df = pd.DataFrame(records)

        print("\n" + "=" * 70)
        print("FIRST-PLAYER ADVANTAGE")
        print("=" * 70)
        for _, r in result_df.iterrows():
            sig = "***" if r["p_value"] < 0.001 else (
                "**" if r["p_value"] < 0.01 else (
                    "*" if r["p_value"] < 0.05 else "n.s."
                )
            )
            print(
                f"  {r['matchup']:<40s}  "
                f"1st-WR={r['first_wr']:.1%}  "
                f"effect={r['effect']:+.1%}  "
                f"p={r['p_value']:.4f} {sig}  "
                f"(n={int(r['n_decisive'])})"
            )
        print()

        # ---- Bar chart ----
        fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(result_df)), 5))
        x = np.arange(len(result_df))
        bars = ax.bar(x, result_df["first_wr"] * 100, color="steelblue", edgecolor="black", linewidth=0.5)
        ax.axhline(50, color="red", linestyle="--", linewidth=1, label="No advantage (50%)")

        # Color significant bars differently
        for i, (_, r) in enumerate(result_df.iterrows()):
            if r["p_value"] < 0.05:
                bars[i].set_color("darkorange")

        ax.set_xticks(x)
        ax.set_xticklabels(result_df["matchup"], rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("First Player Win Rate (%)", fontsize=10)
        ax.set_title("First-Player Advantage by Matchup", fontsize=12)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(self.figures_dir / "first_player_advantage.png", dpi=150)
        plt.close(fig)
        print(f"  [saved] {self.figures_dir / 'first_player_advantage.png'}")

        return result_df

    # ------------------------------------------------------------------ #
    #  4. Luck vs Skill analysis                                          #
    # ------------------------------------------------------------------ #

    def luck_vs_skill(self) -> pd.DataFrame:
        """Variance decomposition and Cohen's d per matchup."""
        records: list[dict] = []

        for src, grp in self.data.groupby("source_file"):
            a1 = grp["agent1_short"].iloc[0]
            a2 = grp["agent2_short"].iloc[0]
            label = f"{a1} vs {a2}"

            s1 = grp["agent1_score"].values.astype(float)
            s2 = grp["agent2_score"].values.astype(float)

            # Within-matchup variance (average of both agents)
            var1 = np.var(s1, ddof=1) if len(s1) > 1 else 0.0
            var2 = np.var(s2, ddof=1) if len(s2) > 1 else 0.0
            within_var = (var1 + var2) / 2

            # Between-agent variance (of means)
            between_var = np.var([s1.mean(), s2.mean()], ddof=0)

            # Cohen's d
            pooled_std = np.sqrt(
                ((len(s1) - 1) * var1 + (len(s2) - 1) * var2)
                / (len(s1) + len(s2) - 2)
            ) if (len(s1) + len(s2)) > 2 else 1.0
            cohens_d = (s1.mean() - s2.mean()) / pooled_std if pooled_std > 0 else 0.0

            # Win rate consistency: std of per-game winner indicator
            a1_win_indicator = (grp["winner"] == "agent1").astype(float).values
            wr_std = np.std(a1_win_indicator, ddof=1) if len(a1_win_indicator) > 1 else 0.0

            # Skill ratio: between / (between + within), bounded
            total_var = between_var + within_var
            skill_ratio = between_var / total_var if total_var > 0 else 0.0

            records.append(
                dict(
                    matchup=label,
                    source=src,
                    within_var=within_var,
                    between_var=between_var,
                    skill_ratio=skill_ratio,
                    cohens_d=cohens_d,
                    wr_consistency_std=wr_std,
                    agent1_score_std=np.sqrt(var1),
                    agent2_score_std=np.sqrt(var2),
                )
            )

        result_df = pd.DataFrame(records)

        print("\n" + "=" * 70)
        print("LUCK vs SKILL ANALYSIS")
        print("=" * 70)
        for _, r in result_df.iterrows():
            d_interp = (
                "negligible" if abs(r["cohens_d"]) < 0.2 else
                "small" if abs(r["cohens_d"]) < 0.5 else
                "medium" if abs(r["cohens_d"]) < 0.8 else
                "large"
            )
            print(f"\n  {r['matchup']}")
            print(f"    Within-matchup variance : {r['within_var']:.2f}")
            print(f"    Between-agent variance  : {r['between_var']:.2f}")
            print(f"    Skill ratio (B/(B+W))   : {r['skill_ratio']:.3f}")
            print(f"    Cohen's d               : {r['cohens_d']:+.3f} ({d_interp})")
            print(f"    WR consistency (std)     : {r['wr_consistency_std']:.3f}")
        print()

        # ---- Grouped bar chart: within-var vs between-var ----
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: variance decomposition
        ax = axes[0]
        x = np.arange(len(result_df))
        w = 0.35
        ax.bar(x - w / 2, result_df["within_var"], w, label="Within (luck)", color="cornflowerblue", edgecolor="black", linewidth=0.5)
        ax.bar(x + w / 2, result_df["between_var"], w, label="Between (skill)", color="salmon", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(result_df["matchup"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Variance", fontsize=10)
        ax.set_title("Score Variance Decomposition", fontsize=11)
        ax.legend(fontsize=9)

        # Right: Cohen's d
        ax = axes[1]
        colors = ["forestgreen" if abs(d) >= 0.8 else "goldenrod" if abs(d) >= 0.5 else "lightgrey"
                  for d in result_df["cohens_d"]]
        ax.barh(x, result_df["cohens_d"], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(x)
        ax.set_yticklabels(result_df["matchup"], fontsize=8)
        ax.set_xlabel("Cohen's d", fontsize=10)
        ax.set_title("Effect Size (Cohen's d)", fontsize=11)
        ax.axvline(0, color="black", linewidth=0.5)
        # Reference lines
        for val, lbl in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
            ax.axvline(val, color="grey", linestyle=":", linewidth=0.5, alpha=0.7)
            ax.axvline(-val, color="grey", linestyle=":", linewidth=0.5, alpha=0.7)

        fig.suptitle("Luck vs Skill Analysis", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(self.figures_dir / "luck_vs_skill.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [saved] {self.figures_dir / 'luck_vs_skill.png'}")

        return result_df

    # ------------------------------------------------------------------ #
    #  Full report                                                        #
    # ------------------------------------------------------------------ #

    def generate_full_report(self) -> Path:
        """Run all four analyses and write a summary text file."""
        print("\n" + "#" * 70)
        print("#  AZUL MARL — FULL ANALYSIS REPORT")
        print("#" * 70)
        print(f"  Results directory : {self.results_dir}")
        print(f"  Figures directory : {self.figures_dir}")
        print(f"  Total games loaded: {len(self.data)}")
        print(f"  Source files      : {self.data['source_file'].nunique()}")

        wr_df = self.win_rate_matrix()
        sd_df = self.score_distributions()
        fp_df = self.first_player_advantage()
        ls_df = self.luck_vs_skill()

        # Write summary text
        report_path = self.figures_dir / "analysis_report.txt"
        with open(report_path, "w") as f:
            f.write("AZUL MARL — ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total games: {len(self.data)}\n")
            f.write(f"Source files: {self.data['source_file'].nunique()}\n\n")

            f.write("WIN RATES\n")
            f.write("-" * 50 + "\n")
            f.write(wr_df.to_string(index=False))
            f.write("\n\n")

            f.write("SCORE STATISTICS\n")
            f.write("-" * 50 + "\n")
            f.write(sd_df.to_string(index=False))
            f.write("\n\n")

            f.write("FIRST-PLAYER ADVANTAGE\n")
            f.write("-" * 50 + "\n")
            f.write(fp_df.to_string(index=False))
            f.write("\n\n")

            f.write("LUCK vs SKILL\n")
            f.write("-" * 50 + "\n")
            f.write(ls_df.to_string(index=False))
            f.write("\n")

        print(f"\n  [saved] {report_path}")
        return report_path
