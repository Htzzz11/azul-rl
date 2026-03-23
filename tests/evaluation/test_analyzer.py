"""Tests for evaluation.analyzer.Analyzer.

Verifies that the Analyzer loads the existing experiment CSVs and that
each analysis method runs without errors, producing expected output types
and saving figure files.
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'evaluation' package is importable
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd
import pytest

from evaluation.analyzer import Analyzer


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
FIGURES_DIR = RESULTS_DIR / "figures_test"


@pytest.fixture(scope="module")
def analyzer():
    """Create a shared Analyzer instance for all tests."""
    return Analyzer(results_dir=RESULTS_DIR, figures_dir=FIGURES_DIR)


class TestAnalyzerLoading:
    def test_loads_csvs(self, analyzer):
        assert len(analyzer.data) > 0

    def test_has_expected_columns(self, analyzer):
        for col in [
            "agent1_type", "agent2_type", "agent1_score", "agent2_score",
            "winner", "starting_player", "source_file",
        ]:
            assert col in analyzer.data.columns

    def test_loads_three_source_files(self, analyzer):
        assert analyzer.data["source_file"].nunique() >= 3


class TestWinRateMatrix:
    def test_returns_dataframe(self, analyzer):
        result = analyzer.win_rate_matrix()
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 3

    def test_win_rates_bounded(self, analyzer):
        result = analyzer.win_rate_matrix()
        assert (result["agent1_wr"] >= 0).all()
        assert (result["agent1_wr"] <= 1).all()

    def test_confidence_intervals(self, analyzer):
        result = analyzer.win_rate_matrix()
        # Use small tolerance for floating point edge cases (e.g. 100% win rate)
        eps = 1e-9
        assert ((result["ci_low"] - eps) <= result["agent1_wr"]).all()
        assert ((result["ci_high"] + eps) >= result["agent1_wr"]).all()

    def test_heatmap_saved(self, analyzer):
        analyzer.win_rate_matrix()
        assert (FIGURES_DIR / "win_rate_heatmap.png").exists()


class TestScoreDistributions:
    def test_returns_stats_dataframe(self, analyzer):
        result = analyzer.score_distributions()
        assert isinstance(result, pd.DataFrame)
        assert "mean" in result.columns
        assert "std" in result.columns

    def test_figures_saved(self, analyzer):
        analyzer.score_distributions()
        assert (FIGURES_DIR / "score_histograms.png").exists()
        assert (FIGURES_DIR / "score_boxplots.png").exists()


class TestFirstPlayerAdvantage:
    def test_returns_dataframe(self, analyzer):
        result = analyzer.first_player_advantage()
        assert isinstance(result, pd.DataFrame)
        assert "p_value" in result.columns
        assert "first_wr" in result.columns

    def test_p_values_valid(self, analyzer):
        result = analyzer.first_player_advantage()
        assert (result["p_value"] >= 0).all()
        assert (result["p_value"] <= 1).all()

    def test_figure_saved(self, analyzer):
        analyzer.first_player_advantage()
        assert (FIGURES_DIR / "first_player_advantage.png").exists()


class TestLuckVsSkill:
    def test_returns_dataframe(self, analyzer):
        result = analyzer.luck_vs_skill()
        assert isinstance(result, pd.DataFrame)
        assert "cohens_d" in result.columns
        assert "skill_ratio" in result.columns

    def test_skill_ratio_bounded(self, analyzer):
        result = analyzer.luck_vs_skill()
        assert (result["skill_ratio"] >= 0).all()
        assert (result["skill_ratio"] <= 1).all()

    def test_figure_saved(self, analyzer):
        analyzer.luck_vs_skill()
        assert (FIGURES_DIR / "luck_vs_skill.png").exists()


class TestFullReport:
    def test_generates_report_file(self, analyzer):
        path = analyzer.generate_full_report()
        assert path.exists()
        content = path.read_text()
        assert "WIN RATES" in content
        assert "SCORE STATISTICS" in content


@pytest.fixture(scope="module", autouse=True)
def cleanup(request):
    """Remove test figures directory after all tests."""
    def _cleanup():
        import shutil
        if FIGURES_DIR.exists():
            shutil.rmtree(FIGURES_DIR)
    request.addfinalizer(_cleanup)
