import unittest

from model_engine import (
    Calibration,
    calibrate_models,
    empirical_bracket_probability,
    choose_temperature_strategy,
    weighted_member_distribution,
)


class ModelEngineTests(unittest.TestCase):
    def test_empirical_temperature_buckets_use_half_open_bounds(self):
        members = [(23.999, 0.2), (24.0, 0.3), (24.999, 0.2), (25.0, 0.3)]
        self.assertAlmostEqual(empirical_bracket_probability("24°C", members), 0.5)
        self.assertAlmostEqual(empirical_bracket_probability("25°C or higher", members), 0.3)

    def test_walk_forward_only_approves_a_materially_better_blend(self):
        obs = {}
        forecasts = {name: {} for name in ("ECMWF", "GFS", "ICON")}
        for index in range(50):
            day = f"2026-01-{index + 1:02d}"
            actual = 22.0 + (index % 7) * 0.3
            obs[day] = actual
            forecasts["ECMWF"][day] = actual + (2.0 if index % 2 else -2.0)
            forecasts["GFS"][day] = actual + (0.2 if index % 3 else -0.2)
            forecasts["ICON"][day] = actual + (0.3 if index % 4 else -0.3)
        result = calibrate_models(obs, forecasts, 2)
        self.assertTrue(result.enabled)
        self.assertLess(result.walkforward_blend_rmse, result.walkforward_ecmwf_rmse * 0.98)
        self.assertAlmostEqual(sum(result.weights.values()), 1.0)

    def test_insufficient_history_disables_multimodel(self):
        days = [f"2026-01-{index + 1:02d}" for index in range(10)]
        obs = {day: 20.0 for day in days}
        forecasts = {name: {day: 20.0 for day in days} for name in ("ECMWF", "GFS", "ICON")}
        result = calibrate_models(obs, forecasts, 1)
        self.assertFalse(result.enabled)
        self.assertIn("mínimo", result.reason)

    def test_unapproved_blend_falls_back_to_ecmwf_members(self):
        calibration = Calibration(False, 1, 10, {}, {}, {}, {}, None, None, "insuficiente")
        distribution = weighted_member_distribution(
            {"ECMWF": [24.0, 25.0], "GFS": [30.0], "ICON": [31.0]}, calibration
        )
        self.assertEqual(distribution, [(24.0, 0.5), (25.0, 0.5)])

    def test_strategy_can_recommend_adjacent_temperature_basket(self):
        markets = [
            {"groupItemTitle": "24°C", "bestAsk": "0.20"},
            {"groupItemTitle": "25°C", "bestAsk": "0.20"},
        ]
        distribution = [(24.2, 0.5), (25.2, 0.5)]
        strategy = choose_temperature_strategy(markets, distribution, lambda price, market: 0.0)
        self.assertEqual(strategy["kind"], "cesta de 2 temperaturas")
        self.assertEqual(strategy["probability"], 1.0)
        self.assertAlmostEqual(strategy["ev"], 0.6)


if __name__ == "__main__":
    unittest.main()
