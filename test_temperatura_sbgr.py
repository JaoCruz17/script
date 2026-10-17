from contextlib import redirect_stdout
from datetime import date, datetime, timezone
from io import StringIO
import unittest
from unittest.mock import call, patch

import requests

import temperatura_sbgr


class DashboardTests(unittest.TestCase):
    def render_market(self, market, orderbook_side_effect, mu=None, sigma=None):
        output = StringIO()
        with (
            patch.object(temperatura_sbgr, "fetch_metars", return_value=[]),
            patch.object(temperatura_sbgr, "fetch_ladder", return_value=[market]),
            patch.object(
                temperatura_sbgr,
                "fetch_orderbook",
                side_effect=orderbook_side_effect,
            ) as fetch_orderbook,
            redirect_stdout(output),
        ):
            temperatura_sbgr.print_dashboard(date(2026, 10, 10), mu, sigma)
        return output.getvalue(), fetch_orderbook

    def test_dashboard_omits_state_column(self):
        now = datetime.now(timezone.utc).timestamp()
        market = {
            "groupItemTitle": "24°C",
            "outcomes": ["Yes", "No"],
            "clobTokenIds": ["yes-token", "no-token"],
        }
        book = {
            "bids": [{"price": "0.20", "size": "20"}],
            "asks": [{"price": "0.30", "size": "20"}],
            "timestamp": str(int(now * 1000)),
        }

        output, _ = self.render_market(market, [book, book])

        lines = output.splitlines()
        value_header = next(
            line for line in lines if line.startswith("|") and "P(YES)" in line
        )
        self.assertNotIn("Estado", value_header)
        self.assertIn("EV% NO", value_header)
        quote_header = next(line for line in lines if "Ask NO" in line)
        self.assertIn("Slippage NO", quote_header)
        self.assertTrue(all(len(line) <= 120 for line in lines if line.startswith("|")))
        self.assertIn("P(YES), Edge e EV suspensos", output)

    def test_dashboard_prints_each_table_as_a_contiguous_block(self):
        markets = [
            {"groupItemTitle": "24°C", "outcomes": [], "clobTokenIds": []},
            {"groupItemTitle": "25°C", "outcomes": [], "clobTokenIds": []},
        ]
        output = StringIO()
        with (
            patch.object(temperatura_sbgr, "fetch_metars", return_value=[]),
            patch.object(temperatura_sbgr, "fetch_ladder", return_value=markets),
            redirect_stdout(output),
        ):
            temperatura_sbgr.print_dashboard(date(2026, 10, 10), None, None)

        lines = output.getvalue().splitlines()
        second_quote_row = next(index for index, line in enumerate(lines) if line.startswith("| 25°C"))
        valuation_heading = lines.index("[PROBABILIDADES E VALOR]")
        second_value_row = next(
            index for index, line in enumerate(lines)
            if index > valuation_heading and line.startswith("| 25°C")
        )
        self.assertLess(second_quote_row, valuation_heading)
        self.assertGreater(second_value_row, valuation_heading)

    def test_probability_and_ev_values_render_when_market_rules_are_confirmed(self):
        market = {
            "groupItemTitle": "24°C",
            "outcomes": ["Yes", "No"],
            "clobTokenIds": ["yes-token", "no-token"],
            "feesEnabled": True,
            "feeSchedule": {"rate": 0, "exponent": 1},
        }
        book = {
            "bids": [{"price": "0.20", "size": "20"}],
            "asks": [{"price": "0.30", "size": "20"}],
        }
        output = StringIO()
        with (
            patch.object(temperatura_sbgr, "fetch_metars", return_value=[]),
            patch.object(temperatura_sbgr, "fetch_ladder", return_value=[market]),
            patch.object(temperatura_sbgr, "fetch_orderbook", side_effect=[book, book]),
            patch.object(temperatura_sbgr, "load_resolution_config", return_value={"bucket_mode": "interval_start"}),
            patch.object(temperatura_sbgr, "resolution_status", return_value=(True, "confirmada")),
            patch.object(temperatura_sbgr, "validate_ladder", return_value={"complete": True, "reason": "válida"}),
            redirect_stdout(output),
        ):
            temperatura_sbgr.print_dashboard(date(2026, 10, 10), 24.5, 1.0)

        valuation_row = next(
            line for line in output.getvalue().splitlines()
            if line.startswith("| 24°C") and "%" in line
        )
        self.assertNotIn("—", valuation_row)
        self.assertIn("EV YES", output.getvalue())
        self.assertIn("EV NO", output.getvalue())

    def test_yes_failure_does_not_hide_no_book_data(self):
        market = {
            "groupItemTitle": "24°C",
            "outcomes": ["Yes", "No"],
            "clobTokenIds": ["yes-token", "no-token"],
        }
        no_book = {
            "bids": [{"price": "0.20", "size": "20"}],
            "asks": [{"price": "0.30", "size": "20"}],
        }
        error = requests.ConnectionError("YES book unavailable")

        output, fetch_orderbook = self.render_market(market, [error, no_book])

        fetch_orderbook.assert_has_calls([call("yes-token"), call("no-token")])
        self.assertIn("Ask NO", output)
        self.assertIn("0.300", output)
        self.assertIn("YES indisponível: YES book unavailable", output)
        self.assertNotIn("Erro ao consultar ladder", output)

    def test_no_failure_does_not_hide_yes_book_data(self):
        market = {
            "groupItemTitle": "24°C",
            "outcomes": ["Yes", "No"],
            "clobTokenIds": ["yes-token", "no-token"],
        }
        yes_book = {
            "bids": [{"price": "0.20", "size": "20"}],
            "asks": [{"price": "0.30", "size": "20"}],
        }
        error = requests.ConnectionError("NO book unavailable")

        output, fetch_orderbook = self.render_market(market, [yes_book, error])

        fetch_orderbook.assert_has_calls([call("yes-token"), call("no-token")])
        self.assertIn("NO indisponível: NO book unavailable", output)
        self.assertIn("Ask YES", output)
        self.assertNotIn("Erro ao consultar ladder", output)

    def test_missing_outcome_token_does_not_abort_ladder_row(self):
        market = {
            "groupItemTitle": "24°C",
            "outcomes": ["Yes"],
            "clobTokenIds": ["yes-token"],
        }
        yes_book = {
            "bids": [{"price": "0.20", "size": "20"}],
            "asks": [{"price": "0.30", "size": "20"}],
        }

        output, _ = self.render_market(market, [yes_book])

        self.assertIn("Faixa", output)
        self.assertIn("token NO ausente no mercado", output)
        self.assertNotIn("Erro ao consultar ladder", output)


if __name__ == "__main__":
    unittest.main()
