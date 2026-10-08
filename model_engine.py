"""SBGR model calibration and multi-model ensemble utilities.

Historical calibration uses fixed-lead ensemble-mean forecasts from Open-Meteo
Previous Runs and decoded SBGR METAR observations from Iowa Environmental
Mesonet. A blend is enabled only when a chronological walk-forward check beats
a bias-corrected ECMWF-only baseline.
"""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass
from datetime import datetime, time as datetime_time, timedelta, timezone
from itertools import product
from typing import Any
from zoneinfo import ZoneInfo

import requests


TZ_NAME = "America/Sao_Paulo"
try:
    TZ = ZoneInfo(TZ_NAME)
except Exception:  # Windows Python may lack the system IANA timezone database.
    TZ = timezone(timedelta(hours=-3), name="BRT")
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
MODEL_CONFIG = {
    "ECMWF": {"live": "ecmwf_ifs025_ensemble", "history": "ecmwf_ifs025_ensemble_mean"},
    "GFS": {"live": "ncep_gefs025", "history": "ncep_gefs025_ensemble_mean"},
    "ICON": {"live": "dwd_icon_global_eps", "history": "dwd_icon_eps_ensemble_mean"},
}
DEFAULT_MODEL_WEIGHTS = {"ECMWF": 0.50, "GFS": 0.30, "ICON": 0.20}
MIN_PAIRED_DAYS = 30
MIN_TRAIN_DAYS = 15
HISTORY_DAYS = 180


@dataclass
class Calibration:
    enabled: bool
    lead_days: int
    sample_days: int
    weights: dict[str, float]
    biases: dict[str, float]
    mae: dict[str, float]
    rmse: dict[str, float]
    walkforward_ecmwf_rmse: float | None
    walkforward_blend_rmse: float | None
    reason: str


def fetch_station_daily_max(days: int = HISTORY_DAYS) -> dict[str, float]:
    """Get local-day maximum air temperature from SBGR decoded METAR archive."""
    today = datetime.now(TZ).date()
    end_day = today - timedelta(days=1)
    start_day = end_day - timedelta(days=days - 1)
    start = datetime.combine(start_day, datetime_time.min, TZ)
    end = datetime.combine(today, datetime_time.min, TZ)
    response = requests.get(
        IEM_ASOS_URL,
        params={
            "station": "SBGR",
            "data": "tmpf",
            "sts": start.isoformat(),
            "ets": end.isoformat(),
            "tz": TZ_NAME,
            "format": "onlycomma",
            "missing": "empty",
        },
        headers={"User-Agent": "SBGR-temperature-ladder/1.0"},
        timeout=20,
    )
    response.raise_for_status()
    maxima: dict[str, float] = {}
    reader = csv.DictReader(io.StringIO(response.text))
    for row in reader:
        try:
            local_day = str(row["valid"])[:10]
            temp_f = float(row["tmpf"])
            if math.isfinite(temp_f) and -100 < temp_f < 150:
                temp_c = (temp_f - 32.0) * (5.0 / 9.0)
                maxima[local_day] = max(maxima.get(local_day, -math.inf), temp_c)
        except (KeyError, TypeError, ValueError):
            continue
    return maxima


def fetch_previous_run_daily_max(model: str, lead_days: int,
                                 days: int = HISTORY_DAYS) -> dict[str, float]:
    """Derive daily forecast maxima from fixed-lead hourly model forecasts."""
    if not 0 <= lead_days <= 7:
        raise ValueError("o arquivo de runs anteriores cobre horizontes D-0 a D-7")
    response = requests.get(
        PREVIOUS_RUNS_URL,
        params={
            "latitude": -23.4356,
            "longitude": -46.4731,
            "hourly": f"temperature_2m_previous_day{lead_days}",
            "models": model,
            "past_days": days,
            "forecast_days": 1,
            "timezone": TZ_NAME,
            "temperature_unit": "celsius",
        },
        headers={"User-Agent": "SBGR-temperature-ladder/1.0"},
        timeout=20,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    values = hourly.get(f"temperature_2m_previous_day{lead_days}") or []
    maxima: dict[str, float] = {}
    for stamp, value in zip(times, values):
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            local_day = str(stamp)[:10]
            maxima[local_day] = max(maxima.get(local_day, -math.inf), float(value))
    today = datetime.now(TZ).date().isoformat()
    return {day: value for day, value in maxima.items() if day < today}


def _fit_weights(rows: list[tuple[float, float, float, float]],
                 model_order: tuple[str, str, str]) -> tuple[dict[str, float], dict[str, float]]:
    """Fit bias corrections and nonnegative weights on prior dates only."""
    biases = {
        name: sum(row[index] - row[3] for row in rows) / len(rows)
        for index, name in enumerate(model_order)
    }
    corrected = [
        (row[0] - biases[model_order[0]], row[1] - biases[model_order[1]],
         row[2] - biases[model_order[2]], row[3])
        for row in rows
    ]
    # Search a small constrained simplex. Mild shrinkage prevents tiny samples
    # from assigning nearly all weight to a single model.
    candidates = [i / 20 for i in range(2, 17)]  # each model: 0.10 to 0.80
    best_weights = (1 / 3, 1 / 3, 1 / 3)
    best_loss = math.inf
    for a, b in product(candidates, repeat=2):
        c = round(1.0 - a - b, 10)
        if c < 0.10 or c > 0.80:
            continue
        weights = (a, b, c)
        mse = sum(
            (sum(weights[i] * row[i] for i in range(3)) - row[3]) ** 2
            for row in corrected
        ) / len(corrected)
        shrink = 0.02 * sum((weight - 1 / 3) ** 2 for weight in weights)
        if mse + shrink < best_loss:
            best_loss, best_weights = mse + shrink, weights
    return dict(zip(model_order, best_weights)), biases


def _rmse(actual: list[float], predicted: list[float]) -> float:
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))


def calibrate_models(
    observations: dict[str, float],
    forecasts: dict[str, dict[str, float]],
    lead_days: int,
    model_order: tuple[str, str, str] = ("ECMWF", "GFS", "ICON"),
) -> Calibration:
    """Fit a blend only if a chronological walk-forward beats ECMWF baseline."""
    common_days = sorted(
        set(observations).intersection(*(set(forecasts.get(name, {})) for name in model_order))
    )
    paired = [
        (forecasts[model_order[0]][day], forecasts[model_order[1]][day],
         forecasts[model_order[2]][day], observations[day])
        for day in common_days
    ]
    metrics_mae = {}
    metrics_rmse = {}
    full_bias = {}
    for index, name in enumerate(model_order):
        errors = [row[index] - row[3] for row in paired]
        if errors:
            metrics_mae[name] = sum(abs(error) for error in errors) / len(errors)
            metrics_rmse[name] = math.sqrt(sum(error * error for error in errors) / len(errors))
            full_bias[name] = sum(errors) / len(errors)

    unavailable = Calibration(False, lead_days, len(paired), {}, full_bias,
                               metrics_mae, metrics_rmse, None, None,
                               "histórico comum insuficiente")
    if len(paired) < MIN_PAIRED_DAYS:
        unavailable.reason = f"apenas {len(paired)} dias comuns; mínimo {MIN_PAIRED_DAYS}"
        return unavailable

    actual_oos: list[float] = []
    ecmwf_oos: list[float] = []
    blend_oos: list[float] = []
    for index in range(MIN_TRAIN_DAYS, len(paired)):
        train = paired[:index]
        test = paired[index]
        _, train_bias = _fit_weights(train, model_order)
        weights, _ = _fit_weights(train, model_order)
        corrected_test = [test[i] - train_bias[model_order[i]] for i in range(3)]
        actual_oos.append(test[3])
        ecmwf_oos.append(corrected_test[0])
        blend_oos.append(sum(weights[name] * corrected_test[i]
                             for i, name in enumerate(model_order)))
    baseline_rmse = _rmse(actual_oos, ecmwf_oos)
    blend_rmse = _rmse(actual_oos, blend_oos)
    weights, biases = _fit_weights(paired, model_order)
    if blend_rmse >= baseline_rmse * 0.98:
        unavailable.weights = weights
        unavailable.biases = biases
        unavailable.walkforward_ecmwf_rmse = baseline_rmse
        unavailable.walkforward_blend_rmse = blend_rmse
        unavailable.reason = "mistura não superou o ECMWF no walk-forward (exige ganho ≥2%)"
        return unavailable
    return Calibration(True, lead_days, len(paired), weights, biases,
                       metrics_mae, metrics_rmse, baseline_rmse, blend_rmse,
                       "mistura aprovada no walk-forward")


def weighted_member_distribution(
    member_sets: dict[str, list[float]], calibration: Calibration | None = None,
    model_order: tuple[str, str, str] = ("ECMWF", "GFS", "ICON"),
    configured_weights: dict[str, float] | None = None,
) -> list[tuple[float, float]]:
    """Return a normalized weighted empirical distribution over available models."""
    # Backward-compatible behavior for callers using the former calibration API.
    if calibration is not None and not calibration.enabled and configured_weights is None:
        values = member_sets.get("ECMWF", [])
        if not values:
            return []
        return [(value, 1.0 / len(values)) for value in values]
    weights = configured_weights or DEFAULT_MODEL_WEIGHTS
    available = {name: member_sets.get(name, []) for name in model_order
                 if member_sets.get(name)}
    total = sum(max(0.0, weights.get(name, 0.0)) for name in available)
    if not available:
        return []
    if total <= 0:
        weights = {name: 1.0 / len(available) for name in available}
        total = 1.0
    distribution: list[tuple[float, float]] = []
    for name in model_order:
        values = available.get(name, [])
        weight = max(0.0, weights.get(name, 0.0)) / total if values else 0.0
        if not values or weight <= 0:
            continue
        per_member = weight / len(values)
        for value in values:
            distribution.append((value, per_member))
    return distribution


def weighted_quantile(distribution: list[tuple[float, float]], quantile: float) -> float | None:
    """Weighted empirical quantile for the forecast range, not a mean confidence interval."""
    if not distribution or not 0 <= quantile <= 1:
        return None
    ordered = sorted((value, weight) for value, weight in distribution if weight > 0)
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        return None
    threshold = quantile * total
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def model_consensus(model_means: dict[str, float]) -> dict[str, Any]:
    """Heuristic agreement indicator; it is not a probability of forecast accuracy."""
    means = list(model_means.items())
    if len(means) < 2:
        return {"spread": None, "score": None, "level": "insuficiente", "zones": []}
    spread = max(value for _, value in means) - min(value for _, value in means)
    pairs = [(f"{means[i][0]} × {means[j][0]}", abs(means[i][1] - means[j][1]))
             for i in range(len(means)) for j in range(i + 1, len(means))]
    zones = [f"{pair} ({delta:.1f}°C)" for pair, delta in pairs if delta >= 1.0]
    score = 100 * sum(math.exp(-delta / 2.0) for _, delta in pairs) / len(pairs)
    level = "alta" if spread <= 1 else "moderada" if spread <= 2 else "baixa"
    return {"spread": spread, "score": score, "level": level, "zones": zones}


def summarize_distribution(distribution: list[tuple[float, float]]) -> tuple[float, float]:
    total_weight = sum(weight for _, weight in distribution)
    if not distribution or total_weight <= 0:
        raise ValueError("distribuição multimodelo vazia")
    normalized = [(value, weight / total_weight) for value, weight in distribution]
    mean = sum(value * weight for value, weight in normalized)
    variance = sum(weight * (value - mean) ** 2 for value, weight in normalized)
    return mean, math.sqrt(variance)


def empirical_bracket_probability(label: str, distribution: list[tuple[float, float]],
                                  bucket_mode: str = "interval_start") -> float | None:
    """Probability for a market bracket, with exact half-open Celsius bounds."""
    import re

    text = (label or "").strip().lower().replace("°", "")
    numbers = [int(number) for number in re.findall(r"-?\d+", text)]
    if not numbers or not distribution:
        return None
    lower = upper = None
    if any(token in text for token in ("below", "or less", "or lower", "ou menos", "ou abaixo")):
        upper = numbers[0] + (0.5 if bucket_mode == "nearest_degree" else 1)
    elif any(token in text for token in ("higher", "or more", "or above", "ou mais", "ou acima")):
        lower = numbers[0] - (0.5 if bucket_mode == "nearest_degree" else 0)
    elif len(numbers) >= 2:
        low, high = sorted(numbers[:2])
        lower, upper = (low - 0.5, high + 0.5) if bucket_mode == "nearest_degree" else (low, high + 1)
    else:
        lower, upper = (numbers[0] - 0.5, numbers[0] + 0.5) if bucket_mode == "nearest_degree" else (numbers[0], numbers[0] + 1)
    total = sum(weight for _, weight in distribution)
    if total <= 0:
        return None
    included = sum(
        weight for value, weight in distribution
        if (lower is None or value >= lower) and (upper is None or value < upper)
    )
    return included / total


def choose_temperature_strategy(markets: list[dict[str, Any]],
                                distribution: list[tuple[float, float]],
                                fee_for_price: Any,
                                bucket_mode: str = "interval_start") -> dict[str, Any] | None:
    """Choose the best positive-net-EV single bucket or adjacent two-bucket basket."""
    priced: list[dict[str, Any]] = []
    for market in markets:
        label = str(market.get("groupItemTitle") or market.get("question") or "")
        probability = empirical_bracket_probability(label, distribution, bucket_mode)
        try:
            ask = float(market.get("bestAsk"))
            if not math.isfinite(ask) or not 0 < ask <= 1:
                continue
        except (TypeError, ValueError):
            continue
        fee = fee_for_price(ask, market)
        if fee is None:
            continue
        cost = ask + fee
        priced.append({"label": label, "p": probability, "ask": ask,
                       "fee": fee, "cost": cost, "ev": probability - cost})

    strategies: list[dict[str, Any]] = []
    for item in priced:
        roi = item["ev"] / item["cost"] if item["cost"] > 0 else -math.inf
        strategies.append({"kind": "temperatura única", "items": [item],
                           "probability": item["p"], "cost": item["cost"],
                           "ev": item["ev"], "roi": roi})
    # Restrict baskets to adjacent integer temperature categories.
    buckets: list[tuple[int, dict[str, Any]]] = []
    for item in priced:
        if any(token in item["label"].lower() for token in ("below", "above", "or less", "or more", "ou menos", "ou mais")):
            continue
        match = re.search(r"(-?\d+)", item["label"])
        if match:
            buckets.append((int(match.group(1)), item))
    by_degree = {degree: item for degree, item in buckets}
    for degree, item in by_degree.items():
        neighbor = by_degree.get(degree + 1)
        if neighbor is None:
            continue
        cost = item["cost"] + neighbor["cost"]
        ev = item["p"] + neighbor["p"] - cost
        strategies.append({"kind": "cesta de 2 temperaturas", "items": [item, neighbor],
                           "probability": item["p"] + neighbor["p"], "cost": cost,
                           "ev": ev, "roi": ev / cost if cost > 0 else -math.inf})
    positive = [strategy for strategy in strategies if strategy["ev"] > 0]
    if not positive:
        return None
    # Prefer the better net return per dollar, then the larger expected value.
    return max(positive, key=lambda strategy: (strategy["roi"], strategy["ev"]))

