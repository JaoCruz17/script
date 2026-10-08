"""Walk-forward probabilistic verification and market-rule safeguards."""

from __future__ import annotations

import math
from itertools import product
from typing import Any


def _weighted_forecast(row: dict[str, float], weights: dict[str, float],
                      biases: dict[str, float] | None = None) -> float | None:
    usable = [(name, value) for name, value in row.items()
              if name in weights and weights[name] > 0 and math.isfinite(value)]
    total = sum(weights[name] for name, _ in usable)
    if total <= 0:
        return None
    biases = biases or {}
    return sum(weights[name] * (value - biases.get(name, 0.0))
               for name, value in usable) / total


def _crps(scenarios: list[float], actual: float) -> float:
    if not scenarios:
        return math.nan
    first = sum(abs(value - actual) for value in scenarios) / len(scenarios)
    # O(n log n) equivalent of half the mean pairwise absolute difference.
    ordered = sorted(scenarios)
    n = len(ordered)
    pair_sum = sum((2 * index - n + 1) * value for index, value in enumerate(ordered))
    return first - pair_sum / (n * n)


def _verification(actuals: list[float], predictions: list[float],
                  residual_sets: list[list[float]]) -> dict[str, float | int]:
    if not actuals:
        return {"n": 0}
    all_values = [math.floor(value) for value in actuals]
    for forecast, errors in zip(predictions, residual_sets):
        all_values.extend(math.floor(forecast + error) for error in errors)
    low, high = min(all_values), max(all_values)
    # Reserve two tail categories to capture scenario values outside the range.
    labels = list(range(low, high + 1))
    brier_scores: list[float] = []
    log_losses: list[float] = []
    crps_scores: list[float] = []
    confidence_hits: list[tuple[float, float]] = []
    covered = 0
    for actual, forecast, errors in zip(actuals, predictions, residual_sets):
        scenarios = [forecast + error for error in errors]
        raw = [sum(1 for value in scenarios if math.floor(value) == bucket) / len(scenarios)
               for bucket in labels]
        # Small smoothing avoids infinite log loss when a finite residual sample misses a bucket.
        epsilon = 1e-6
        denom = 1.0 + epsilon * len(labels)
        probabilities = [(probability + epsilon) / denom for probability in raw]
        observed = math.floor(actual)
        brier_scores.append(sum((probability - (1.0 if bucket == observed else 0.0)) ** 2
                                for bucket, probability in zip(labels, probabilities)))
        log_losses.append(-math.log(max(epsilon, probabilities[labels.index(observed)]))
                          if observed in labels else -math.log(epsilon))
        top_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        confidence_hits.append((probabilities[top_index], float(labels[top_index] == observed)))
        crps_scores.append(_crps(scenarios, actual))
        sorted_scenarios = sorted(scenarios)
        p05 = sorted_scenarios[max(0, math.ceil(0.05 * len(sorted_scenarios)) - 1)]
        p95 = sorted_scenarios[min(len(sorted_scenarios) - 1,
                                  math.ceil(0.95 * len(sorted_scenarios)) - 1)]
        covered += int(p05 <= actual <= p95)
    n = len(actuals)
    ece = 0.0
    for lower in (0.0, 0.2, 0.4, 0.6, 0.8):
        group = [(confidence, hit) for confidence, hit in confidence_hits
                 if lower <= confidence < lower + 0.2 or (lower == 0.8 and confidence == 1.0)]
        if group:
            ece += len(group) / n * abs(sum(c for c, _ in group) / len(group)
                                       - sum(hit for _, hit in group) / len(group))
    return {"n": n, "brier": sum(brier_scores) / n,
            "log_loss": sum(log_losses) / n, "crps": sum(crps_scores) / n,
            "coverage_90": covered / n, "top_bin_ece": ece}


def fit_adaptive_weights(
    observations: dict[str, float], forecasts: dict[str, dict[str, float]],
    prior_weights: dict[str, float], min_train: int = 30,
    min_evaluation: int = 30, recent_window: int = 90,
) -> dict[str, Any]:
    """Optimize bounded weights on recent, chronological out-of-sample errors."""
    names = [name for name in prior_weights if name in forecasts and forecasts[name]]
    prior_total = sum(max(0.0, prior_weights.get(name, 0.0)) for name in names)
    if not names:
        return {"weights": {}, "adaptive": False, "reason": "sem previsões históricas disponíveis", "n": 0}
    if len(names) == 1:
        return {"weights": {names[0]: 1.0}, "adaptive": False,
                "reason": "apenas um modelo disponível; peso 100%", "n": 0}
    if prior_total <= 0:
        prior = {name: 1 / len(names) for name in names}
    else:
        prior = {name: max(0.0, prior_weights.get(name, 0.0)) / prior_total for name in names}

    common = sorted(set(observations).intersection(*(set(forecasts[name]) for name in names)))
    # Need past-only bias estimates plus enough independent forecast-error dates.
    required = min_train + min_evaluation
    if len(common) < required:
        return {"weights": prior, "adaptive": False, "reason":
                f"histórico comum insuficiente ({len(common)}; mínimo {required})", "n": len(common)}

    first_eval = max(min_train, len(common) - recent_window)
    rows: list[tuple[dict[str, float], float, str]] = []
    model_errors: dict[str, list[float]] = {name: [] for name in names}
    # Each row's bias is learned strictly from dates before that forecast.
    for index in range(first_eval, len(common)):
        day = common[index]
        train_days = common[max(0, index - 120):index]
        biases = {name: sum(forecasts[name][d] - observations[d] for d in train_days) / len(train_days)
                  for name in names}
        corrected = {name: forecasts[name][day] - biases[name] for name in names}
        actual = observations[day]
        rows.append((corrected, actual, day))
        for name in names:
            model_errors[name].append(corrected[name] - actual)
    if len(rows) < min_evaluation:
        return {"weights": prior, "adaptive": False,
                "reason": f"apenas {len(rows)} erros fora da amostra; mínimo {min_evaluation}", "n": len(rows)}

    # Pairwise error correlations quantify redundancy. The blend's OOS loss
    # already reflects covariance; this small explicit penalty discourages
    # assigning excess mass to highly redundant models.
    correlations: dict[tuple[str, str], float] = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            x, y = model_errors[left], model_errors[right]
            mx, my = sum(x) / len(x), sum(y) / len(y)
            vx = sum((value - mx) ** 2 for value in x)
            vy = sum((value - my) ** 2 for value in y)
            corr = sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy) if vx > 0 and vy > 0 else 0.0
            correlations[(left, right)] = max(-1.0, min(1.0, corr))

    candidates: list[dict[str, float]] = []
    grid = [step / 20 for step in range(2, 19)]  # 10%–90%, 5-point increments
    for values in product(grid, repeat=len(names) - 1):
        last = round(1.0 - sum(values), 10)
        vector = (*values, last)
        if last < 0.10 or last > 0.90:
            continue
        candidates.append(dict(zip(names, vector)))
    if not candidates:
        candidates = [prior]

    # Half-life 30 days favors recent performance without discarding history.
    recency = [0.5 ** ((len(rows) - 1 - i) / 30.0) for i in range(len(rows))]
    norm = sum(recency)

    def score(weights: dict[str, float]) -> tuple[float, float, float]:
        errors = [sum(weights[name] * row[0][name] for name in names) - row[1] for row in rows]
        rmse = math.sqrt(sum(w * err * err for w, err in zip(recency, errors)) / norm)
        mae = sum(w * abs(err) for w, err in zip(recency, errors)) / norm
        corr_penalty = sum(weights[a] * weights[b] * max(0.0, corr)
                           for (a, b), corr in correlations.items())
        regularizer = sum((weights[name] - prior[name]) ** 2 for name in names)
        objective = rmse + 0.15 * mae + rmse * (0.15 * corr_penalty + 0.10 * regularizer)
        return objective, rmse, mae

    base_score, base_rmse, base_mae = score(prior)
    best = min(candidates, key=lambda candidate: score(candidate)[0])
    best_score, best_rmse, best_mae = score(best)
    if best_score >= base_score * 0.99:
        return {"weights": prior, "adaptive": False,
                "reason": "otimização não melhorou ≥1% a validação recente; prior preservado",
                "n": len(rows), "base_rmse": base_rmse, "adaptive_rmse": best_rmse,
                "correlations": {f"{a}/{b}": value for (a, b), value in correlations.items()}}
    return {"weights": best, "adaptive": True,
            "reason": "pesos walk-forward aprovados; OOS recente favorece a combinação",
            "n": len(rows), "base_rmse": base_rmse, "adaptive_rmse": best_rmse,
            "adaptive_mae": best_mae,
            "correlations": {f"{a}/{b}": value for (a, b), value in correlations.items()}}


def walk_forward_probabilistic_backtest(
    observations: dict[str, float], forecasts: dict[str, dict[str, float]],
    prior_weights: dict[str, float], min_train: int = 60,
    residual_warmup: int = 15, min_evaluation: int = 30,
) -> dict[str, Any]:
    """Score models, fixed prior and adaptively re-fit blend on strictly past data."""
    # Restrict evaluation to models with an explicit active prior. A model
    # outage must not leave it in the blend with an implicit zero/missing weight.
    names = [name for name in prior_weights if name in forecasts and forecasts[name]]
    if not names:
        return {"n": 0, "reason": "nenhum modelo com histórico disponível"}
    common = sorted(set(observations).intersection(*(set(forecasts.get(n, {})) for n in names)))
    required = min_train + residual_warmup + min_evaluation
    if len(common) < required:
        return {"n": 0, "reason": f"histórico comum insuficiente ({len(common)} dias; necessário {required} para ≥{min_evaluation} datas de avaliação)"}

    strategy_names = (*names, "PRIOR", "ADAPTATIVA")
    residuals: dict[str, list[float]] = {name: [] for name in strategy_names}
    actuals: dict[str, list[float]] = {name: [] for name in strategy_names}
    predictions: dict[str, list[float]] = {name: [] for name in strategy_names}
    samples: dict[str, list[list[float]]] = {name: [] for name in strategy_names}
    active_prior = {name: prior_weights.get(name, 0.0) for name in prior_weights
                    if name in forecasts and name in names}
    prior_total = sum(active_prior.values())
    if prior_total <= 0:
        active_prior = {name: 1 / len(names) for name in names}
    else:
        active_prior = {name: value / prior_total for name, value in active_prior.items()}

    for index in range(min_train, len(common)):
        day = common[index]
        train_days = common[max(0, index - 120):index]
        train_obs = {d: observations[d] for d in train_days}
        train_forecasts = {name: {d: forecasts[name][d] for d in train_days} for name in names}
        adaptive = fit_adaptive_weights(train_obs, train_forecasts, active_prior,
                                       min_train=30, min_evaluation=30)
        adaptive_weights = adaptive["weights"] or active_prior
        biases = {name: sum(forecasts[name][d] - observations[d] for d in train_days) / len(train_days)
                  for name in names}
        row = {name: forecasts[name][day] - biases[name] for name in names}
        truth = observations[day]
        point_predictions = {name: row[name] for name in names}
        point_predictions["PRIOR"] = sum(active_prior[name] * row[name] for name in names)
        point_predictions["ADAPTATIVA"] = _weighted_forecast(row, adaptive_weights)
        for name, prediction in point_predictions.items():
            error = truth - prediction
            if len(residuals[name]) >= residual_warmup:
                actuals[name].append(truth)
                predictions[name].append(prediction)
                samples[name].append(list(residuals[name]))
            residuals[name].append(error)

    metrics = {name: _verification(actuals[name], predictions[name], samples[name])
               for name in strategy_names}
    metrics["common_days"] = len(common)
    metrics["evaluation_days"] = len(actuals["ADAPTATIVA"])
    metrics["method"] = "nested walk-forward; historical hourly maxima + expanding residual scenarios; bins [n,n+1)°C"
    return metrics


def resolution_status(config: dict[str, Any], market: dict[str, Any],
                      event_slug: str) -> tuple[bool, str]:
    """Require an explicitly reviewed contract mapping before showing market EV."""
    required = ("confirmed", "event_slug_prefix", "station_id", "source_name",
                "source_url", "timezone", "unit", "precision", "period_local", "bucket_mode")
    if not all(config.get(field) not in (None, "", False) for field in required):
        return False, "fonte de resolução não confirmada/configuração incompleta"
    if config.get("confirmed") is not True:
        return False, "fonte de resolução não confirmada"
    if config.get("unit") != "C" or config.get("bucket_mode") not in ("interval_start", "nearest_degree"):
        return False, "unidade ou regra de fronteira não é suportada/configurada"
    if str(config.get("precision", "")).replace(" ", "") not in ("1°C", "1C"):
        return False, "precisão não suportada; o cálculo atual aceita apenas faixas de 1°C"
    if not event_slug.startswith(str(config["event_slug_prefix"])):
        return False, "configuração de resolução não corresponde ao evento"
    rules = " ".join(str(market.get(key) or "") for key in
                      ("description", "resolutionSource", "resolvedBy", "_event_description"))
    configured_station = str(config["station_id"]).casefold()
    if configured_station not in rules.casefold():
        return False, f"regras do mercado não confirmam a estação configurada ({config['station_id']})"
    return True, "confirmada pela configuração e pelo texto das regras"


def parse_integer_bucket(label: str, bucket_mode: str = "interval_start") -> tuple[float | None, float | None] | None:
    """Interpret market labels as half-open Celsius intervals [lower, upper)."""
    import re
    text = (label or "").strip().lower().replace("°", "")
    nums = [int(x) for x in re.findall(r"-?\d+", text)]
    if not nums:
        return None
    half = 0.5 if bucket_mode == "nearest_degree" else 0.0
    if any(word in text for word in ("below", "less", "lower", "ou menos", "ou abaixo")):
        return None, nums[0] + half if half else nums[0] + 1
    if any(word in text for word in ("higher", "more", "above", "ou mais", "ou acima")):
        return nums[0] - half, None
    if len(nums) >= 2:
        low, high = sorted(nums[:2])
        return (low - half, high + half) if half else (low, high + 1)
    return (nums[0] - half, nums[0] + half) if half else (nums[0], nums[0] + 1)


def validate_ladder(markets: list[dict[str, Any]], bucket_mode: str = "interval_start") -> dict[str, Any]:
    parsed = []
    for market in markets:
        label = str(market.get("groupItemTitle") or "")
        if not label:
            return {"complete": False, "reason": "mercado sem título de faixa groupItemTitle; limites não presumidos"}
        bounds = parse_integer_bucket(label, bucket_mode)
        if bounds is None:
            return {"complete": False, "reason": f"faixa não interpretada: {label}"}
        parsed.append((bounds, label))
    if not parsed:
        return {"complete": False, "reason": "ladder vazia"}
    parsed.sort(key=lambda item: float("-inf") if item[0][0] is None else item[0][0])
    if parsed[0][0][0] is not None or parsed[-1][0][1] is not None:
        return {"complete": False, "reason": "faltam faixa inferior ou superior sem limite"}
    for current, following in zip(parsed, parsed[1:]):
        upper, next_lower = current[0][1], following[0][0]
        if upper is None or next_lower is None or upper != next_lower:
            return {"complete": False, "reason": f"lacuna/sobreposição entre {current[1]} e {following[1]}"}
    return {"complete": True, "reason": "faixas contíguas, sem sobreposição aparente"}
