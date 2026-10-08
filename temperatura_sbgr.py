#!/usr/bin/env python3
"""Radar de temperatura SBGR + ladder Polymarket.

Os preços vêm da API pública. A previsão combina membros de ECMWF, GFS e ICON
com pesos configuráveis, comparação de consenso e validação probabilística.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

import requests
from quant_analysis import (
    fit_adaptive_weights,
    resolution_status,
    validate_ladder,
    walk_forward_probabilistic_backtest,
)
from model_engine import (
    DEFAULT_MODEL_WEIGHTS,
    MODEL_CONFIG,
    choose_temperature_strategy,
    empirical_bracket_probability,
    fetch_previous_run_daily_max,
    fetch_station_daily_max,
    summarize_distribution,
    weighted_member_distribution,
    weighted_quantile,
    model_consensus,
)

# Preserve Portuguese accents, Greek μ/σ, and table separators in Windows terminals.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


METAR_URL = "https://aviationweather.gov/api/data/metar"
GAMMA_URL = "https://gamma-api.polymarket.com/events"
CLOB_URL = "https://clob.polymarket.com"
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
# Coordenadas do Aeroporto Internacional de São Paulo/Guarulhos (SBGR).
SBGR_LATITUDE = -23.4356
SBGR_LONGITUDE = -46.4731
HEADERS = {"User-Agent": "SBGR-temperature-ladder/1.0 (personal dashboard)"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)
_PROBABILITY_BACKTEST_CACHE: dict[tuple[int, tuple[tuple[str, float], ...]], tuple[float, dict[str, Any]]] = {}


def get_json(url: str, params: dict[str, Any]) -> Any:
    response = SESSION.get(url, params=params, timeout=12)
    response.raise_for_status()
    return response.json()


def fetch_metars() -> list[dict[str, Any]]:
    """Return the last 24 hours of parsed reports for SBGR only."""
    data = get_json(METAR_URL, {"ids": "SBGR", "hours": 24, "format": "json"})
    return data or []


def describe_metar(report: dict[str, Any]) -> list[str]:
    """Decode common METAR groups using AviationWeather's parsed JSON fields."""
    lines: list[str] = []
    raw = report.get("rawOb") or report.get("raw")
    if raw:
        lines.append(f"Mensagem original: {raw}")

    observed = report.get("obsTime")
    if observed:
        try:
            stamp = datetime.fromtimestamp(float(observed)).astimezone()
            lines.append(f"Observação: {stamp:%d/%m/%Y %H:%M:%S %Z}")
        except (TypeError, ValueError, OSError):
            lines.append(f"Observação (UTC): {observed}")

    wind = report.get("wdir")
    speed = report.get("wspd")
    gust = report.get("wgst")
    if speed is not None:
        direction = "variável" if wind in (None, "VRB") else f"{wind}°"
        gust_text = f", rajadas {gust} kt" if gust is not None else ""
        lines.append(f"Vento: {direction}, {speed} kt{gust_text}")
    if report.get("visib") is not None:
        lines.append(f"Visibilidade: {report['visib']} mi")

    weather = report.get("wxString")
    if weather:
        lines.append(f"Tempo presente: {weather}")

    clouds = report.get("clouds") or []
    if clouds:
        cloud_names = {
            "FEW": "poucas nuvens", "SCT": "nuvens dispersas",
            "BKN": "céu parcialmente encoberto", "OVC": "céu encoberto",
            "VV": "visibilidade vertical",
        }
        decoded = []
        for cloud in clouds:
            cover = cloud.get("cover", "")
            base = cloud.get("base")
            label = cloud_names.get(cover, cover or "camada")
            if base is not None:
                label += f" a {base} ft"
            decoded.append(label)
        lines.append("Nuvens: " + "; ".join(decoded))

    if report.get("temp") is not None:
        lines.append(f"Temperatura: {report['temp']} °C")
    if report.get("dewp") is not None:
        lines.append(f"Ponto de orvalho: {report['dewp']} °C")
    if report.get("altim") is not None:
        lines.append(f"Pressão QNH: {report['altim']} hPa")
    if len(lines) <= 2:
        lines.append("A API não retornou campos decodificados adicionais.")
    return lines


def event_slug(day: date) -> str:
    month = day.strftime("%B").lower()
    return f"highest-temperature-in-sao-paulo-on-{month}-{day.day}-{day.year}"


def fetch_ladder(day: date) -> list[dict[str, Any]]:
    data = get_json(GAMMA_URL, {"slug": event_slug(day)})
    if not data:
        return []
    event = data[0]
    markets = event.get("markets", []) or []
    for market in markets:
        market["_event_slug"] = event.get("slug") or event_slug(day)
        market["_event_title"] = event.get("title")
        market["_event_description"] = event.get("description")
    return markets


def as_list(value: Any) -> list[Any]:
    """Gamma sometimes serializes array fields as JSON strings."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def no_token_id(market: dict[str, Any]) -> str | None:
    outcomes = as_list(market.get("outcomes"))
    token_ids = as_list(market.get("clobTokenIds"))
    for index, outcome in enumerate(outcomes):
        if str(outcome).strip().casefold() == "no" and index < len(token_ids):
            return str(token_ids[index])
    return None


def outcome_token_id(market: dict[str, Any], outcome_name: str) -> str | None:
    outcomes = as_list(market.get("outcomes"))
    token_ids = as_list(market.get("clobTokenIds"))
    for index, outcome in enumerate(outcomes):
        if str(outcome).strip().casefold() == outcome_name.casefold() and index < len(token_ids):
            return str(token_ids[index])
    return None


def fetch_orderbook(token_id: str) -> dict[str, Any]:
    return get_json(f"{CLOB_URL}/book", {"token_id": token_id})


def estimate_buy(book: dict[str, Any], contracts: float,
                 fee_schedule: tuple[float, float] | None) -> dict[str, float | None]:
    """Estimate the average taker fill by walking ask depth for fixed shares."""
    asks = []
    for level in book.get("asks") or []:
        price = as_price(level.get("price"))
        try:
            size = float(level.get("size"))
        except (TypeError, ValueError):
            continue
        if price is not None and 0 < price <= 1 and math.isfinite(size) and size > 0:
            asks.append((price, size))
    asks.sort()
    if not asks:
        return {"best": None, "average": None, "slippage": None, "fee": None}
    remaining = contracts
    paid = fee_total = 0.0
    filled = 0.0
    rate, exponent = fee_schedule if fee_schedule is not None else (math.nan, math.nan)
    for price, size in asks:
        quantity = min(size, remaining)
        paid += quantity * price
        filled += quantity
        if fee_schedule is not None:
            fee_total += quantity * rate * (price * (1.0 - price)) ** exponent
        remaining -= quantity
        if remaining <= 1e-9:
            break
    if remaining > 1e-9:
        return {"best": asks[0][0], "average": None, "slippage": None, "fee": None}
    average = paid / filled
    return {"best": asks[0][0], "average": average,
            "slippage": average - asks[0][0],
            "fee": fee_total / filled if fee_schedule is not None else None}


def orderbook_age_minutes(book: dict[str, Any]) -> float | None:
    raw = book.get("timestamp") or book.get("last_update") or book.get("updatedAt")
    if raw is None:
        return None
    try:
        if isinstance(raw, (int, float)):
            stamp = float(raw)
            if stamp > 10_000_000_000:
                stamp /= 1000
            return max(0.0, (time.time() - stamp) / 60)
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return max(0.0, (datetime.now(parsed.tzinfo) - parsed).total_seconds() / 60)
    except (TypeError, ValueError, OSError):
        return None


def load_resolution_config() -> dict[str, Any]:
    path = Path(__file__).with_name("market_resolution.json")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def get_probability_backtest(lead_days: int, prior_weights: dict[str, float],
                             available_models: list[str] | None = None) -> dict[str, Any]:
    """Fit weights from history before the target and return a separate OOS scorecard."""
    if not 0 <= lead_days <= 7:
        return {"metrics": {"n": 0, "reason": "arquivo histórico disponível apenas para D-0 a D-7"},
                "fit": {"weights": {}, "adaptive": False, "reason": "horizonte fora do arquivo", "n": 0}}
    available_models = available_models or list(prior_weights)
    key = (lead_days, tuple(sorted(prior_weights.items())) + tuple((name, -1.0) for name in available_models))
    cached = _PROBABILITY_BACKTEST_CACHE.get(key)
    if cached and time.time() - cached[0] < 6 * 60 * 60:
        return cached[1]
    observations = fetch_station_daily_max()
    # Fit and score exactly the models that contributed live members. This
    # also makes a partial outage graceful instead of referencing absent priors.
    active_models = [name for name in available_models if name in MODEL_CONFIG]
    forecasts: dict[str, dict[str, float]] = {}
    history_errors: dict[str, str] = {}
    for name in active_models:
        try:
            forecasts[name] = fetch_previous_run_daily_max(MODEL_CONFIG[name]["history"], lead_days)
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            forecasts[name] = {}
            history_errors[name] = str(exc)
    blend_prior = {name: prior_weights.get(name, 0.0) for name in active_models
                   if forecasts.get(name)}
    if not forecasts:
        return {"fit": {"weights": {}, "adaptive": False,
                        "reason": "nenhum modelo histórico disponível", "n": 0},
                "metrics": {"n": 0, "reason": "nenhum modelo histórico disponível"}}
    if not blend_prior:
        detail = "; ".join(f"{name}: {error}" for name, error in history_errors.items())
        return {"fit": {"weights": {}, "adaptive": False,
                        "reason": "nenhum modelo ativo tem arquivo histórico" + (f" ({detail})" if detail else ""), "n": 0},
                "metrics": {"n": 0, "reason": "nenhum modelo ativo tem arquivo histórico"}}
    fit = fit_adaptive_weights(observations, forecasts, blend_prior)
    metrics = walk_forward_probabilistic_backtest(observations, forecasts, blend_prior)
    missing_history = sorted(set(active_models) - set(blend_prior))
    if missing_history:
        fit["reason"] += "; sem arquivo para " + ", ".join(missing_history)
    result = {"fit": fit, "metrics": metrics}
    _PROBABILITY_BACKTEST_CACHE[key] = (time.time(), result)
    return result


def fetch_best_quotes(token_id: str) -> tuple[float | None, float | None]:
    """Return best bid and ask from one outcome token's public CLOB book."""
    book = get_json(f"{CLOB_URL}/book", {"token_id": token_id})
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    bid_prices = [price for item in bids if (price := as_price(item.get("price"))) is not None]
    ask_prices = [price for item in asks if (price := as_price(item.get("price"))) is not None]
    return (max(bid_prices) if bid_prices else None,
            min(ask_prices) if ask_prices else None)


def market_fee_schedule(market: dict[str, Any]) -> tuple[float, float] | None:
    """Read the current per-market taker fee schedule; None means unknown."""
    trading = market.get("trading") or {}
    enabled = market.get("feesEnabled", trading.get("feesEnabled"))
    schedule = market.get("feeSchedule") or trading.get("feeSchedule")
    if enabled is False:
        return 0.0, 1.0
    if not isinstance(schedule, dict):
        return None
    try:
        rate = float(schedule["rate"])
        exponent = float(schedule.get("exponent", 1))
    except (KeyError, TypeError, ValueError):
        return None
    if rate < 0 or exponent <= 0:
        return None
    return rate, exponent


def taker_fee_per_contract(price: float, schedule: tuple[float, float] | None) -> float | None:
    """Polymarket fee per share: rate × (price × (1-price)) ** exponent."""
    if schedule is None:
        return None
    rate, exponent = schedule
    return rate * (price * (1.0 - price)) ** exponent


def fetch_temperature_forecast(
    day: date,
    model: str = "ecmwf_ifs025_ensemble",
    source_name: str = "ECMWF IFS ensemble 0,25° (~25 km)",
    max_days: float = 15,
) -> tuple[float, float, int, str, list[float]]:
    """Estimate daily-maximum mu and sigma from an ensemble's members."""
    days_ahead = (day - date.today()).days
    if days_ahead < 0:
        raise ValueError("a previsão automática só está disponível para hoje ou datas futuras")
    if days_ahead >= max_days:
        raise ValueError(f"o modelo {source_name} cobre até {max_days:g} dias à frente")

    payload = get_json(ENSEMBLE_URL, {
        "latitude": SBGR_LATITUDE,
        "longitude": SBGR_LONGITUDE,
        "daily": "temperature_2m_max",
        "models": model,
        "timezone": "America/Sao_Paulo",
        "forecast_days": math.ceil(max_days),
    })
    daily = payload.get("daily") or {}
    dates = daily.get("time") or []
    try:
        index = dates.index(day.isoformat())
    except ValueError as exc:
        raise ValueError("o modelo ainda não disponibilizou previsão para essa data") from exc

    # Open-Meteo names daily variables per ensemble member. Keep only actual
    # members so metadata/summary series cannot be mistaken for scenarios.
    member_series = [
        values for key, values in daily.items()
        if re.fullmatch(r"temperature_2m_max_member\d+", key)
        and isinstance(values, list) and index < len(values)
        and isinstance(values[index], (int, float))
    ]
    members = [float(values[index]) for values in member_series]
    if len(members) < 2:
        raise ValueError("a API não retornou ao menos dois cenários do ensemble para essa data")

    mu = sum(members) / len(members)
    variance = sum((value - mu) ** 2 for value in members) / (len(members) - 1)
    sigma = math.sqrt(variance)
    if sigma <= 0:
        raise ValueError("o ensemble retornou dispersão nula; não é possível estimar probabilidades")
    return mu, sigma, len(members), source_name, members


def normal_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def bracket_probability(label: str, mu: float, sigma: float) -> float | None:
    """PDF's Celsius ladder semantics: integer bucket [n, n+1)."""
    text = (label or "").strip().lower().replace("°", "")
    nums = [int(n) for n in re.findall(r"-?\d+", text)]
    if not nums:
        return None
    if any(word in text for word in ("below", "or less", "or lower", "ou menos", "ou abaixo")):
        return normal_cdf(nums[0] + 1, mu, sigma)
    if any(word in text for word in ("higher", "or more", "or above", "ou mais", "ou acima")):
        return 1.0 - normal_cdf(nums[0], mu, sigma)
    if len(nums) >= 2:
        low, high = sorted(nums[:2])
        return normal_cdf(high + 1, mu, sigma) - normal_cdf(low, mu, sigma)
    return normal_cdf(nums[0] + 1, mu, sigma) - normal_cdf(nums[0], mu, sigma)


def as_price(value: Any) -> float | None:
    try:
        number = float(value)
        return number if 0 <= number <= 1 else None
    except (TypeError, ValueError):
        return None


def print_dashboard(day: date, mu: float | None, sigma: float | None,
                    forecast_info: tuple[int, str] | None = None,
                    forecast_error: str | None = None,
                    comparison_forecasts: dict[str, tuple[float, float, int, str] | str] | None = None,
                    probability_distribution: list[tuple[float, float]] | None = None,
                    configured_weights: dict[str, float] | None = None,
                    effective_weights: dict[str, float] | None = None,
                    prediction_interval: tuple[float, float] | None = None,
                    backtest: dict[str, Any] | None = None,
                    contracts_to_estimate: float = 10.0,
                    quantiles: tuple[float, float, float] | None = None,
                    forecast_fetched_at: datetime | None = None) -> None:
    print("=" * 92)
    if mu is not None and sigma is not None:
        count, source = forecast_info or (0, "parâmetros informados manualmente")
        source_text = f"{source}; {count} membros" if count else source
        print(f"[PREVISÃO DA MÁXIMA] μ={mu:.2f} °C | σ={sigma:.2f} °C | {source_text}")
        if probability_distribution:
            print("  Probabilidades calculadas pela frequência ponderada dos membros; estimativas, não garantias.")
            if prediction_interval:
                print(f"  Intervalo de previsão empírico 95%: [{prediction_interval[0]:.2f}, {prediction_interval[1]:.2f}] °C; não é IC da média.")
            if quantiles:
                print(f"  P5={quantiles[0]:.2f}°C | P50={quantiles[1]:.2f}°C | P95={quantiles[2]:.2f}°C")
            if forecast_fetched_at:
                print(f"  Consultado em: {forecast_fetched_at.astimezone().strftime('%d/%m/%Y %H:%M:%S %Z')} (horário local; não representa a hora de inicialização do modelo).")
            if configured_weights and effective_weights:
                configured = ", ".join(f"{name} {weight:.0%}" for name, weight in configured_weights.items())
                active = ", ".join(f"{name} {weight:.0%}" for name, weight in effective_weights.items())
                print(f"  Pesos configurados: {configured} | aplicados: {active}")
            means = {name: item[0] for name, item in (comparison_forecasts or {}).items()
                     if isinstance(item, tuple)}
            consensus = model_consensus(means)
            if consensus["score"] is not None:
                print(f"  Confiança relativa pela concordância: {consensus['level']} (heurística {consensus['score']:.0f}/100); divergência máxima {consensus['spread']:.2f}°C.")
                if consensus["zones"]:
                    print("  Zonas de desacordo: " + "; ".join(consensus["zones"]))
                print("  Consenso mede proximidade entre modelos; não é chance de acerto.")
        else:
            print("  Probabilidades derivadas de distribuição normal ajustada ao ensemble; não são garantias.")
        if not probability_distribution:
            print("  P(YES) e EV usam esta previsão principal; GFS e ICON são exibidos para comparação.")
    elif forecast_error:
        print(f"[PREVISÃO DA MÁXIMA] indisponível: {forecast_error}")
    for model_name, result in (comparison_forecasts or {}).items():
        if isinstance(result, tuple):
            model_mu, model_sigma, count, source = result
            print(f"[PREVISÃO {model_name}] μ={model_mu:.2f} °C | σ={model_sigma:.2f} °C | {source}; {count} membros")
        else:
            print(f"[PREVISÃO {model_name}] indisponível: {result}")
    if backtest:
        fit = backtest.get("fit", {})
        metrics = backtest.get("metrics", backtest)
        lead = max(0, (day - date.today()).days)
        if fit:
            print(f"[PESOS ADAPTATIVOS D-{lead}] {'ajustados' if fit.get('adaptive') else 'prior/fallback'} — {fit.get('reason', 'sem detalhe')}; amostra recente: {fit.get('n', 0)} dias.")
            if fit.get("base_rmse") is not None and fit.get("adaptive_rmse") is not None:
                print(f"  RMSE OOS de ajuste: prior {fit['base_rmse']:.2f}°C → candidato {fit['adaptive_rmse']:.2f}°C.")
            if fit.get("correlations"):
                print("  Correlação dos erros por par: " + ", ".join(f"{pair} {value:+.2f}" for pair, value in fit["correlations"].items()))
        elif effective_weights:
            print(f"[PESOS ADAPTATIVOS D-{lead}] prior/fallback — histórico indisponível; distribuição atual reequilibrada entre modelos disponíveis: "
                  + ", ".join(f"{name} {weight:.0%}" for name, weight in effective_weights.items()) + ".")
        adaptive_metrics = metrics.get("ADAPTATIVA", {})
        if adaptive_metrics.get("n"):
            print(f"[BACKTEST PROBABILÍSTICO WALK-FORWARD D-{lead}] {adaptive_metrics['n']} dias; "
                  f"Brier {adaptive_metrics['brier']:.3f} | log loss {adaptive_metrics['log_loss']:.3f} | "
                  f"CRPS {adaptive_metrics['crps']:.2f}°C | cobertura P5–P95 {adaptive_metrics['coverage_90']:.0%} | ECE {adaptive_metrics['top_bin_ece']:.3f}")
            for name in ("PRIOR", "ECMWF", "GFS", "ICON"):
                item = metrics.get(name, {})
                if item.get("n"):
                    print(f"  {name}: Brier {item['brier']:.3f} | log loss {item['log_loss']:.3f} | CRPS {item['crps']:.2f}°C | cobertura {item['coverage_90']:.0%} | ECE {item['top_bin_ece']:.3f}")
            print("  Método: decisões sequenciais com apenas histórico anterior; máximas horárias arquivadas + distribuição de erros walk-forward, bins [n,n+1)°C.")
        else:
            print(f"[BACKTEST PROBABILÍSTICO] indisponível: {metrics.get('reason', 'dados insuficientes')}.")
    print(f"RADAR METEOROLÓGICO — SBGR | Ladder do dia {day:%d/%m/%Y}")
    print("=" * 92)
    try:
        reports = fetch_metars()
        print("\n[METAR SBGR — Aviation Weather Center]")
        print("-" * 92)
        if reports:
            # The API returns newest first. Show recent observed temperatures,
            # then fully decode the newest report.
            for report in reports[:6]:
                stamp = report.get("obsTime")
                try:
                    local_time = datetime.fromtimestamp(float(stamp)).astimezone()
                    temperature = report.get("temp")
                    temp_text = f"{temperature:.2f} °C" if isinstance(temperature, (int, float)) else "sem temperatura"
                    print(f"  {local_time:%d/%m %H:%M} — {temp_text}")
                except (TypeError, ValueError, OSError):
                    print(f"  Temperatura observada: {report.get('temp', 'indisponível')} °C")
            target_day_temps = []
            for report in reports:
                try:
                    local_time = datetime.fromtimestamp(float(report["obsTime"])).astimezone()
                    temp = report.get("temp")
                    if local_time.date() == day and isinstance(temp, (int, float)):
                        target_day_temps.append(float(temp))
                except (KeyError, TypeError, ValueError, OSError):
                    continue
            if target_day_temps:
                print(f"  Máxima METAR observada em {day:%d/%m}: {max(target_day_temps):.2f} °C")
            report = reports[0]
            print("-" * 92)
            print("  Decodificação do METAR mais recente:")
            for line in describe_metar(report):
                print("  " + line)
        else:
            print("  Nenhum METAR recente encontrado para SBGR.")
            print("-" * 92)
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"\n[METAR SBGR] Erro ao consultar: {exc}")

    print("\n" + "=" * 92)
    print(f"\n[POLYMARKET — {event_slug(day)}]")
    print("-" * 92)
    try:
        markets = fetch_ladder(day)
        if not markets:
            print("  Evento/ladder não encontrado para essa data.")
            return
        resolution = load_resolution_config()
        resolution_checks = [resolution_status(resolution, market, market.get("_event_slug", event_slug(day))) for market in markets]
        resolved = all(item[0] for item in resolution_checks)
        resolution_reason = next((item[1] for item in resolution_checks if not item[0]), "regras verificadas para todos os outcomes")
        bucket_mode = resolution.get("bucket_mode", "interval_start")
        ladder = validate_ladder(markets, bucket_mode)
        can_price_ev = resolved and ladder["complete"]
        print("\n[REGRAS DE RESOLUÇÃO]")
        print(f"  Situação: {'confirmada' if resolved else 'FONTE DE RESOLUÇÃO NÃO CONFIRMADA'} — {resolution_reason}")
        print(f"  Estação que precisa ser confirmada: {resolution.get('station_id') or 'não configurada'}")
        if resolution.get("source_name"):
            print(f"  Fonte configurada: {resolution.get('source_name')} | estação: {resolution.get('station_id')} | fuso: {resolution.get('timezone')} | unidade: {resolution.get('unit')} | precisão: {resolution.get('precision')} | período: {resolution.get('period_local')}")
        print(f"  Semântica de fronteira configurada: {resolution.get('bucket_mode') or 'não informada'}")
        print(f"  Fonte indicada nas regras/API: {markets[0].get('resolutionSource') or markets[0].get('resolvedBy') or 'não informada'}")
        print(f"  Regras: {resolution.get('source_url') or markets[0].get('resolutionSource') or 'URL de regras não disponível'}")
        print(f"  Ladder: {'válida' if ladder['complete'] else 'não validada'} — {ladder['reason']}")
        if not resolved:
            print("  P(modelo), EV e recomendação ficam suspensos até confirmar a regra, estação e precisão no arquivo market_resolution.json.")
        elif not ladder["complete"]:
            print("  P(modelo), EV e recomendação ficam suspensos até corrigir/confirmar as fronteiras das faixas.")
        columns = [("Faixa", 20), ("A.Y", 7), ("Spr.Y", 7), ("Sl.Y", 7), ("A.N", 7), ("Spr.N", 7), ("Sl.N", 7), ("P(impl.)", 10), ("P(YES)", 10), ("Edge", 9), ("EV YES", 10), ("EV% YES", 10), ("EV NO", 10), ("EV% NO", 10), ("Estado", 12)]
        def render_row(values: list[str]) -> str:
            cells = [f"{values[0]:<20.20}"] + [f"{value:>{columns[i][1]}}" for i, value in enumerate(values[1:], 1)]
            return "| " + " | ".join(cells) + " |"
        table_rule = "-" * len(render_row([name for name, _ in columns]))
        print(table_rule)
        print(render_row([name for name, _ in columns]))
        print(table_rule)
        probabilities_total = 0.0
        strategy_markets: list[dict[str, Any]] = []
        for market in markets:
            label = market.get("groupItemTitle") or market.get("question") or "Faixa sem nome"
            bid = as_price(market.get("bestBid"))
            ask = as_price(market.get("bestAsk"))
            yes_bid, yes_ask = bid, ask
            bid_no = ask_no = None
            yes_fill = {"average": None, "slippage": None, "fee": None}
            no_fill = {"average": None, "slippage": None, "fee": None}
            ages: list[float] = []
            schedule = market_fee_schedule(market)
            yes_id = outcome_token_id(market, "Yes")
            no_id = outcome_token_id(market, "No")
            try:
                if yes_id:
                    yes_book = fetch_orderbook(yes_id)
                    yes_levels_bid = [as_price(x.get("price")) for x in (yes_book.get("bids") or [])]
                    yes_bid = max((x for x in yes_levels_bid if x is not None), default=yes_bid)
                    yes_fill = estimate_buy(yes_book, contracts_to_estimate, schedule)
                    age = orderbook_age_minutes(yes_book)
                    if age is not None:
                        ages.append(age)
                if no_id:
                    no_book = fetch_orderbook(no_id)
                    no_levels_bid = [as_price(x.get("price")) for x in (no_book.get("bids") or [])]
                    bid_no = max((x for x in no_levels_bid if x is not None), default=None)
                    no_fill = estimate_buy(no_book, contracts_to_estimate, schedule)
                    ask_no = no_fill["best"]
                    age = orderbook_age_minutes(no_book)
                    if age is not None:
                        ages.append(age)
            except requests.RequestException:
                pass
            ask = yes_fill["best"] if yes_fill["best"] is not None else yes_ask
            spread_yes = ask - yes_bid if ask is not None and yes_bid is not None else None
            spread_no = ask_no - bid_no if ask_no is not None and bid_no is not None else None
            model_probability = (
                empirical_bracket_probability(str(label), probability_distribution, bucket_mode)
                if probability_distribution
                else bracket_probability(str(label), mu, sigma) if mu is not None and sigma is not None else None
            ) if can_price_ev else None
            probability = model_probability
            if probability is not None:
                probabilities_total += probability
            no_probability = 1.0 - probability if probability is not None else None
            fresh = not ages or max(ages) <= 15
            complete_depth_yes = yes_fill["average"] is not None
            complete_depth_no = no_fill["average"] is not None
            fee_yes, fee_no = yes_fill["fee"], no_fill["fee"]
            ev_yes_net = probability - yes_fill["average"] - fee_yes if can_price_ev and fresh and complete_depth_yes and fee_yes is not None and probability is not None else None
            ev_no_net = no_probability - no_fill["average"] - fee_no if can_price_ev and fresh and complete_depth_no and fee_no is not None and no_probability is not None else None
            ev_yes_pct = ev_yes_net / (yes_fill["average"] + fee_yes) * 100 if ev_yes_net is not None and yes_fill["average"] is not None and fee_yes is not None and yes_fill["average"] + fee_yes > 0 else None
            ev_no_pct = ev_no_net / (no_fill["average"] + fee_no) * 100 if ev_no_net is not None and no_fill["average"] is not None and fee_no is not None and no_fill["average"] + fee_no > 0 else None
            implied = ((yes_bid + ask) / 2) if yes_bid is not None and ask is not None else (ask if ask is not None else yes_bid)
            edge = probability - implied if probability is not None and implied is not None else None
            if not ages:
                status = "idade ?"
            elif not fresh:
                status = f"stale {max(ages):.0f}m"
            elif schedule is None:
                status = "taxa ?"
            elif not complete_depth_yes or not complete_depth_no:
                status = "profundidade < alvo"
            else:
                status = "atual"

            def cell(value: float | None, fmt: str) -> str:
                return format(value, fmt) if value is not None else "—"

            row = [str(label), cell(ask, ".3f"), cell(spread_yes, ".3f"), cell(yes_fill["slippage"], ".3f"), cell(ask_no, ".3f"), cell(spread_no, ".3f"), cell(no_fill["slippage"], ".3f"), f"{cell(implied * 100 if implied is not None else None, '.3f')}%", f"{cell(probability * 100 if probability is not None else None, '.3f')}%", f"{cell(edge * 100 if edge is not None else None, '+.3f')}%", cell(ev_yes_net, "+.3f"), f"{cell(ev_yes_pct, '+.3f')}%", cell(ev_no_net, "+.3f"), f"{cell(ev_no_pct, '+.3f')}%", status]
            print(render_row(row))
            print(table_rule)
            if can_price_ev and probability_distribution and yes_fill["average"] is not None and fee_yes is not None and fresh:
                effective_cost = yes_fill["average"] + fee_yes
                if effective_cost < 1:
                    strategy_markets.append({**market, "bestAsk": effective_cost})
        print("\n  B/A = best bid/ask; Spr. = Ask − Bid. O ask já incorpora o spread pago ao cruzar o livro;")
        print(f"  Sl. = preço médio estimado acima do melhor ask para {contracts_to_estimate:g} contratos; EV usa profundidade, taxa por nível e slippage estimado.")
        print("  Preço implícito usa o meio do spread quando bid/ask existem; é referência, não preço executável.")
        if can_price_ev:
            print(f"  Soma das probabilidades da ladder: {probabilities_total * 100:.2f}%" +
                  (" — alerta: confira faixas/regras." if abs(probabilities_total - 1.0) > 0.03 else " — cobertura coerente."))
        if can_price_ev and probability_distribution:
            strategy = choose_temperature_strategy(
                strategy_markets,
                probability_distribution,
                lambda price, market: 0.0,
                bucket_mode,
            )
            print("\n  SUGESTÃO DE TEMPERATURA (estimativa; sem execução de ordens):")
            if strategy:
                labels = " + ".join(item["label"] for item in strategy["items"])
                print(f"    {strategy['kind']}: {labels}")
                print(f"    Probabilidade de acerto conjunta estimada: {strategy['probability'] * 100:.2f}%")
                print(f"    Custo total estimado: ${strategy['cost']:.3f} | EV líquido: ${strategy['ev']:+.3f} | EV%: {strategy['roi'] * 100:+.2f}%")
                print(f"    Estimativa baseada no livro para {contracts_to_estimate:g} contratos, com taxa e slippage; não garante lucro.")
            else:
                print("    Nenhuma faixa ou cesta adjacente apresentou EV líquido positivo com as cotações atuais.")
        if mu is None:
            print("  Sem previsão: probabilidade estimada e EV não foram calculados.")
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        print(f"  Erro ao consultar ladder: {exc}")
    print("=" * 92)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitora o METAR de SBGR, ensembles ECMWF/GFS/ICON e ladder Polymarket.")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(), help="Data do mercado, YYYY-MM-DD (padrão: hoje).")
    parser.add_argument("--mu", type=float, help="Substitui a média automática da máxima diária em °C (μ).")
    parser.add_argument("--sigma", type=float, help="Substitui o desvio padrão automático em °C (σ), maior que zero.")
    parser.add_argument("--watch", action="store_true", help="Atualiza continuamente até Ctrl+C.")
    parser.add_argument("--interval", type=int, default=300, help="Intervalo entre consultas em segundos (padrão: 300).")
    parser.add_argument("--weights", default="ECMWF=0.5,GFS=0.3,ICON=0.2",
                        help="Pesos dos modelos, por exemplo ECMWF=0.5,GFS=0.3,ICON=0.2 (soma 1).")
    parser.add_argument("--contracts", type=float, default=10.0,
                        help="Tamanho usado para estimar slippage no livro (padrão: 10 contratos).")
    args = parser.parse_args()
    if (args.mu is None) != (args.sigma is None):
        parser.error("informe --mu e --sigma juntos")
    if args.sigma is not None and args.sigma <= 0:
        parser.error("--sigma deve ser maior que zero")
    if args.interval < 60:
        parser.error("--interval deve ser de pelo menos 60 segundos para respeitar os serviços consultados")
    try:
        args.model_weights = {}
        for component in args.weights.split(","):
            name, value = component.split("=", 1)
            name = name.strip().upper()
            if name not in DEFAULT_MODEL_WEIGHTS or name in args.model_weights:
                raise ValueError
            args.model_weights[name] = float(value)
        if set(args.model_weights) != set(DEFAULT_MODEL_WEIGHTS) or any(
            not math.isfinite(value) or value < 0 for value in args.model_weights.values()
        ) or not math.isclose(sum(args.model_weights.values()), 1.0, abs_tol=1e-6):
            raise ValueError
    except (ValueError, TypeError):
        parser.error("--weights deve listar ECMWF, GFS e ICON com pesos não negativos somando 1")
    if not math.isfinite(args.contracts) or args.contracts <= 0:
        parser.error("--contracts deve ser maior que zero")
    return args


def main() -> None:
    args = parse_args()
    while True:
        mu, sigma = args.mu, args.sigma
        forecast_info: tuple[int, str] | None = None
        forecast_error: str | None = None
        comparison_forecasts: dict[str, tuple[float, float, int, str] | str] = {}
        model_specs = (
            ("ECMWF", "ecmwf_ifs025_ensemble", "ECMWF IFS ensemble 0,25° (~25 km)", 15),
            ("GFS", "ncep_gefs025", "GFS ensemble 0,25° (~25 km)", 10),
            ("ICON", "dwd_icon_global_eps", "ICON global ensemble (~26 km)", 7.5),
        )
        member_sets: dict[str, list[float]] = {}
        model_summaries: dict[str, tuple[float, float, int, str]] = {}
        for model_name, model_id, source, horizon in model_specs:
            try:
                result = fetch_temperature_forecast(args.date, model_id, source, horizon)
                model_mu, model_sigma, count, model_source, members = result
                member_sets[model_name] = members
                model_summaries[model_name] = (model_mu, model_sigma, count, model_source)
                comparison_forecasts[model_name] = model_summaries[model_name]
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                comparison_forecasts[model_name] = str(exc)
                if mu is None and not member_sets:
                    forecast_error = f"nenhum ensemble disponível: {exc}"

        probability_distribution: list[tuple[float, float]] | None = None
        effective_weights: dict[str, float] = {}
        prediction_interval: tuple[float, float] | None = None
        forecast_quantiles: tuple[float, float, float] | None = None
        forecast_fetched_at: datetime | None = None
        backtest: dict[str, Any] | None = None
        if args.mu is None and args.sigma is None and member_sets:
            lead_days = max(0, (args.date - date.today()).days)
            try:
                backtest = get_probability_backtest(lead_days, args.model_weights, list(member_sets))
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                backtest = {"metrics": {"n": 0, "reason": f"não foi possível obter o arquivo histórico: {exc}"},
                            "fit": {"weights": {}, "adaptive": False, "reason": "histórico indisponível; usando pesos-base", "n": 0}}
            fitted = (backtest.get("fit") or {}).get("weights") or {}
            fitted_live = {name: fitted[name] for name in member_sets if name in fitted}
            if fitted_live:
                # Do not pretend a live model is calibrated when its historical
                # archive is absent; renormalize across the historically usable set.
                effective_weights = fitted_live
                norm = sum(effective_weights.values())
                effective_weights = {name: value / norm for name, value in effective_weights.items()} if norm > 0 else {}
            if not effective_weights:
                prior_total = sum(args.model_weights[name] for name in member_sets)
                if prior_total > 0:
                    effective_weights = {name: args.model_weights[name] / prior_total for name in member_sets}
                elif member_sets:
                    effective_weights = {name: 1 / len(member_sets) for name in member_sets}
            probability_distribution = weighted_member_distribution(
                member_sets, configured_weights=effective_weights
            )
            if probability_distribution:
                mu, sigma = summarize_distribution(probability_distribution)
                forecast_fetched_at = datetime.now().astimezone()
                count = sum(len(values) for values in member_sets.values())
                forecast_info = (count, "mistura empírica " + "+".join(member_sets))
                low, high = weighted_quantile(probability_distribution, 0.025), weighted_quantile(probability_distribution, 0.975)
                if low is not None and high is not None:
                    prediction_interval = (low, high)
                p05 = weighted_quantile(probability_distribution, 0.05)
                p50 = weighted_quantile(probability_distribution, 0.50)
                p95 = weighted_quantile(probability_distribution, 0.95)
                if p05 is not None and p50 is not None and p95 is not None:
                    forecast_quantiles = (p05, p50, p95)
            else:
                forecast_error = "os modelos disponíveis têm peso efetivo zero"
        elif mu is not None and sigma is not None:
            forecast_info = (0, "μ e σ informados manualmente")

        print_dashboard(args.date, mu, sigma, forecast_info, forecast_error,
                        comparison_forecasts, probability_distribution,
                        args.model_weights, effective_weights, prediction_interval,
                        backtest, args.contracts, forecast_quantiles, forecast_fetched_at)
        if not args.watch:
            break
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoramento encerrado.")
            break


if __name__ == "__main__":
    main()
