#!/usr/bin/env python3
"""Radar de temperatura SBGR + ladder Polymarket.

Os preços vêm da API pública. A distribuição de temperatura usa os membros do
ECMWF IFS e, quando aprovada pelo backtest histórico walk-forward de SBGR,
combina também GFS e ICON. Também é possível sobrescrever mu e sigma.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
import json
import math
import re
import sys
import time
from typing import Any

import requests
from model_engine import (
    Calibration,
    MODEL_CONFIG,
    calibrate_models,
    choose_temperature_strategy,
    empirical_bracket_probability,
    fetch_previous_run_daily_max,
    fetch_station_daily_max,
    summarize_distribution,
    weighted_member_distribution,
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
_CALIBRATION_CACHE: dict[int, tuple[float, Calibration]] = {}


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
    return data[0].get("markets", []) or []


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


def get_model_calibration(lead_days: int) -> Calibration:
    """Fetch and cache 35-day fixed-lead verification data for SBGR."""
    now = time.time()
    cached = _CALIBRATION_CACHE.get(lead_days)
    if cached and now - cached[0] < 6 * 60 * 60:
        return cached[1]
    if not 0 <= lead_days <= 7:
        result = Calibration(False, lead_days, 0, {}, {}, {}, {}, None, None,
                             "arquivo de verificação disponível apenas em D-0 a D-7")
        _CALIBRATION_CACHE[lead_days] = (now, result)
        return result

    observations = fetch_station_daily_max()
    historical: dict[str, dict[str, float]] = {}
    for name, config in MODEL_CONFIG.items():
        historical[name] = fetch_previous_run_daily_max(
            config["history"], lead_days
        )
    result = calibrate_models(observations, historical, lead_days)
    _CALIBRATION_CACHE[lead_days] = (time.time(), result)
    return result


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
                    calibration: Calibration | None = None) -> None:
    print("=" * 92)
    if mu is not None and sigma is not None:
        count, source = forecast_info or (0, "parâmetros informados manualmente")
        source_text = f"{source}; {count} membros" if count else source
        print(f"[PREVISÃO DA MÁXIMA] μ={mu:.2f} °C | σ={sigma:.2f} °C | {source_text}")
        if probability_distribution:
            print("  Probabilidades calculadas pela frequência ponderada dos membros; são estimativas, não garantias.")
        else:
            print("  Probabilidades derivadas de distribuição normal ajustada ao ensemble; não são garantias.")
        if calibration and calibration.enabled:
            print("  P(YES) e EV usam a mistura ECMWF/GFS/ICON aprovada no walk-forward.")
        elif probability_distribution:
            print("  P(YES) e EV usam o ensemble ECMWF; GFS e ICON ficam como comparação porque a mistura não foi aprovada.")
        else:
            print("  P(YES) e EV usam esta previsão principal; GFS e ICON são exibidos para comparação.")
    elif forecast_error:
        print(f"[PREVISÃO DA MÁXIMA] indisponível: {forecast_error}")
    for model_name, result in (comparison_forecasts or {}).items():
        if isinstance(result, tuple):
            model_mu, model_sigma, count, source = result
            print(f"[PREVISÃO {model_name}] μ={model_mu:.2f} °C | σ={model_sigma:.2f} °C | {source}; {count} membros")
        else:
            print(f"[PREVISÃO {model_name}] indisponível: {result}")
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
        columns = [("Faixa", 20), ("A.Y", 7), ("Spr.Y", 7), ("A.N", 7), ("Spr.N", 7), ("P(impl.)", 10), ("P(YES)", 10), ("EV YES", 10), ("EV% YES", 10), ("EV NO", 10), ("EV% NO", 10)]
        def render_row(values: list[str]) -> str:
            cells = [f"{values[0]:<20.20}"] + [f"{value:>{columns[i][1]}}" for i, value in enumerate(values[1:], 1)]
            return "| " + " | ".join(cells) + " |"
        table_rule = "-" * len(render_row([name for name, _ in columns]))
        print(table_rule)
        print(render_row([name for name, _ in columns]))
        print(table_rule)
        for market in markets:
            label = market.get("groupItemTitle") or market.get("question") or "Faixa sem nome"
            bid = as_price(market.get("bestBid"))
            ask = as_price(market.get("bestAsk"))
            bid_no = ask_no = None
            no_id = no_token_id(market)
            if no_id:
                try:
                    bid_no, ask_no = fetch_best_quotes(no_id)
                except requests.RequestException:
                    # Keep the rest of the ladder visible if the NO book is unavailable.
                    pass
            probability = (
                empirical_bracket_probability(str(label), probability_distribution)
                if probability_distribution
                else bracket_probability(str(label), mu, sigma) if mu is not None and sigma is not None else None
            )
            no_probability = 1.0 - probability if probability is not None else None
            schedule = market_fee_schedule(market)
            fee_yes = taker_fee_per_contract(ask, schedule) if ask is not None else None
            fee_no = taker_fee_per_contract(ask_no, schedule) if ask_no is not None else None
            ev_yes_net = probability - ask - fee_yes if probability is not None and ask is not None and fee_yes is not None else None
            ev_no_net = no_probability - ask_no - fee_no if no_probability is not None and ask_no is not None and fee_no is not None else None
            ev_yes_pct = ev_yes_net / (ask + fee_yes) * 100 if ev_yes_net is not None and ask is not None and fee_yes is not None and ask + fee_yes > 0 else None
            ev_no_pct = ev_no_net / (ask_no + fee_no) * 100 if ev_no_net is not None and ask_no is not None and fee_no is not None and ask_no + fee_no > 0 else None
            spread_yes = ask - bid if ask is not None and bid is not None else None
            spread_no = ask_no - bid_no if ask_no is not None and bid_no is not None else None
            implied = ((bid + ask) / 2) if bid is not None and ask is not None else (ask if ask is not None else bid)

            def cell(value: float | None, fmt: str) -> str:
                return format(value, fmt) if value is not None else "—"

            row = [str(label), cell(ask, ".3f"), cell(spread_yes, ".3f"), cell(ask_no, ".3f"), cell(spread_no, ".3f"), f"{cell(implied * 100 if implied is not None else None, '.3f')}%", f"{cell(probability * 100 if probability is not None else None, '.3f')}%", cell(ev_yes_net, "+.3f"), f"{cell(ev_yes_pct, '+.3f')}%", cell(ev_no_net, "+.3f"), f"{cell(ev_no_pct, '+.3f')}%"]
            print(render_row(row))
            print(table_rule)
        print("\n  B/A = best bid/ask; Spr. = Ask − Bid. O ask já incorpora o spread pago ao cruzar o livro;")
        print("  ele não é subtraído duas vezes. EV YES/NO e EV% já incluem o desconto da taxa taker do mercado.")
        print("  EV líquido = P(resultado) − Ask − taxa; EV% = EV líquido ÷ (Ask + taxa) × 100.")
        if probability_distribution:
            strategy = choose_temperature_strategy(
                markets,
                probability_distribution,
                lambda price, market: taker_fee_per_contract(price, market_fee_schedule(market)),
            )
            print("\n  SUGESTÃO DE TEMPERATURA (estimativa; sem execução de ordens):")
            if strategy:
                labels = " + ".join(item["label"] for item in strategy["items"])
                print(f"    {strategy['kind']}: {labels}")
                print(f"    Probabilidade de acerto conjunta estimada: {strategy['probability'] * 100:.2f}%")
                print(f"    Custo total estimado: ${strategy['cost']:.3f} | EV líquido: ${strategy['ev']:+.3f} | EV%: {strategy['roi'] * 100:+.2f}%")
                print("    Considera taxa taker disponível; exclui slippage e não garante lucro.")
            else:
                print("    Nenhuma faixa ou cesta adjacente apresentou EV líquido positivo com as cotações atuais.")
            if calibration:
                if calibration.enabled:
                    print(f"    Pesos validados em walk-forward D-{calibration.lead_days} ({calibration.sample_days} dias): "
                          + ", ".join(f"{name} {weight:.0%}" for name, weight in calibration.weights.items()))
                    if calibration.walkforward_ecmwf_rmse is not None and calibration.walkforward_blend_rmse is not None:
                        print(f"    RMSE fora da amostra: ECMWF {calibration.walkforward_ecmwf_rmse:.2f}°C → mistura {calibration.walkforward_blend_rmse:.2f}°C")
                else:
                    print(f"    Mistura desativada; P/EV usa ECMWF sem combinação: {calibration.reason}.")
                if calibration.sample_days and calibration.rmse:
                    metric_lines = []
                    for name in ("ECMWF", "GFS", "ICON"):
                        if name in calibration.rmse:
                            metric_lines.append(
                                f"{name}: viés {calibration.biases[name]:+.2f}°C, "
                                f"MAE {calibration.mae[name]:.2f}°C, "
                                f"RMSE {calibration.rmse[name]:.2f}°C"
                            )
                    if metric_lines:
                        print(f"    Verificação de máximas observadas em SBGR, D-{calibration.lead_days}, "
                              f"{calibration.sample_days} dias: " + " | ".join(metric_lines))
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
    args = parser.parse_args()
    if (args.mu is None) != (args.sigma is None):
        parser.error("informe --mu e --sigma juntos")
    if args.sigma is not None and args.sigma <= 0:
        parser.error("--sigma deve ser maior que zero")
    if args.interval < 60:
        parser.error("--interval deve ser de pelo menos 60 segundos para respeitar os serviços consultados")
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
                if model_name != "ECMWF":
                    comparison_forecasts[model_name] = model_summaries[model_name]
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                if model_name == "ECMWF" and mu is None:
                    forecast_error = str(exc)
                elif model_name != "ECMWF":
                    comparison_forecasts[model_name] = str(exc)

        probability_distribution: list[tuple[float, float]] | None = None
        calibration: Calibration | None = None
        if args.mu is None and args.sigma is None and "ECMWF" in member_sets:
            lead_days = max(0, (args.date - date.today()).days)
            try:
                calibration = get_model_calibration(lead_days)
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                calibration = Calibration(False, lead_days, 0, {}, {}, {}, {}, None, None,
                                           f"não foi possível obter o histórico: {exc}")
            if calibration.enabled and not all(name in member_sets for name in MODEL_CONFIG):
                calibration.enabled = False
                calibration.reason = "um ou mais ensembles atuais estão indisponíveis"
            probability_distribution = weighted_member_distribution(member_sets, calibration)
            if probability_distribution:
                mu, sigma = summarize_distribution(probability_distribution)
            else:
                mu, sigma, count, source = model_summaries["ECMWF"]
                forecast_info = (count, source)
            if calibration.enabled:
                weights_text = ", ".join(
                    f"{name} {weight:.0%}" for name, weight in calibration.weights.items()
                )
                forecast_info = (sum(len(values) for values in member_sets.values()),
                                 f"mistura multimodelo calibrada — {weights_text}")
            elif probability_distribution:
                count = len(member_sets["ECMWF"])
                forecast_info = (count, "ECMWF IFS ensemble (mistura não aprovada no backtest)")
        elif mu is not None and sigma is not None:
            forecast_info = (0, "μ e σ informados manualmente")

        print_dashboard(args.date, mu, sigma, forecast_info, forecast_error,
                        comparison_forecasts, probability_distribution, calibration)
        if not args.watch:
            break
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoramento encerrado.")
            break


if __name__ == "__main__":
    main()
