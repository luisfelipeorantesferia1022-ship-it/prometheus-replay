"""
The live agent, replayed as one continuous book through 2025.

Entry logic is trading_agent_v3.try_market_open_entry + scan_universe_and_rank
and paper_trading.select_option / deploy_one. Exit logic is
paper_trading.run_exit_engine_v3. Those functions call Tradier and Google
Sheets directly, so they are ported here line for line rather than imported;
every rule is annotated with where it comes from.

Two execution modes:

  copy  What the agent itself would have done and booked. Quotes are 15 minutes
        delayed (Tradier sandbox). Exit triggers use paper_trading's
        _calc_option_pnl (fixed delta 0.35 on the stock move, no decay). Entry
        fills at the chain mid; the sale is booked at the option's last trade,
        else its bid, exactly like _execute_sell.

  real  What the market would actually have paid for the same decisions logic.
        No delay. Triggers use the held option's real mid. Buys at the ask,
        sells at the bid.

Two selectors (who gets picked each morning):

  lstm     the live agent: per-ticker LSTM direction, must agree with SPY's
           20-day trend, |move| >= 1%, ranked by signal-to-noise
  spy_rsi  control: the sim's rule on the same whitelist. Direction from SPY's
           trend for every ticker, ranked by RSI score. No LSTM.

    python -m replay.agent --mode copy --selector lstm
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import asof, config, dbn
from .forecasts import earnings_soon
from .weekly import whitelist_timeline

ONE_MIN = pd.Timedelta(minutes=1)


# ─────────────────────────────────────────────────────────────────────────────
#  Black-Scholes helpers (delta for the chain; OPRA carries no greeks)
# ─────────────────────────────────────────────────────────────────────────────

def bs_price(S, K, T, r, sig, call):
    if T <= 0 or sig <= 0:
        return max(0.0, (S - K) if call else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if call:
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(price, S, K, T, r, call):
    lo, hi = 0.01, 5.0
    if price <= max(0.0, (S - K) if call else (K - S)) + 1e-6:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, r, mid, call) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def bs_delta(S, K, T, r, sig, call):
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    return float(norm.cdf(d1)) if call else float(abs(norm.cdf(d1) - 1.0))


# ─────────────────────────────────────────────────────────────────────────────
#  Book
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    id: int
    ticker: str
    direction: str
    opt_type: str          # CALL / PUT
    raw_symbol: str
    strike: float
    expiry: date
    entry_ts: pd.Timestamp
    entry_stock: float     # price passed to deploy_one (scan-time quote)
    option_cost: float     # per-share premium booked as cost basis
    entry_mid: float
    entry_ask: float
    contracts: int
    total_spent: float
    entry_delta: float
    snr: float | None = None
    tom_pct: float | None = None
    rank_position: int = 0
    n_candidates: int = 0
    exit_ts: pd.Timestamp | None = None
    exit_reason: str = ""
    exit_price: float | None = None
    realized_pnl: float | None = None
    peak_pnl_pct: float | None = None


@dataclass
class Book:
    balance: float = config.STARTING_BALANCE
    open: list = field(default_factory=list)
    closed: list = field(default_factory=list)
    peaks: dict = field(default_factory=dict)
    next_id: int = 1
    daily: list = field(default_factory=list)
    entry_log: list = field(default_factory=list)


class Replay:
    def __init__(self, mode: str, selector: str, variant: str):
        assert mode in ("copy", "real") and selector in ("lstm", "spy_rsi")
        self.mode, self.selector, self.variant = mode, selector, variant
        self.delay = pd.Timedelta(minutes=config.COPY_QUOTE_DELAY_MIN if mode == "copy" else 0)
        self.book = Book()
        self.whitelists = whitelist_timeline(variant)
        self.day_to_asof = {}
        for a in self.whitelists:
            for d in asof.week_trading_days(a):
                self.day_to_asof[d] = a
        self.forecasts = {}
        if selector == "lstm":
            from .forecasts import load
            self.forecasts = load()
        self._opt_cache: dict[str, pd.DataFrame] = {}
        self._trade_cache: dict[str, pd.DataFrame] = {}

    # ── Prices ─────────────────────────────────────────────────────────────
    def stock_price(self, ticker: str, ts: pd.Timestamp, delayed=True) -> float | None:
        """fetch_live_price: last trade as of ts (minus the sandbox delay)."""
        t = ts - (self.delay if delayed else pd.Timedelta(0))
        bars = dbn.equity_bars(ticker, ts.date(), purpose="agent_bars")
        done = bars[bars.index + ONE_MIN <= t] if not bars.empty else bars
        if not done.empty:
            return float(done["close"].iloc[-1])
        return asof.unadjusted_close(ticker, asof.prev_trading_day(ts.date()))

    def _quotes_for(self, tr: Trade) -> pd.DataFrame:
        if tr.raw_symbol not in self._opt_cache:
            q = dbn.option_quotes([tr.raw_symbol],
                                  pd.Timestamp(f"{tr.entry_ts.date()} 09:00"),
                                  pd.Timestamp(f"{tr.expiry} 16:15"),
                                  purpose="held_option_quotes")
            self._opt_cache[tr.raw_symbol] = q.set_index("ts_et")[["bid", "ask"]] if not q.empty \
                else pd.DataFrame(columns=["bid", "ask"])
        return self._opt_cache[tr.raw_symbol]

    def option_quote(self, tr: Trade, ts: pd.Timestamp, delayed: bool):
        q = self._quotes_for(tr)
        t = ts - (self.delay if delayed else pd.Timedelta(0))
        q = q[q.index <= t]
        if q.empty:
            return None, None
        r = q.iloc[-1]
        bid, ask = float(r["bid"]), float(r["ask"])
        return (bid if bid > 0 and np.isfinite(bid) else None,
                ask if ask > 0 and np.isfinite(ask) else None)

    def option_last_trade(self, tr: Trade, ts: pd.Timestamp) -> float | None:
        """
        Last trade before the (delayed) exit time. Only the exit day is bought:
        busy contracts print thousands of trades over a hold, and the sale only
        needs the most recent one. No print that day -> caller uses the bid,
        as _execute_sell does when `last` is missing.
        """
        cut = ts - self.delay
        if cut.time() < pd.Timestamp("09:31").time():
            # delayed quote still shows the prior session: its last hour
            p = asof.prev_trading_day(cut.date())
            start, end = pd.Timestamp(f"{p} 15:00"), pd.Timestamp(f"{p} 16:15")
        else:
            start, end = pd.Timestamp(f"{cut.date()} 09:30"), cut + ONE_MIN
        t = dbn.option_trades(tr.raw_symbol, start, end, purpose="exit_option_trades")
        if t.empty:
            return None
        t = t[t["ts_et"] <= cut]
        return float(t["price"].iloc[-1]) if not t.empty else None

    # ── SPY trend (get_market_trend: Tradier daily, through prior close) ──
    def spy_trend(self, d: date) -> str:
        p = asof.prev_trading_day(d)
        spy = asof.split_adjusted("SPY", p - timedelta(days=90), p + timedelta(days=1))
        if spy.empty or len(spy) < 21:
            return "Neutral"
        close = spy["Close"]
        return "Bullish" if float(close.iloc[-1]) > float(close.rolling(20).mean().iloc[-1]) else "Bearish"

    def rsi(self, ticker: str, d: date, period: int = 14) -> float | None:
        p = asof.prev_trading_day(d)
        df = asof.split_adjusted(ticker, p - timedelta(days=period * 3 + 10), p + timedelta(days=1))
        if df.empty or len(df) < period + 1:
            return None
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean().replace(0, 1e-10)
        v = float((100 - 100 / (1 + gain / loss)).iloc[-1])
        return None if v != v else v

    # ── Scan (scan_universe_and_rank) ──────────────────────────────────────
    def scan(self, d: date, universe: list[str], spy_trend: str, clock: pd.Timestamp):
        passed, skips = [], []
        for ticker in universe:
            if earnings_soon(ticker, d):
                skips.append((ticker, "earnings within 7 days"))
                clock += pd.Timedelta(seconds=config.SCAN_SECONDS_EARNINGS_SKIP)
                continue
            price = self.stock_price(ticker, clock)
            if price is None:
                skips.append((ticker, "could not fetch live price"))
                continue
            if self.selector == "spy_rsi":
                clock += pd.Timedelta(seconds=config.SCAN_SECONDS_EARNINGS_SKIP)
                r = self.rsi(ticker, d)
                if r is None:
                    skips.append((ticker, "RSI failed"))
                    continue
                direction = "Bearish" if spy_trend == "Bearish" else "Bullish"
                passed.append({"ticker": ticker, "direction": direction, "current_price": price,
                               "rank": (100 - r) if direction == "Bullish" else r,
                               "snr": None, "tom_pct": None})
                continue

            clock += pd.Timedelta(seconds=config.SCAN_SECONDS_PER_TICKER)
            fc = self.forecasts.get((d, ticker))
            if not fc or not fc.get("ok"):
                skips.append((ticker, "LSTM failed / returned None"))
                continue
            tomorrow = float(fc["tomorrow_price"])
            if price > 0 and abs(tomorrow - price) / price > 0.60:
                skips.append((ticker, "forecast deviates >60%"))
                continue
            tom_pct = (tomorrow - price) / price * 100
            mae_pct = float(fc["daily_mae"]) / price * 100
            snr = abs(tom_pct) / max(mae_pct, 0.1)
            direction = "Bullish" if tom_pct > 0 else "Bearish"
            if abs(tom_pct) < config.ENTRY_MIN_LSTM_MOVE_PCT:
                skips.append((ticker, f"predicted move {tom_pct:+.2f}% below 1%"))
                continue
            if spy_trend != "Neutral" and direction != spy_trend:
                skips.append((ticker, f"SPY filter: {direction} vs {spy_trend}"))
                continue
            passed.append({"ticker": ticker, "direction": direction, "current_price": price,
                           "rank": snr, "snr": snr, "tom_pct": tom_pct})
        passed.sort(key=lambda c: c["rank"], reverse=True)
        return passed, skips, clock

    # ── select_option ──────────────────────────────────────────────────────
    def select_option(self, ticker, direction, budget, ts: pd.Timestamp):
        if budget < config.SELECT_OPTION_MIN_BUDGET:
            return None, "budget below $50"
        is_call = direction == "Bullish"
        current = self.stock_price(ticker, ts)
        if current is None:
            return None, "no live price"
        defs = dbn.option_definitions(ticker, ts.date())
        if defs.empty:
            return None, "no expirations"
        now = ts.to_pydatetime()
        target = fallback = None
        for exp in sorted(defs["expiration"].unique()):
            days_out = (datetime.combine(exp, datetime.min.time()) - now).days
            if days_out < 12 or days_out > 24:
                continue
            if fallback is None:
                fallback = exp
            if 14 <= days_out <= 21:
                target = exp
                break
        target = target or fallback
        if target is None:
            return None, "no expiry in 12-24 DTE"

        cls = "C" if is_call else "P"
        opts = defs[(defs["expiration"] == target) & (defs["instrument_class"] == cls)].copy()
        if opts.empty:
            return None, "empty chain"
        opts["strike"] = opts["strike_price"].astype(float)
        if is_call:
            lo, hi = current * 1.00, current * 1.03
        else:
            lo, hi = current * 0.97, current * 1.001
        valid = opts[(opts["strike"] >= lo) & (opts["strike"] <= hi)]
        if valid.empty:
            valid = opts[(opts["strike"] >= current) & (opts["strike"] <= current * 1.05)] if is_call \
                else opts[(opts["strike"] >= current * 0.95) & (opts["strike"] <= current * 1.001)]
            if valid.empty:
                return None, "no strike in range"
        ideal = current * 1.01 if is_call else current * 0.99
        valid = valid.assign(_d=(valid["strike"] - ideal).abs()).sort_values("_d", kind="stable")

        start = pd.Timestamp(f"{asof.prev_trading_day(ts.date())} 15:30")
        q = dbn.option_quotes(valid["raw_symbol"].tolist(), start, ts + ONE_MIN, purpose="chain_quotes")
        qt = ts - self.delay
        last_q = (q[q["ts_et"] <= qt].groupby("symbol" if "symbol" in q.columns else "raw_symbol").last()
                  if not q.empty else pd.DataFrame())

        T = max((datetime.combine(target, datetime.min.time()) + timedelta(hours=16) - now)
                .total_seconds() / (365 * 86400), 1 / 365)
        scored = []
        for _, o in valid.iterrows():
            sym = o["raw_symbol"]
            if sym not in last_q.index:
                continue
            bid, ask = float(last_q.at[sym, "bid"]), float(last_q.at[sym, "ask"])
            if not (bid > 0 and ask > 0):
                continue
            mid = (bid + ask) / 2
            if (ask - bid) / mid > config.MAX_SPREAD_PCT:
                continue
            iv = implied_vol(mid, current, o["strike"], T, config.RISK_FREE, is_call)
            if iv is not None:
                delta = bs_delta(current, o["strike"], T, config.RISK_FREE, iv, is_call)
            else:
                delta = max(0.05, 0.50 - abs(o["strike"] - current) / current * 2.0)
            if delta < config.MIN_DELTA:
                continue
            value_score = delta * current * config.ASSUMED_MOVE_PCT / mid
            if mid * 100 <= budget:
                scored.append((value_score, o, mid, bid, ask, delta))
        if not scored:
            return None, "no contract within budget after scoring"
        scored.sort(key=lambda x: x[0], reverse=True)
        vs, o, mid, bid, ask, delta = scored[0]
        contracts = max(1, int(budget / (mid * 100)))
        fill = mid if self.mode == "copy" else ask
        if (target - ts.date()).days < config.ENTRY_MIN_DTE:
            return None, "DTE < 3"
        return {"raw_symbol": o["raw_symbol"], "strike": float(o["strike"]), "expiry": target,
                "opt_type": "CALL" if is_call else "PUT", "mid": mid, "ask": ask,
                "fill": fill, "contracts": contracts,
                "total_cost": round(contracts * fill * 100, 2), "delta": delta,
                "current": current}, None

    # ── Exit engine (run_exit_engine_v3) ───────────────────────────────────
    def pnl_pct(self, tr: Trade, ts: pd.Timestamp) -> float | None:
        if self.mode == "copy":
            cur = self.stock_price(tr.ticker, ts)
            if cur is None:
                return None
            move = cur - tr.entry_stock
            if tr.direction == "Bearish":
                move = -move
            exit_prem = max(0.0, tr.option_cost + move * config.COPY_FIXED_DELTA)
            pnl = round((exit_prem - tr.option_cost) * 100 * tr.contracts, 2)
            return round(pnl / tr.total_spent * 100, 1) if tr.total_spent > 0 else 0.0
        bid, ask = self.option_quote(tr, ts, delayed=False)
        if bid is None or ask is None:
            return None
        return ((bid + ask) / 2 * 100 * tr.contracts - tr.total_spent) / tr.total_spent * 100

    def exit_value(self, tr: Trade, ts: pd.Timestamp) -> float:
        """Per-share sale price."""
        if self.mode == "copy":
            last = self.option_last_trade(tr, ts)
            if last is not None:
                return last
            bid, _ = self.option_quote(tr, ts, delayed=True)
            if bid is not None:
                return bid
            cur = self.stock_price(tr.ticker, ts) or tr.entry_stock
            move = (cur - tr.entry_stock) * (-1 if tr.direction == "Bearish" else 1)
            return max(0.0, tr.option_cost + move * config.COPY_FIXED_DELTA)
        bid, _ = self.option_quote(tr, ts, delayed=False)
        if bid is not None:
            return bid
        cur = self.stock_price(tr.ticker, ts, delayed=False) or tr.entry_stock
        return max(0.0, (cur - tr.strike) if tr.opt_type == "CALL" else (tr.strike - cur))

    def close(self, tr: Trade, ts: pd.Timestamp, reason: str, peak: float):
        px = self.exit_value(tr, ts)
        realized = round((px - tr.option_cost) * 100 * tr.contracts, 2)
        tr.exit_ts, tr.exit_reason, tr.exit_price = ts, reason, px
        tr.realized_pnl, tr.peak_pnl_pct = realized, peak
        self.book.balance = round(self.book.balance + tr.total_spent + realized, 2)
        self.book.open.remove(tr)
        self.book.closed.append(tr)
        self.book.peaks.pop(tr.id, None)

    def exit_tick(self, ts: pd.Timestamp):
        for tr in list(self.book.open):
            p = self.pnl_pct(tr, ts)
            force = ts.date() >= tr.expiry - timedelta(days=1) and ts.hour >= config.EXIT_FORCE_HOUR_ET
            if p is None:
                if force:   # no quote at all on the force-exit day: settle anyway
                    self.close(tr, ts, "Thursday-before-expiry force-exit (no quote)",
                               self.book.peaks.get(tr.id, float("nan")))
                continue
            peak = max(self.book.peaks.get(tr.id, p), p)
            self.book.peaks[tr.id] = peak
            if p <= config.EXIT_STOP_LOSS_PCT:
                self.close(tr, ts, "Stop loss", peak)
            elif p >= config.EXIT_HARD_CEILING_PCT:
                self.close(tr, ts, "Hard ceiling", peak)
            elif force:
                self.close(tr, ts, "Thursday-before-expiry force-exit", peak)
            elif peak >= config.EXIT_TRAIL_TRIGGER_PCT and p <= peak - config.EXIT_TRAIL_GIVEBACK_PCT:
                self.close(tr, ts, "Trail stop", peak)

    # ── Entry (try_market_open_entry) ──────────────────────────────────────
    def try_entry(self, d: date, clock: pd.Timestamp) -> pd.Timestamp:
        a = self.day_to_asof.get(d)
        universe = self.whitelists.get(a, []) if a else []
        log = {"date": d.isoformat(), "universe": " ".join(universe)}
        if not universe or d > pd.Timestamp(config.AGENT_LAST_ENTRY_DAY).date():
            log["result"] = "empty universe" if universe == [] else "after replay window"
        elif d.weekday() not in config.ENTRY_ALLOWED_WEEKDAYS:
            log["result"] = "weekday gate"
        elif config.MAX_OPEN_POSITIONS - len(self.book.open) <= 0:
            log["result"] = "max positions"
        elif round(self.book.balance * config.BASE_POSITION_PCT, 2) < config.MIN_TRADE_BUDGET:
            log["result"] = "budget below minimum"
        else:
            slots = config.MAX_OPEN_POSITIONS - len(self.book.open)
            dbn.prefetch_equity_day(d, universe + [t.ticker for t in self.book.open], "agent_bars")
            trend = self.spy_trend(d)
            passed, skips, clock = self.scan(d, universe, trend, clock)
            log.update({"spy_trend": trend, "passed": " ".join(c["ticker"] for c in passed),
                        "skips": "; ".join(f"{t}: {r}" for t, r in skips)})
            deployed, held = [], {t.ticker for t in self.book.open}
            for rank, c in enumerate(passed, 1):
                if c["ticker"] in held:
                    continue
                budget = round(self.book.balance * config.BASE_POSITION_PCT, 2)
                if budget < config.MIN_TRADE_BUDGET:
                    break
                opt, why = self.select_option(c["ticker"], c["direction"], budget, clock)
                if opt is None:
                    skips.append((c["ticker"], f"select_option: {why}"))
                    continue
                tr = Trade(id=self.book.next_id, ticker=c["ticker"], direction=c["direction"],
                           opt_type=opt["opt_type"], raw_symbol=opt["raw_symbol"],
                           strike=opt["strike"], expiry=opt["expiry"], entry_ts=clock,
                           entry_stock=c["current_price"], option_cost=opt["fill"],
                           entry_mid=opt["mid"], entry_ask=opt["ask"],
                           contracts=opt["contracts"], total_spent=opt["total_cost"],
                           entry_delta=opt["delta"], snr=c["snr"], tom_pct=c["tom_pct"],
                           rank_position=rank, n_candidates=len(passed))
                self.book.next_id += 1
                self.book.balance = round(self.book.balance - tr.total_spent, 2)
                self.book.open.append(tr)
                deployed.append(tr.ticker)
                held.add(tr.ticker)
                if len(deployed) >= slots:
                    break
            log["deployed"] = " ".join(deployed)
            log["result"] = "deployed" if deployed else "all candidates exhausted" if passed \
                else "no candidates passed filters"
            log["select_skips"] = "; ".join(f"{t}: {r}" for t, r in skips if r.startswith("select_option"))
        log["entry_done_at"] = str(clock.time())
        self.book.entry_log.append(log)
        return clock

    # ── Day loop ───────────────────────────────────────────────────────────
    def mark(self, d: date, ts: pd.Timestamp) -> float:
        mv = 0.0
        for tr in self.book.open:
            bid, ask = self.option_quote(tr, ts, delayed=False)
            if bid is not None and ask is not None:
                mv += (bid + ask) / 2 * 100 * tr.contracts
            else:
                mv += tr.total_spent
        return mv

    def run_day(self, d: date):
        sched = asof.nyse_schedule().loc[d]
        open_ts = pd.Timestamp(sched["open_et"])
        last_tick = pd.Timestamp(f"{d} 15:59")
        a = self.day_to_asof.get(d)
        could_enter = bool(a and self.whitelists.get(a)) and d.weekday() in config.ENTRY_ALLOWED_WEEKDAYS
        if not self.book.open and not could_enter:
            self.book.daily.append({"date": d, "cash": self.book.balance, "open_mv": 0.0,
                                    "equity": self.book.balance, "n_open": 0})
            return
        if self.book.open:
            dbn.prefetch_equity_day(d, [t.ticker for t in self.book.open], "agent_bars")
        self.exit_tick(open_ts)                       # first tick: exits before entry
        clock = self.try_entry(d, open_ts)            # once per day
        ts = max(open_ts + ONE_MIN, clock.ceil("min"))
        while ts <= last_tick and self.book.open:
            self.exit_tick(ts)
            ts += ONE_MIN
        mv = self.mark(d, pd.Timestamp(f"{d} 15:59"))
        self.book.daily.append({"date": d, "cash": self.book.balance, "open_mv": round(mv, 2),
                                "equity": round(self.book.balance + mv, 2),
                                "n_open": len(self.book.open)})

    def run(self):
        days = asof.trading_days(min(self.day_to_asof), config.AGENT_LAST_EXIT_DAY)
        last_entry = pd.Timestamp(config.AGENT_LAST_ENTRY_DAY).date()
        for d in days:
            if d > last_entry and not self.book.open:
                break
            try:
                self.run_day(d)
            except dbn.BudgetExceeded as e:
                print(f"[agent] STOPPED on {d}: {e}", flush=True)
                self.stopped = f"{d}: {e}"
                break
            except Exception as e:
                # Keep what was done so far for diagnosis, then fail loudly.
                self.stopped = f"{d}: crashed with {type(e).__name__}: {e}"
                self.write()
                raise
            if d.day == 1 or d.weekday() == 4:
                print(f"[agent {self.variant} {self.mode}/{self.selector}] {d} cash ${self.book.balance:,.2f} "
                      f"open {len(self.book.open)} closed {len(self.book.closed)} "
                      f"| databento ${dbn.ledger()['spent_usd']:.2f}", flush=True)
        return self.write()

    # ── Output ─────────────────────────────────────────────────────────────
    def write(self) -> dict:
        od = config.RESULTS_DIR / "agent" / self.variant / f"{self.selector}_{self.mode}"
        od.mkdir(parents=True, exist_ok=True)
        trades = pd.DataFrame([asdict(t) for t in self.book.closed + self.book.open])
        trades.to_csv(od / "trades.csv", index=False)
        daily = pd.DataFrame(self.book.daily)
        daily.to_csv(od / "daily_equity.csv", index=False)
        pd.DataFrame(self.book.entry_log).to_csv(od / "entry_log.csv", index=False)
        s = summarize(trades, daily)
        s.update({"variant": self.variant, "mode": self.mode, "selector": self.selector,
                  "stopped_early": getattr(self, "stopped", None),
                  "databento_spent_usd": dbn.ledger()["spent_usd"]})
        (od / "summary.json").write_text(json.dumps(s, indent=2, default=str))
        print(json.dumps(s, indent=2, default=str), flush=True)
        return s


def summarize(trades: pd.DataFrame, daily: pd.DataFrame) -> dict:
    closed = trades[trades["realized_pnl"].notna()] if len(trades) else trades
    out = {"start_balance": config.STARTING_BALANCE}
    if len(daily):
        eq = daily["equity"].astype(float)
        out["end_equity"] = round(float(eq.iloc[-1]), 2)
        out["return_pct"] = round((eq.iloc[-1] / config.STARTING_BALANCE - 1) * 100, 2)
        out["max_drawdown_pct"] = round(float(((eq / eq.cummax()) - 1).min() * 100), 2)
        spy = asof.split_adjusted("SPY", daily["date"].iloc[0], pd.Timestamp(daily["date"].iloc[-1]) + pd.Timedelta(days=1))
        if not spy.empty:
            out["spy_return_pct_same_window"] = round((spy["Close"].iloc[-1] / spy["Open"].iloc[0] - 1) * 100, 2)
    if len(closed):
        pnl = closed["realized_pnl"].astype(float)
        out.update({
            "n_trades": int(len(closed)),
            "win_rate_pct": round(float((pnl > 0).mean() * 100), 1),
            "total_realized_pnl": round(float(pnl.sum()), 2),
            "avg_win": round(float(pnl[pnl > 0].mean()), 2) if (pnl > 0).any() else 0.0,
            "avg_loss": round(float(pnl[pnl < 0].mean()), 2) if (pnl < 0).any() else 0.0,
            "exit_reasons": closed["exit_reason"].value_counts().to_dict(),
            "by_ticker": closed.groupby("ticker")["realized_pnl"].agg(["count", "sum"]).round(2)
                         .to_dict(orient="index"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["copy", "real"], required=True)
    ap.add_argument("--selector", choices=["lstm", "spy_rsi"], default="lstm")
    ap.add_argument("--variant", default=next(iter(config.SIM_MIN_TRADES_VARIANTS)))
    a = ap.parse_args()
    Replay(a.mode, a.selector, a.variant).run()


if __name__ == "__main__":
    main()
