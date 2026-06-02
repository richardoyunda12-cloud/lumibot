"""
═══════════════════════════════════════════════════════════════════════
  CITADEL BASELINE v1 — STABLE
═══════════════════════════════════════════════════════════════════════
Momentum strategy untuk US stocks via Lumibot.

KONFIGURASI BASELINE (terkunci):
- 1x NO LEVERAGE (leverage decay terbukti merugikan → use_leverage=False)
- AI Agent OFF (deterministik; committee bisa diaktifkan terpisah)
- Bear defense = CASH (exit semua saat BEAR, preserve capital)
- 3 lapis exit:
    Lapis 1: momentum break per-saham (roc20<0 + harga<SMA20) — cut losers cepat
    Lapis 2: trailing stop ATR
    Lapis 3: regime BEAR → cash
- Universe: cache/weekly_universe_history_v32.csv (222 saham, scanner v32)

HASIL VALIDASI (cohort 2015-2025, 1x):
- Mengalahkan SPY di SEMUA 11 cohort (universe bersih MAUPUN diracuni jebakan)
- CAGR realistis ~20-25% (setelah koreksi survivorship bias)
- Max drawdown ~20% (lebih kecil dari SPY)
- Edge terbukti NYATA: lolos uji momentum-trap (PTON/ZM/ROKU dll)

CARA PAKAI:
1. python generate_universe_v32.py   (refresh universe; bulanan untuk live)
2. python citadel_baseline_v1.py      (atau panggil via runner backtest)

CATATAN LIVE TRADING:
- Untuk live, ganti universe statis dengan MTUM holdings + screening likuiditas
  (dinamis, menangkap pendatang baru). Lihat fundamental_live.py untuk tilt
  fundamental (earnings/revenue growth) — LIVE-ONLY, jangan untuk backtest.
═══════════════════════════════════════════════════════════════════════
"""

import math, os
import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
from datetime import datetime, date as date_type
from typing import Optional
from lumibot.strategies.strategy import Strategy
from lumibot.backtesting import YahooDataBacktesting
from lumibot.entities import TradingFee

# AI Agent Committee (Layer 2) - Tetap di-import meski tidak dipanggil di mode deterministik
from agent_committee import AgentCommittee
from news_proxy import detect_catalyst

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — safe scalar extraction
# ─────────────────────────────────────────────────────────────────────────────
def _s(x) -> float:
    if isinstance(x, pd.Series):
        x = x.iloc[0]
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)

# ─────────────────────────────────────────────────────────────────────────────
# [S3] DYNAMIC SCORE WEIGHTS PER REGIME
# ─────────────────────────────────────────────────────────────────────────────
REGIME_SCORE_WEIGHTS = {
    "BULL":  (0.30, 0.25, 0.15, 0.20, 0.10),
    "TRANS": (0.20, 0.30, 0.25, 0.15, 0.10),
    "BEAR":  (0.10, 0.35, 0.30, 0.10, 0.15),
}

def _norm(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if (mx - mn) > 1e-6 else pd.Series(0.5, index=s.index)

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY — V8S30 (DETERMINISTIC)
# ─────────────────────────────────────────────────────────────────────────────
class CitadelLevelQuantV8S30(Strategy):
    parameters = {
        "benchmark_asset":      "SPY",
        "cash_reserve":         0.02,
        "entry_score_min":      2,
        "roc_period":           20,
        "adx_min":              20,
        "max_drawdown_limit":   0.25,
        "cooldown_days":        3,
        "amnesia_days":         60,
        "buy_fee_pct":          0.0005,
        "sell_fee_pct":         0.0005,
        "quality_gate_window":  20,
        "max_per_sector":       3,
        
        # ── V8S30: AI DIMATIKAN (DETERMINISTIC MODE) ──
        "use_agent":            False, 
        "use_macro":            False,
        "universe_csv":         "cache/weekly_universe_history_v32.csv", # <-- BACA V32
        # ── AI POSITION SIZING (layer baru) ──
        "use_ai_sizing":        False,  # True = AI atur conviction/size_mult
        "log_decisions":        True,   # tulis log keputusan detail
        "run_name":             "citadel",
        # ── TELEGRAM (live-only) ──
        "use_telegram":         False,  # True hanya saat live/paper
        "telegram_level":       "all",  # "all" | "actions" | "summary"
        
        # ── TIER 1 PARAMETERS ──
        "leverage_bull":        1.5,
        "leverage_trans":       1.0,
        "leverage_bear":        0.0,
        "use_leverage":         False,  # 1x — tanpa leverage, tanpa margin
        "use_bear_defense":     True,
        "bear_mode":            "cash",  # "cash" | "inverse"
        "inverse_etf":          "SH",   
        "inverse_min_hold_weeks": 3,    
        "bear_inverse_alloc":   0.50,   
        
        # ── TIER 2 PARAMETERS ──
        "use_vol_targeting":    True,
        "target_portfolio_vol": 0.15,   
        "use_crash_protection": True,
        "crash_vix_proxy":      0.04,   
    }

    def initialize(self):
        self.sleeptime = "1D"
        p = self.parameters
        self.cash_reserve         = p["cash_reserve"]
        self.entry_score_min      = p["entry_score_min"]
        self.roc_period           = p["roc_period"]
        self.adx_min              = p["adx_min"]
        self.cooldown_days        = p["cooldown_days"]
        self.max_drawdown_limit   = p["max_drawdown_limit"]
        self.amnesia_days         = p["amnesia_days"]
        self.quality_gate_window  = p["quality_gate_window"]
        self.max_per_sector       = p["max_per_sector"]

        self.use_leverage     = p["use_leverage"]
        self.leverage_bull    = p["leverage_bull"]
        self.leverage_trans   = p["leverage_trans"]
        self.leverage_bear    = p["leverage_bear"]
        self.use_bear_defense = p["use_bear_defense"]
        self.bear_mode        = p["bear_mode"]
        self.inverse_etf      = p["inverse_etf"]
        self.inverse_min_hold_weeks = p["inverse_min_hold_weeks"]
        self.vars_inverse_entry_week = None
        self.bear_inverse_alloc = p["bear_inverse_alloc"]
        self.current_leverage = 1.0   
        
        self.use_vol_targeting    = p["use_vol_targeting"]
        self.target_portfolio_vol = p["target_portfolio_vol"]
        self.use_crash_protection = p["use_crash_protection"]
        self.crash_vix_proxy      = p["crash_vix_proxy"]
        self.vars_crash_active    = False
        self.use_macro            = p["use_macro"]

        # ── AI Sizing + Decision Logging ──
        self.use_ai_sizing = p.get("use_ai_sizing", False)
        self.log_decisions = p.get("log_decisions", True)
        self.ai_committee  = None
        self.dlogger       = None
        if self.use_ai_sizing:
            try:
                from ai_sizing import AISizingCommittee
                _is_bt = getattr(self, "is_backtesting", False)
                self.ai_committee = AISizingCommittee(verbose=False, live_mode=(not _is_bt))
            except Exception as e:
                self.log_message(f"⚠️ AI sizing tidak tersedia: {e}")
                self.use_ai_sizing = False
        if self.log_decisions:
            try:
                from decision_logger import DecisionLogger
                self.dlogger = DecisionLogger(run_name=p.get("run_name", "citadel"))
            except Exception as e:
                self.log_message(f"⚠️ Logger tidak tersedia: {e}")
        self.vars.last_sizing = {}   # cache size_mult per minggu

        # ── Telegram notifier (live-only) ──
        # Deteksi backtest: kalau is_backtesting True → telegram dimatikan.
        self.notifier = None
        use_tg = p.get("use_telegram", False)
        is_bt = getattr(self, "is_backtesting", False)
        try:
            from telegram_notifier import TelegramNotifier
            self.notifier = TelegramNotifier(
                is_live=(use_tg and not is_bt),
                level=p.get("telegram_level", "all"),
            )
        except Exception as e:
            self.log_message(f"⚠️ Telegram tidak tersedia: {e}")

        self.vars.active_universe = []
        self.vars.last_scan_week  = None   
        self.vars.peak_portfolio  = self.portfolio_value
        self.vars.days_in_dd      = 0
        self.vars.regime          = "BULL"
        self.vars.sym             = {}
        self.vars.score_history   = {"BULL": [], "TRANS": [], "BEAR": []}

        # AI Committee Setup (Dinonaktifkan via parameter)
        self.use_agent = p["use_agent"]
        self.vars.weekly_verdict = None   
        if self.use_agent:
            self.committee = AgentCommittee(cache_dir="cache/agent_verdicts", verbose=True)
            self.log_message("🤖 AI Agent Committee aktif")
        else:
            self.committee = None
            self.log_message("📊 Mode deterministik (AI OFF) - 100% Quant Engine")

        self.min_per_instrument = 0.15
        self.max_per_instrument = 0.25
        self.current_atr_mult   = 2.0

        # Load CSV
        self.bt_universe_df = None
        csv_path = p["universe_csv"]
        if os.path.exists(csv_path):
            self.bt_universe_df = pd.read_csv(csv_path, parse_dates=["week_start"])
            self.log_message(f"✅ Universe loaded: {csv_path} ({len(self.bt_universe_df)} rows)")
        else:
            self.log_message("⚠️ CSV universe TIDAK DITEMUKAN!")

    def _init_sym_state(self):
        return {"entry_price": None, "entry_date": None, "highest": None, "last_exit_date": None, "atr": 0.0, "sector": "Unknown"}

    def _st(self, sym):
        if sym not in self.vars.sym:
            self.vars.sym[sym] = self._init_sym_state()
        return self.vars.sym[sym]

    def _update_regime(self):
        bars = self.get_historical_prices("SPY", 220, "day")
        if not bars or bars.df is None or len(bars.df) < 55: return

        c     = bars.df["close"]
        price = _s(c.iloc[-1])
        sma50 = _s(ta.sma(c, length=50).iloc[-1])
        sma200= _s(ta.sma(c, length=200).iloc[-1])

        if self.use_crash_protection:
            daily_ret = c.pct_change().dropna()
            recent_vol = float(daily_ret.iloc[-10:].std()) if len(daily_ret) >= 10 else 0.0
            was_active = self.vars_crash_active
            self.vars_crash_active = recent_vol >= self.crash_vix_proxy
            if self.vars_crash_active and not was_active:
                self.log_message(f"⚡ CRASH PROTECTION aktif (vol harian {recent_vol:.1%})")
            elif not self.vars_crash_active and was_active:
                self.log_message("✅ Crash protection nonaktif (vol normal)")

        if price > sma50:
            if self.vars.regime != "BULL":
                self.log_message("🟢 REGIME → BULL (Aggressive Sizing)")
                if self.notifier: self.notifier.notify_regime(self.vars.regime, "BULL")
            self.vars.regime        = "BULL"
            self.min_per_instrument = 0.15
            self.max_per_instrument = 0.25
            self.current_atr_mult   = 2.0
            self.current_leverage   = self.leverage_bull if self.use_leverage else 1.0

        elif sma200 < price <= sma50:
            if self.vars.regime != "TRANS":
                self.log_message("🟡 REGIME → TRANS (Defensive Sizing)")
                if self.notifier: self.notifier.notify_regime(self.vars.regime, "TRANS")
            self.vars.regime        = "TRANS"
            self.min_per_instrument = 0.10
            self.max_per_instrument = 0.15
            self.current_atr_mult   = 1.5
            self.current_leverage   = self.leverage_trans if self.use_leverage else 1.0

        else:
            if self.vars.regime != "BEAR":
                self.log_message("🔴 REGIME → BEAR (No long entry)")
                if self.notifier: self.notifier.notify_regime(self.vars.regime, "BEAR")
            self.vars.regime        = "BEAR"
            self.min_per_instrument = 0.0
            self.max_per_instrument = 0.0
            self.current_atr_mult   = 1.0
            self.current_leverage   = self.leverage_bear if self.use_leverage else 0.0

    def _build_snapshot(self, sym):
        if not sym or sym == "CASH": return None
        bars = self.get_historical_prices(sym, 100, "day")
        if not bars or bars.df is None or len(bars.df) < 40: return None
        df = bars.df.copy().dropna(subset=["high", "low", "close"])
        if len(df) < 40: return None

        df["SMA_50"] = ta.sma(df["close"], length=50)
        df["EMA_20"] = ta.ema(df["close"], length=20)
        adx_df       = ta.adx(df["high"], df["low"], df["close"], length=14)
        df["ADX"]    = adx_df.get("ADX_14") if adx_df is not None else None
        df["ATR"]    = ta.atr(df["high"], df["low"], df["close"], length=14)

        d       = df.iloc[-1]
        atr_val = round(_s(d.get("ATR") or d["close"] * 0.02), 2)
        self._st(sym)["atr"] = atr_val

        return {
            "sym":          sym,
            "price":        round(_s(d["close"]), 2),
            "weekly_trend": "UP" if _s(d["close"]) > _s(d.get("SMA_50") or 0) else "DOWN",
            "daily_trend":  "UP" if _s(d["close"]) > _s(d.get("EMA_20") or 0) else "DOWN",
            "adx":          _s(d.get("ADX") or 0.0),
            "atr":          atr_val,
        }

    def _score_signal(self, snap):
        score = 0
        if snap["weekly_trend"] == "UP" and snap["daily_trend"] == "UP": score += 2
        elif snap["weekly_trend"] == "UP" or snap["daily_trend"] == "UP": score += 1
        if (snap.get("adx") or 0) >= self.parameters["adx_min"]: score += 1
        signal = "BUY" if score >= self.parameters["entry_score_min"] else "HOLD"
        return signal, score

    def _week_key(self, dt) -> str:
        d = dt.date() if hasattr(dt, "date") else dt
        return f"{d.isocalendar()[0]}-{d.isocalendar()[1]:02d}"

    def _check_multitf_roc(self, sym) -> bool:
        try:
            bars = self.get_historical_prices(sym, 70, "day")
            if not bars or bars.df is None or len(bars.df) < 22: return True
            c = bars.df["close"]
            roc5  = (_s(c.iloc[-1]) / _s(c.iloc[-6])  - 1) * 100
            roc20 = (_s(c.iloc[-1]) / _s(c.iloc[-21]) - 1) * 100

            if self.vars.regime == "BULL": return roc5 > 0 and roc20 > 0
            if len(c) < 62: return roc5 > 0 and roc20 > 0
            
            roc60 = (_s(c.iloc[-1]) / _s(c.iloc[-61]) - 1) * 100
            return roc5 > 0 and roc20 > 0 and roc60 > 0
        except Exception: return True

    def _score_universe_candidates(self, universe_records: list) -> list:
        if not universe_records: return []
        rows = [r for r in universe_records if r.get("symbol") != "CASH"]
        if not rows: return []

        df = pd.DataFrame(rows)
        for col in ["roc20", "rs", "adx", "roc_adj", "roc60"]:
            if col not in df.columns: df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        regime = self.vars.regime
        w_roc20, w_rs, w_adx, w_roc_adj, w_roc60 = REGIME_SCORE_WEIGHTS.get(regime, REGIME_SCORE_WEIGHTS["BULL"])

        if df["roc_adj"].abs().sum() < 1e-6: df["roc_adj"] = df["roc20"]

        df["composite"] = (
            w_roc20  * _norm(df["roc20"])   +
            w_rs     * _norm(df["rs"])       +
            w_adx    * _norm(df["adx"])      +
            w_roc_adj* _norm(df["roc_adj"])  +
            w_roc60  * _norm(df["roc60"])
        )

        if not df.empty:
            self.vars.score_history.setdefault(regime, []).append(float(df["composite"].median()))

        all_sorted = df.sort_values("composite", ascending=False)
        if regime == "BULL": return all_sorted.head(10).to_dict(orient="records")

        hist = self.vars.score_history.get(regime, [])
        if len(hist) >= 5:
            gate_threshold = float(np.median(hist[-self.quality_gate_window:]))
            df["passed_gate"] = df["composite"] >= gate_threshold
        else:
            df["passed_gate"] = True  

        qualified = df[df["passed_gate"]].sort_values("composite", ascending=False)
        if len(qualified) < 3: return all_sorted.head(5).to_dict(orient="records")
        return qualified.to_dict(orient="records")

    def _update_dd(self):
        if self.vars.peak_portfolio <= 0: self.vars.peak_portfolio = self.portfolio_value
        dd = (self.vars.peak_portfolio - self.portfolio_value) / self.vars.peak_portfolio

        if dd > 0.05: self.vars.days_in_dd += 1
        else: self.vars.days_in_dd  = 0; self.vars.peak_portfolio = self.portfolio_value

        if self.vars.days_in_dd >= self.amnesia_days:
            self.vars.peak_portfolio = self.portfolio_value
            self.vars.days_in_dd     = 0

        return dd > self.max_drawdown_limit, dd

    def _compute_allocations(self, snaps: dict, scored_candidates: list) -> dict:
        if self.max_per_instrument <= 0: return {}
        qualified_syms = {r["symbol"] for r in scored_candidates}

        scores = {}
        for sym, snap in snaps.items():
            if sym not in qualified_syms: continue
            signal, sc = self._score_signal(snap)
            if signal == "BUY": scores[sym] = sc

        if not scores: return {}
        total = sum(scores.values())
        weights = {s: v / total for s, v in scores.items()}

        if self.use_vol_targeting:
            vol_map = {r["symbol"]: r.get("realized_vol", 0.02) for r in scored_candidates}
            inv_vol = {}
            for s in weights:
                v = vol_map.get(s, 0.02)
                inv_vol[s] = 1.0 / max(v, 0.005) 
            inv_total = sum(inv_vol.values())
            if inv_total > 0:
                for s in weights:
                    iv_w = inv_vol[s] / inv_total
                    weights[s] = 0.5 * weights[s] + 0.5 * iv_w
                wt = sum(weights.values())
                weights = {s: w / wt for s, w in weights.items()}

        # ── AI POSITION SIZING (layer baru) ──────────────────────────
        # AI menilai conviction tiap kandidat → size_mult mengali bobot.
        # Konsentrasi di pemenang sejati (emerging+PEAD kuat), kecil utk
        # momentum biasa. Stock selection tetap deterministik.
        weights_determ = dict(weights)  # simpan untuk perbandingan log
        if self.use_ai_sizing and self.ai_committee is not None:
            cand_map = {r["symbol"]: r for r in scored_candidates}
            ai_input = [cand_map[s] for s in weights if s in cand_map]
            sizing, provider = self.ai_committee.decide(ai_input, self.vars.regime)
            size_mults = {d["symbol"]: d["size_mult"] for d in sizing}
            # Terapkan size_mult lalu renormalisasi
            for s in weights:
                weights[s] *= size_mults.get(s, 1.0)
            wt = sum(weights.values())
            if wt > 0:
                weights = {s: w / wt for s, w in weights.items()}
            # Log keputusan AI sizing
            if self.dlogger:
                week = str(self.get_datetime().date())
                decisions = []
                for d in sizing:
                    s = d["symbol"]
                    c = cand_map.get(s, {})
                    decisions.append({
                        "symbol": s, "score": c.get("composite", c.get("score", 0)),
                        "roc20": c.get("roc20", 0), "pead": c.get("pead", 0),
                        "conviction": d["conviction"], "size_mult": d["size_mult"],
                        "weight_ai": weights.get(s, 0), "weight_determ": weights_determ.get(s, 0),
                        "reasoning": d.get("reasoning", ""),
                    })
                self.dlogger.log_sizing(week, self.vars.regime, decisions, provider)
            if self.notifier:
                week = str(self.get_datetime().date())
                self.notifier.notify_sizing(week, self.vars.regime, sizing, provider)
            self.vars.last_sizing = {d["symbol"]: d["conviction"] for d in sizing}

        effective_leverage = self.current_leverage
        if self.use_crash_protection and self.vars_crash_active: effective_leverage *= 0.5
        
        deployable = self.portfolio_value * (1 - self.cash_reserve) * effective_leverage
        sector_map = {r["symbol"]: r.get("sector", "Unknown") for r in scored_candidates}

        alloc = {}
        sector_count = {}
        for sym in sorted(weights, key=lambda x: weights[x], reverse=True):
            sector = sector_map.get(sym, "Unknown")
            if sector_count.get(sector, 0) >= self.max_per_sector: continue

            w = weights[sym]
            dollar = min(w * deployable, self.portfolio_value * self.max_per_instrument)
            dollar = max(dollar, self.portfolio_value * self.min_per_instrument)
            alloc[sym] = dollar
            sector_count[sector] = sector_count.get(sector, 0) + 1

        return alloc

    def _manage_bear_defense(self):
        etf = self.inverse_etf
        long_positions = [p for p in self.get_positions() if p.quantity > 0 and p.asset.symbol != etf]

        if self.bear_mode == "cash":
            if long_positions:
                for p in long_positions: self.submit_order(self.create_order(p.asset.symbol, quantity=p.quantity, side="sell"))
            return

        pos = self.get_position(etf)
        if pos is not None and pos.quantity > 0: return

        if long_positions:
            for p in long_positions: self.submit_order(self.create_order(p.asset.symbol, quantity=p.quantity, side="sell"))
            return

        try:
            price = self.get_last_price(etf)
            if not price or price <= 0: return
            alloc_dollar = self.portfolio_value * self.bear_inverse_alloc
            qty = int(alloc_dollar / price)
            if qty >= 1:
                self.submit_order(self.create_order(etf, quantity=qty, side="buy"))
                self.vars_inverse_entry_week = self.get_datetime().isocalendar()[1]
        except Exception: pass

    def _exit_inverse_etf(self):
        if self.bear_mode != "inverse": return
        etf = self.inverse_etf
        pos = self.get_position(etf)
        if pos is None or pos.quantity <= 0: return
        if self.vars_inverse_entry_week is not None:
            cur_week = self.get_datetime().isocalendar()[1]
            if abs(cur_week - self.vars_inverse_entry_week) < self.inverse_min_hold_weeks: return 
        self.submit_order(self.create_order(etf, quantity=pos.quantity, side="sell"))
        self.vars_inverse_entry_week = None

    def _manage_exit(self, sym, price):
        if not self.get_position(sym): return
        st = self._st(sym)
        st["highest"] = max(st["highest"] or price, price)
        trail_stop    = st["highest"] - (st["atr"] * self.current_atr_mult)

        # ── LAPIS 1: MOMENTUM BREAK PER-SAHAM (V32 exit fix) ──────────
        # Keluar kalau momentum SAHAM ITU patah, lepas dari regime pasar.
        # Ini membuang jebakan (PTON/ZM) saat mulai ambruk — tidak nunggu
        # pasar masuk BEAR. Pakai roc20<0 (BUKAN roc5) supaya koreksi sehat
        # NVDA (1-2 minggu) tidak ikut terbuang, hanya kerusakan struktural.
        momentum_broken = False
        try:
            bars = self.get_historical_prices(sym, 30, "day")
            if bars is not None and bars.df is not None and len(bars.df) >= 21:
                c = bars.df["close"]
                roc20 = (_s(c.iloc[-1]) / _s(c.iloc[-21]) - 1) * 100
                sma20 = _s(c.iloc[-20:].mean())
                cur   = _s(c.iloc[-1])
                # Patah = tren 1-bulan negatif DAN harga di bawah SMA20-nya
                # (dua konfirmasi → bukan noise sesaat)
                if roc20 < 0 and cur < sma20:
                    momentum_broken = True
        except Exception:
            momentum_broken = False

        if price <= trail_stop or momentum_broken:
            pos = self.get_position(sym)
            if pos and pos.quantity > 0:
                # Hitung P&L + info entry untuk notifikasi
                entry = st.get("entry_price") or price
                entry_date = st.get("entry_date", "")
                pnl_pct = (price / entry - 1) if entry > 0 else 0.0
                reason = "momentum break" if momentum_broken else "trailing stop"
                self.submit_order(self.create_order(sym, quantity=int(pos.quantity), side="sell"))
                self.log_message(f"🚪 EXIT {sym} ({reason})")
                if self.notifier:
                    self.notifier.notify_exit(sym, price, int(pos.quantity), pnl_pct,
                                              reason, entry_price=entry, entry_date=entry_date)
                if self.dlogger:
                    self.dlogger.log_exec(str(self.get_datetime().date()), "EXIT", sym,
                                          price, int(pos.quantity), exit_layer=reason,
                                          pnl_pct=pnl_pct, reason=f"entry=${entry:.2f}")
            st.update({"entry_price": None, "highest": None, "entry_date": None,
                       "last_exit_date": str(self.get_datetime().date())})

    def _close_all(self):
        for pos in self.get_positions():
            if pos.quantity > 0:
                sym = pos.asset.symbol
                st = self._st(sym)
                entry = st.get("entry_price")
                try:
                    price = self.get_last_price(sym) or entry or 0
                except Exception:
                    price = entry or 0
                pnl_pct = (price / entry - 1) if entry and entry > 0 else 0.0
                self.submit_order(self.create_order(sym, quantity=int(pos.quantity), side="sell"))
                if self.notifier and price:
                    self.notifier.notify_exit(sym, price, int(pos.quantity), pnl_pct,
                                              "regime", entry_price=entry,
                                              entry_date=st.get("entry_date", ""))
                if self.dlogger and price:
                    self.dlogger.log_exec(str(self.get_datetime().date()), "EXIT", sym,
                                          price, int(pos.quantity), exit_layer="regime",
                                          pnl_pct=pnl_pct, reason="BEAR close all")
                self._st(sym).update({"entry_price": None, "highest": None, "entry_date": None,
                                      "last_exit_date": str(self.get_datetime().date())})

    def _update_universe(self):
        dt = self.get_datetime()
        wk_key = self._week_key(dt)
        if self.vars.last_scan_week == wk_key: return

        if self.bt_universe_df is not None:
            cur  = pd.Timestamp(dt.date())
            past = self.bt_universe_df[self.bt_universe_df["week_start"] <= cur]
            if not past.empty:
                latest_wk   = past["week_start"].max()
                latest_rows = past[past["week_start"] == latest_wk].to_dict(orient="records")
                scored = self._score_universe_candidates(latest_rows)
                if scored: self.vars.active_universe = scored
                else: self.vars.active_universe = [{"symbol": "CASH", "sector": "market", "composite": 0.0}]
                # Log scanner decision
                if self.dlogger and scored:
                    week = str(cur.date())
                    cand_sorted = sorted(scored, key=lambda x: x.get("composite", 0), reverse=True)
                    self.dlogger.log_scanner(week, self.vars.regime, cand_sorted)
        self.vars.last_scan_week = wk_key

    def on_trading_iteration(self):
        self._update_regime()

        breached, dd = self._update_dd()
        if breached:
            self._close_all()
            return

        self._update_universe()

        held_syms  = [p.asset.symbol for p in self.get_positions() if p.quantity > 0]
        is_cash_wk = any(u["symbol"] == "CASH" for u in self.vars.active_universe) if self.vars.active_universe else True
        scan_syms  = [u["symbol"] for u in self.vars.active_universe if u["symbol"] != "CASH"] if self.vars.active_universe else []

        snaps = {}
        for sym in list(set(scan_syms + held_syms)):
            try:
                s = self._build_snapshot(sym)
                if s: snaps[sym] = s
            except Exception: pass

        for sym in held_syms:
            if sym in snaps: self._manage_exit(sym, snaps[sym]["price"])
        held_syms = [p.asset.symbol for p in self.get_positions() if p.quantity > 0]

        if self.vars.regime == "BEAR":
            if self.use_bear_defense: self._manage_bear_defense()
            return

        if self.use_bear_defense: self._exit_inverse_etf()

        if is_cash_wk or not snaps: return

        scored_candidates = [u for u in self.vars.active_universe if u["symbol"] != "CASH"]
        allocations = self._compute_allocations({s: snaps[s] for s in scan_syms if s in snaps}, scored_candidates)

        for sym in scan_syms:
            if sym not in snaps or sym in held_syms: continue

            last_exit = self._st(sym).get("last_exit_date")
            if last_exit:
                try:
                    days_since = (self.get_datetime().date() - date_type.fromisoformat(str(last_exit))).days
                    if days_since < self.cooldown_days: continue
                except Exception: pass

            signal, score = self._score_signal(snaps[sym])
            if signal != "BUY" or sym not in allocations: continue

            if not self._check_multitf_roc(sym): continue

            price    = snaps[sym]["price"]
            avail    = max(0.0, float(self.cash) - self.portfolio_value * self.cash_reserve)
            cash_use = min(allocations[sym], avail)
            if cash_use < price: continue
            
            shares = math.floor(cash_use / price)
            if shares <= 0: continue

            self.submit_order(self.create_order(sym, quantity=shares, side="buy"))
            pct_port = (shares * price) / self.portfolio_value if self.portfolio_value > 0 else 0
            conviction = self.vars.last_sizing.get(sym, "")
            if self.notifier:
                self.notifier.notify_buy(sym, price, shares, pct_port, conviction)
            if self.dlogger:
                self.dlogger.log_exec(str(self.get_datetime().date()), "BUY", sym,
                                      price, shares, pct_portfolio=pct_port,
                                      reason=f"conviction={conviction}")
            self._st(sym).update({
                "entry_price": float(price),
                "entry_date":  str(self.get_datetime().date()),
                "highest":     float(price),
                "sector":      next((r.get("sector", "Unknown") for r in scored_candidates if r["symbol"] == sym), "Unknown"),
            })

# ─────────────────────────────────────────────────────────────────────────────
# 11-YEAR RUNNER ONLY
# ─────────────────────────────────────────────────────────────────────────────
BENCHMARKS = ["SPY", "QQQ", "SOXX"]

def get_benchmark_returns(start: datetime, end: datetime) -> dict:
    try:
        df = yf.download(" ".join(BENCHMARKS), start=start, end=end, auto_adjust=True, progress=False)["Close"]
        results = {}
        for sym in BENCHMARKS:
            if sym in df.columns:
                prices = df[sym].dropna()
                if len(prices) >= 2: results[sym] = float(prices.iloc[-1] / prices.iloc[0] - 1)
        return results
    except Exception: return {}

import re, glob, os

def _parse_lumibot_log(log_text: str) -> Optional[float]:
    patterns = [r"Total[\s_]Return[^\d-]*([+-]?[\d,]+\.?\d*)\s*%", r"CAGR[^\d-]*([+-]?[\d,]+\.?\d*)\s*%"]
    for pat in patterns:
        m = re.search(pat, log_text, re.IGNORECASE)
        if m:
            try: return float(m.group(1).replace(",", "")) / 100.0
            except: pass
    return None

def _read_tearsheet_return(csv_path: str) -> Optional[float]:
    try:
        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip() for c in df.columns]
        strat_col = next((c for c in df.columns if c.lower() in ["strategy", "strat"]), None)
        if not strat_col: return None
        metric_col = df.columns[0]
        for _, row in df.iterrows():
            metric = str(row[metric_col]).lower().strip()
            if "cagr" in metric or "annual return" in metric:
                try: return float(str(row[strat_col]).replace("%", "").replace(",", "").strip()) / 100.0
                except: pass
        for _, row in df.iterrows():
            metric = str(row[metric_col]).lower().strip()
            if "total return" in metric:
                try: return float(str(row[strat_col]).replace("%", "").replace(",", "").strip()) / 100.0
                except: pass
    except: pass
    return None

def _read_tearsheet_metrics(csv_path: str) -> dict:
    metrics = {}
    try:
        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip() for c in df.columns]
        strat_col = next((c for c in df.columns if c.lower() in ["strategy", "strat"]), None)
        if not strat_col: return metrics
        metric_col = df.columns[0]
        for _, row in df.iterrows():
            k, v = str(row[metric_col]).strip(), str(row[strat_col]).strip()
            if k and k != "nan" and v and v != "nan": metrics[k] = v
    except: pass
    return metrics

PERIODS = [
    ("FULL 11Y", datetime(2015, 1, 1), datetime(2026, 5, 28), "Full 11+ tahun"),
]

def run_all():
    print("\n📡 Fetching benchmark returns...")
    all_benchmarks = {label: get_benchmark_returns(start, end) for label, start, end, _ in PERIODS}
    all_cagr, all_total_ret, all_sharpe, all_maxdd = {}, {}, {}, {}

    for label, start, end, desc in PERIODS:
        print(f"\n🔄 [{label}] {start.date()} → {end.date()} ({desc})")
        existing = set(glob.glob("logs/*tearsheet*.csv"))

        try:
            CitadelLevelQuantV8S30.backtest(
                YahooDataBacktesting, start, end,
                show_plot=False, show_tearsheet=False,
                starting_balance=100_000, benchmark_asset="SPY",
                buy_trading_fees=[TradingFee(percent_fee=CitadelLevelQuantV8S30.parameters["buy_fee_pct"])],
                sell_trading_fees=[TradingFee(percent_fee=CitadelLevelQuantV8S30.parameters["sell_fee_pct"])],
                parameters=CitadelLevelQuantV8S30.parameters,
            )
        except Exception as ex:
            print(f"  ⚠️ Error: {ex}"); all_cagr[label] = None; continue

        new_tearsheets = sorted(set(glob.glob("logs/*tearsheet*.csv")) - existing, key=os.path.getmtime, reverse=True)
        cagr, metrics = None, {}
        
        if new_tearsheets:
            csv_path = new_tearsheets[0]
            cagr    = _read_tearsheet_return(csv_path)
            metrics = _read_tearsheet_metrics(csv_path)
            print(f"  📄 Tearsheet: {os.path.basename(csv_path)}")
        else:
            all_ts = sorted(glob.glob("logs/*tearsheet*.csv"), key=os.path.getmtime, reverse=True)
            if all_ts:
                cagr    = _read_tearsheet_return(all_ts[0])
                metrics = _read_tearsheet_metrics(all_ts[0])

        all_cagr[label]      = cagr
        all_total_ret[label] = metrics.get("Total Return")
        all_sharpe[label]    = metrics.get("Sharpe")
        all_maxdd[label]     = metrics.get("Max Drawdown")

        if cagr is not None:
            print(f"  ✅ CAGR: {cagr:>+.2%}  |  Total Return: {all_total_ret.get(label, '?')}  |  Sharpe: {all_sharpe.get(label,'?')}  |  MaxDD: {all_maxdd.get(label,'?')}")
        else:
            print(f"  ⚠️ Return tidak terbaca dari tearsheet")

    print(f"\n\n{'='*85}\n  📊 CITADEL V8S30 DETERMINISTIC + V32 UNIVERSE — 11Y TEST\n{'='*85}")
    h = f"  {'Period':<13} {'CAGR':>8} {'TotalRet':>10} {'Sharpe':>8} {'MaxDD':>9} {'SPY':>8} {'QQQ':>8} {'SOXX':>8}"
    print(h + "\n  " + "-"*83)

    for label, _, _, _ in PERIODS:
        cagr = all_cagr.get(label); bm = all_benchmarks.get(label, {})
        tr = all_total_ret.get(label, "N/A"); sh = all_sharpe.get(label, "N/A"); dd = all_maxdd.get(label, "N/A")
        row = f"  {label:<13}"
        row += f" {cagr:>+7.1%}" if cagr is not None else f" {'N/A':>8}"
        row += f" {str(tr):>10} {str(sh):>8} {str(dd):>9}"
        for sym in BENCHMARKS: row += f" {bm.get(sym):>+7.1%}" if bm.get(sym) is not None else f" {'N/A':>8}"
        print(row)
    print("  " + "-"*83 + "\n")

if __name__ == "__main__":
    run_all()