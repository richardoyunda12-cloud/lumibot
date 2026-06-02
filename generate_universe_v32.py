"""
generate_universe_v26.py
Scanner V32 — + PEAD factor (earnings drift proxy, backtestable) — Phase 1 Full Upgrade (dynamic scanner)

Perubahan dari V22:
  [1] REGIME CLASSIFIER     — 3-state market regime: trending_bull / sideways / high_vol
  [2] DYNAMIC SCORE WEIGHTS — bobot score berubah sesuai regime
  [3] MULTI-TIMEFRAME ROC   — ROC5 + ROC20 + ROC60 harus aligned sebelum entry
  [4] ATR-ADJUSTED MOMENTUM — risk-adjusted score: roc20 / atr_pct
  [5] DYNAMIC QUALITY GATE  — hanya masuk kalau score >= rolling 20-week median
  [6] SECTOR CAP            — max 3 saham per sektor dalam 15 kandidat

Output: cache/weekly_universe_history_v25.csv

Jalankan SEKALI sebelum backtest V25:
  python generate_universe_v25.py
"""

import os, logging
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
from collections import defaultdict

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("UniverseGenV32")


def scalar(x) -> float:
    """
    Safely extract a Python float from any pandas/numpy element.
    Handles: float, int, np.floating, single-element Series, 0-d array.
    Fixes FutureWarning: 'Calling float on a single element Series'.
    """
    if isinstance(x, pd.Series):
        x = x.iloc[0]
    if hasattr(x, "item"):          # np.floating / np.int_ / 0-d array
        return float(x.item())
    return float(x)

# ─────────────────────────────────────────────────────────────────────
# UNIVERSE V28 TIER-1: large-cap (asli) + mid-cap (BARU)
# Momentum premium lebih kuat di mid-cap. Ditambahkan ~40 mid-cap likuid
# lintas sektor untuk memperbanyak kandidat (fix root cause 2020-2021).
# ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────
# UNIVERSE V31 — DIPERLUAS ~300 saham likuid US
# Tujuan: tangkap "NVDA baru" yang mungkin tidak ada di list sempit.
# Untuk LIVE: ganti dengan MTUM holdings + screening likuiditas (dinamis).
# Untuk BACKTEST: list statis ini (terima survivorship bias, fokus validasi logika).
# Semua saham besar & likuid lintas sektor.
# ─────────────────────────────────────────────────────────────────────
MEGA_TECH = [
    "NVDA","AAPL","MSFT","GOOGL","GOOG","AMZN","META","TSLA","AVGO","TSM",
    "ORCL","CRM","ADBE","AMD","INTC","QCOM","CSCO","TXN","INTU","NOW",
    "MU","AMAT","LRCX","KLAC","ADI","SNPS","CDNS","MRVL","PANW","ANET",
    "FTNT","CRWD","ZS","DDOG","NET","SNOW","PLTR","TEAM","WDAY","HUBS",
    "MDB","ZM","OKTA","DOCU","TWLO","ON","MCHP","NXPI","ASML","ARM",
]
GROWTH_CONSUMER = [
    "NFLX","DIS","CMCSA","SHOP","ABNB","UBER","DASH","BKNG","MELI","SE",
    "LULU","NKE","SBUX","MCD","CMG","ROST","TJX","ULTA","DECK","RH",
    "RBLX","ROKU","SPOT","PINS","SNAP","ETSY","CHWY","W","DKNG","COIN",
]
FINANCIAL = [
    "JPM","BAC","WFC","GS","MS","C","SCHW","BLK","AXP","V",
    "MA","PYPL","COF","DFS","FITB","HBAN","USB","PNC","TFC","BK",
    "SQ","HOOD","SOFI","ICE","CME","SPGI","MCO","AON","PGR","TRV",
]
HEALTHCARE = [
    "LLY","UNH","JNJ","ABBV","MRK","PFE","TMO","DHR","ABT","BMY",
    "AMGN","GILD","VRTX","REGN","ISRG","MDT","CVS","CI","HUM","ELV",
    "MRNA","DXCM","ALGN","IDXX","BIIB","ZTS","BSX","SYK","BDX","HCA",
]
INDUSTRIAL_ENERGY = [
    "CAT","DE","HON","GE","BA","LMT","RTX","UNP","UPS","FDX",
    "URI","PWR","ETN","EMR","PH","ROK","CMI","GD","NOC","MMM",
    "XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY","DVN",
    "FANG","HAL","WMB","KMI","LNG","FSLR","ENPH","SEDG","NEE","DUK",
]
MATERIALS_OTHER = [
    "LIN","APD","SHW","FCX","NEM","NUE","DOW","DD","ECL","CTVA",
    "PG","KO","PEP","COST","WMT","HD","LOW","TGT","DG","DLTR",
    "MO","PM","CL","KMB","GIS","MDLZ","KHC","STZ","KDP","MNST",
    "AMT","PLD","EQIX","CCI","PSA","O","SPG","WELL","DLR","T",
    "VZ","TMUS","CMCSA",
]

BASE_UNIVERSE = list(dict.fromkeys(
    MEGA_TECH + GROWTH_CONSUMER + FINANCIAL + HEALTHCARE +
    INDUSTRIAL_ENERGY + MATERIALS_OTHER
)) + ["SPY"]

# ─────────────────────────────────────────────
# PHASE 1 CHANGE [1]: REGIME CLASSIFIER
# ─────────────────────────────────────────────
# Kenapa: V22 tidak punya market context sama sekali.
# Scanner memilih kandidat dengan cara identik di semua kondisi.
# Di sideways market, ROC-heavy scoring justru pilih saham overbought.
#
# 3 regime:
#   trending_bull  → SPY di atas MA50 & MA200, VIX < 20, ADX_SPY > 25
#   high_vol       → VIX >= 25 ATAU SPY drop > 5% dalam 4 minggu
#   sideways       → kondisi di antaranya

REGIME_WEIGHTS = {
    # (w_roc20, w_rs, w_adx, w_roc_adj, w_roc60)
    # V31 FIX: trending_bull geser bobot dari roc_adj (yg hukum volatilitas)
    # ke roc20 mentah → pemenang volatil seperti NVDA tidak dihukum berlebihan.
    # Di bull, kita MAU momentum mentah, bukan yang "halus".
    "trending_bull": (0.42, 0.23, 0.13, 0.07, 0.15),
    "sideways":      (0.20, 0.30, 0.25, 0.15, 0.10),
    "high_vol":      (0.10, 0.35, 0.30, 0.10, 0.15),
    # vshape: prioritaskan saham yang sudah mulai rebound kuat
    "vshape":        (0.15, 0.30, 0.15, 0.25, 0.15),
}

def classify_regime(spy_hist: pd.Series, vix_hist: pd.Series | None, wk: pd.Timestamp) -> str:
    """
    Klasifikasi regime market untuk minggu tertentu.
    Returns: 'trending_bull' | 'sideways' | 'high_vol' | 'vshape'

    'vshape' = sub-regime dari high_vol: crash yang sudah mulai V-shape recovery.
    Deteksi: SPY naik > 8% dari low 10 hari dalam 10 hari terakhir.
    V-shape crash (COVID 2020, AI recovery 2023) → pakai ROC5 gate → catch early.
    Crash bertahap (2018 Q4, 2022 bear) → tidak ada velocity → tetap ROC20 gate.
    """
    spy_slice = spy_hist[spy_hist.index <= wk]
    if len(spy_slice) < 200:
        return "sideways"

    price    = scalar(spy_slice.iloc[-1])
    ma50     = scalar(spy_slice.rolling(50).mean().iloc[-1])
    ma200    = scalar(spy_slice.rolling(200).mean().iloc[-1])
    price_4w = scalar(spy_slice.iloc[-21]) if len(spy_slice) >= 21 else price
    dd_4w    = (price - price_4w) / price_4w * 100

    spy_adx_proxy = abs(scalar(spy_slice.pct_change(5).rolling(10).mean().iloc[-1])) * 100

    vix_level = 18.0
    if vix_hist is not None:
        vix_slice = vix_hist[vix_hist.index <= wk]
        if not vix_slice.empty:
            vix_level = scalar(vix_slice.iloc[-1])

    # Base regime
    if vix_level >= 25 or dd_4w <= -5.0:
        base = "high_vol"
    elif price > ma50 and price > ma200 and vix_level < 20:
        return "trending_bull"
    else:
        return "sideways"

    # Sub-classify high_vol: cek apakah ini V-shape recovery
    # Velocity = seberapa jauh SPY naik dari low 10 hari dalam 10 hari terakhir
    if len(spy_slice) >= 10:
        low10  = float(spy_slice.iloc[-10:].min())
        velocity = (price - low10) / low10 if low10 > 0 else 0
        if velocity >= 0.08:
            # SPY sudah naik >8% dari low 10 hari → V-shape recovery
            return "vshape"

    return base


SECTOR_CACHE = {}
def get_sector(sym):
    if sym in SECTOR_CACHE: return SECTOR_CACHE[sym]
    try:
        SECTOR_CACHE[sym] = yf.Ticker(sym).info.get("sector", "Unknown")
    except:
        SECTOR_CACHE[sym] = "Unknown"
    return SECTOR_CACHE[sym]


def generate(start_dt: datetime, end_dt: datetime,
             out_csv: str = "cache/weekly_universe_history_v32.csv"):

    os.makedirs("cache", exist_ok=True)
    dl_start = start_dt - timedelta(days=400)
    logger.info(f"📥 Download {dl_start.date()} → {end_dt.date()}...")

    # Download semua saham
    df_all = yf.download(
        BASE_UNIVERSE, start=dl_start, end=end_dt + timedelta(days=7),
        interval="1d", auto_adjust=True, progress=True,
        group_by="ticker", threads=True,
    )
    if df_all.empty:
        raise RuntimeError("Download gagal.")

    dfs = {}
    for sym in BASE_UNIVERSE:
        try:
            if sym in df_all.columns.get_level_values(0):
                d = df_all[sym].dropna()
                if len(d) > 60:
                    dfs[sym] = d
        except Exception:
            pass
    logger.info(f"✅ {len(dfs)} symbols loaded")

    # Download VIX untuk regime classifier
    logger.info("📊 Downloading VIX...")
    vix_hist = None
    try:
        vix_raw = yf.download("^VIX", start=dl_start, end=end_dt + timedelta(days=7),
                               interval="1d", auto_adjust=True, progress=False)
        if not vix_raw.empty:
            vix_hist = vix_raw["Close"].dropna()
            logger.info(f"✅ VIX loaded: {len(vix_hist)} rows")
    except Exception as e:
        logger.warning(f"⚠️  VIX download gagal: {e} — regime pakai SPY-only mode")

    # Rolling Beta 252 hari (sama dengan V22)
    logger.info("📊 Computing rolling beta...")
    beta_series = {}
    spy_close = None
    if "SPY" in dfs:
        spy_close = dfs["SPY"]["Close"]
        spy_ret   = spy_close.pct_change()
        spy_var   = spy_ret.rolling(252).var()
        for sym, df in dfs.items():
            if sym == "SPY": continue
            cov = df["Close"].pct_change().rolling(252).cov(spy_ret)
            beta_series[sym] = cov / spy_var

    weeks = pd.date_range(start=start_dt, end=end_dt, freq="W-FRI")
    rows, cash_weeks = [], 0

    # ─────────────────────────────────────────────
    # PHASE 1 CHANGE [5]: DYNAMIC QUALITY GATE
    # ─────────────────────────────────────────────
    # Rolling buffer: simpan median score 20 minggu terakhir
    # per regime — kalau tidak ada kandidat yang tembus, CASH lebih baik
    recent_scores_by_regime: dict[str, list[float]] = defaultdict(list)
    QUALITY_GATE_WINDOW = 20

    logger.info(f"🔄 Scanning {len(weeks)} weeks...")
    for i, wk in enumerate(weeks):
        if i % 52 == 0:
            logger.info(f"  {i/len(weeks)*100:.0f}% — {wk.date()}")

        # ── Regime detection ──────────────────────────────────────
        # PHASE 1 CHANGE [1]
        if spy_close is not None:
            regime = classify_regime(spy_close, vix_hist, wk)
        else:
            regime = "sideways"

        w_roc20, w_rs, w_adx, w_roc_adj, w_roc60 = REGIME_WEIGHTS[regime]

        # SPY ROC untuk relative strength
        spy_roc20 = 0.0
        if spy_close is not None:
            spy_c = spy_close[spy_close.index <= wk]
            if len(spy_c) >= 21:
                spy_roc20 = (scalar(spy_c.iloc[-1]) / scalar(spy_c.iloc[-21]) - 1) * 100

        candidates = []
        for sym, df in dfs.items():
            if sym == "SPY": continue
            hist = df[df.index <= wk]
            if len(hist) < 200: continue

            try:
                c, v, h, l = hist["Close"], hist["Volume"], hist["High"], hist["Low"]
                price = scalar(c.iloc[-1])
                if price < 10: continue

                adv20 = scalar((c * v).rolling(20).mean().iloc[-1])
                if adv20 < 50_000_000: continue

                # ── PHASE 1 CHANGE [3]: MULTI-TIMEFRAME ROC ──────────
                # Kenapa: ROC20 saja tidak cukup — bisa sudah di puncak
                # short-term move. Butuh konfirmasi dari 3 timeframe.
                # Entry hanya kalau ROC5 > 0 AND ROC20 > 0 AND ROC60 > 0
                # (semua aligned ke atas = genuine momentum, bukan noise)
                if len(c) < 62: continue

                roc5  = (scalar(c.iloc[-1]) / scalar(c.iloc[-6])  - 1) * 100
                roc20 = (scalar(c.iloc[-1]) / scalar(c.iloc[-21]) - 1) * 100
                roc60 = (scalar(c.iloc[-1]) / scalar(c.iloc[-61]) - 1) * 100
                rs    = roc20 - spy_roc20

                # ── EMERGING LEADER FAST-TRACK (V31) ─────────────────────
                # Menangkap "NVDA baru" lebih awal. Ciri leader sejati saat
                # baru muncul (Livermore/Darvas/O'Neil/Minervini, terbukti 100thn):
                #   1. Akselerasi: pace 5-hari > pace 20-hari (makin kencang)
                #   2. Breakout: harga >= 95% dari high 120-hari (new-high leadership)
                #   3. Volume surge: volume 5-hari > 1.5x rata-rata 60-hari (institusi)
                #   4. RS sangat kuat: rs > 5 (jauh ungguli market)
                # Kalau KEEMPAT terpenuhi → lolos MESKI roc60 masih negatif
                # (saham baru keluar dari base, 3-bulan masih merah tapi meledak).
                # Ini DURABLE: ciri universal emerging leader, bukan tuning NVDA.
                is_emerging_leader = False
                try:
                    if len(c) >= 120:
                        # 1. Akselerasi (annualized pace)
                        pace5  = roc5 / 5
                        pace20 = roc20 / 20
                        accel = pace5 > pace20 * 1.2
                        # 2. Breakout ke high 120-hari
                        high120 = scalar(c.iloc[-120:].max())
                        near_high = price >= 0.95 * high120
                        # 3. Volume surge
                        vol5  = scalar(v.iloc[-5:].mean())
                        vol60 = scalar(v.iloc[-60:].mean())
                        vol_surge = vol60 > 0 and vol5 > 1.5 * vol60
                        # 4. RS sangat kuat
                        strong_rs = rs > 5.0
                        # Butuh akselerasi + breakout + (volume ATAU rs kuat)
                        is_emerging_leader = (accel and near_high and
                                              (vol_surge or strong_rs) and
                                              roc5 > 0 and roc20 > 0)
                except Exception:
                    is_emerging_leader = False

                # ── Momentum gate — REGIME AWARE (V8S27) ─────────────────
                # trending_bull : ROC5 + ROC20 + ROC60 positif + RS > 0
                #                 ATAU emerging leader (jalur cepat)
                # sideways      : ROC5 + ROC20 positif + RS > 0 ATAU emerging
                # high_vol      : ROC5 + ROC20 positif
                # vshape        : ROC5 saja
                if regime == "trending_bull":
                    passed_normal = (roc5 > 0.0 and roc20 > 0.0 and
                                     roc60 > 0.0 and rs > 0.0)
                    if not (passed_normal or is_emerging_leader):
                        continue
                elif regime == "sideways":
                    if roc5 <= 0.0 or roc20 <= 0.0:
                        continue
                    if rs <= 0.0:
                        continue
                elif regime == "vshape":
                    # V-shape recovery: cukup ROC5 > 0
                    # Saham yang rebound dalam 5 hari = pemimpin recovery
                    # RS boleh negatif karena market baru saja crash
                    if roc5 <= 0.0:
                        continue
                else:
                    # high_vol (crash bertahap): ROC5 + ROC20 harus positif
                    # Lebih ketat dari vshape untuk hindari false entry di downtrend
                    if roc5 <= 0.0 or roc20 <= 0.0:
                        continue

                # ── ADX filter ────────────────────────────────────────
                adx_df = ta.adx(h, l, c, length=14)
                adx = scalar(adx_df["ADX_14"].iloc[-1]) \
                      if adx_df is not None and not adx_df.empty else 0.0
                # ADX threshold regime-aware:
                # vshape   → 12 (rebound awal, ADX belum naik)
                # high_vol → 17 (volatile tapi bukan recovery)
                # lainnya  → 20
                if regime == "vshape":
                    adx_min_thr = 12
                elif regime == "high_vol":
                    adx_min_thr = 17
                else:
                    adx_min_thr = 20
                # Emerging leader: ADX boleh lebih rendah (breakout baru, ADX
                # belum sempat naik). Turunkan threshold ke 15 untuk mereka.
                if is_emerging_leader:
                    adx_min_thr = min(adx_min_thr, 15)
                if adx < adx_min_thr: continue

                # ── Beta filter ───────────────────────────────────────
                beta_val = 1.0
                if sym in beta_series:
                    b = beta_series[sym][beta_series[sym].index <= wk].dropna()
                    if not b.empty:
                        beta_val = scalar(b.iloc[-1])
                # Emerging leader dikecualikan dari beta filter (sering high-beta)
                if beta_val < 1.0 and not is_emerging_leader: continue

                # ── PHASE 1 CHANGE [4]: ATR-ADJUSTED MOMENTUM ────────
                # Kenapa: saham high-volatility punya ROC tinggi secara
                # natural. Normalisasi dengan ATR% memberi score yang
                # lebih fair — momentum nyata vs sekedar volatile.
                atr_df = ta.atr(h, l, c, length=14)
                atr_pct = 1.0  # default fallback
                if atr_df is not None and not atr_df.empty:
                    atr_raw = scalar(atr_df.iloc[-1])
                    atr_pct = max((atr_raw / price) * 100, 0.01)

                roc_adj = roc20 / atr_pct  # risk-adjusted momentum

                # ── PEAD PROXY (backtestable, price-based, NO look-ahead) ──
                # Post-Earnings-Announcement-Drift via reaksi harga:
                # Saham yang mengalami gap/lonjakan besar dalam 60 hari terakhir
                # (proxy earnings surprise) lalu terus naik = drift positif.
                # Hanya pakai price data s/d minggu ini → jujur untuk backtest.
                pead_factor = 0.0
                try:
                    rets = c.pct_change().dropna()
                    if len(rets) >= 60:
                        r60 = rets.iloc[-60:]
                        # Cari "earnings-like event": hari dengan return > 2.5 std
                        std60 = float(r60.std())
                        if std60 > 0:
                            big_moves = r60[abs(r60) > 2.5 * std60]
                            if len(big_moves) > 0:
                                # Ambil event terbesar & arahnya, bobot drift sejak event
                                last_event_ret = float(big_moves.iloc[-1])
                                # Posisi event dalam window (makin baru makin relevan)
                                event_pos = r60.index.get_loc(big_moves.index[-1])
                                days_since = len(r60) - 1 - event_pos
                                # Drift sejak event (PEAD inti)
                                if days_since >= 1 and days_since <= 45:
                                    price_at_event = float(c.iloc[-(days_since+1)])
                                    drift = (price - price_at_event) / price_at_event
                                    # PEAD positif: event naik + harga lanjut naik
                                    if last_event_ret > 0:
                                        pead_factor = drift
                except Exception:
                    pead_factor = 0.0

                # ── TIER 2 [E]: MULTI-FACTOR (price-based, NO look-ahead) ──
                # Semua faktor dihitung dari price history s/d minggu ini saja.
                daily_ret = c.pct_change().dropna()
                ret_60 = daily_ret.iloc[-60:] if len(daily_ret) >= 60 else daily_ret

                # Low-vol factor: volatilitas realized 60 hari (LEBIH RENDAH = LEBIH BAIK)
                # Disimpan sebagai negatif agar "tinggi = baik" konsisten dgn faktor lain
                realized_vol = float(ret_60.std()) if len(ret_60) > 5 else 0.05
                low_vol_factor = -realized_vol  # negatif: vol rendah → skor tinggi

                # Quality-from-price factor: konsistensi uptrend
                # = % hari positif - (max drawdown 60 hari). Smooth uptrend = quality.
                pct_positive = float((ret_60 > 0).mean()) if len(ret_60) > 5 else 0.5
                cummax = c.iloc[-60:].cummax() if len(c) >= 60 else c.cummax()
                dd_series = (c.iloc[-len(cummax):] / cummax - 1)
                max_dd_60 = float(dd_series.min()) if len(dd_series) else -0.2
                quality_factor = pct_positive + max_dd_60  # max_dd negatif → mengurangi

                candidates.append({
                    "sym":     sym,
                    "roc20":   roc20,
                    "roc5":    roc5,
                    "roc60":   roc60,
                    "rs":      rs,
                    "adx":     adx,
                    "beta":    beta_val,
                    "roc_adj": roc_adj,
                    "low_vol": low_vol_factor,
                    "quality": quality_factor,
                    "pead":    pead_factor,
                    "realized_vol": realized_vol,
                    "is_emerging": is_emerging_leader,
                })
            except Exception:
                continue

        if len(candidates) < 5:
            rows.append({
                "week_start": wk.strftime("%Y-%m-%d"),
                "symbol": "CASH", "sector": "market",
                "score": 0.0, "roc20": 0.0, "rs": 0.0,
                "adx": 0.0, "regime": regime,
            })
            cash_weeks += 1
            continue

        # ── PHASE 1 CHANGE [2]: DYNAMIC SCORE WEIGHTS ────────────────
        # Scoring dengan bobot yang sudah disesuaikan ke regime
        df_m = pd.DataFrame(candidates)

        def norm(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn) if (mx - mn) > 1e-6 else pd.Series(0.5, index=s.index)

        # Momentum composite (bobot regime-aware seperti sebelumnya)
        momentum_score = (
            w_roc20  * norm(df_m["roc20"])   +
            w_rs     * norm(df_m["rs"])       +
            w_adx    * norm(df_m["adx"])      +
            w_roc_adj* norm(df_m["roc_adj"])  +
            w_roc60  * norm(df_m["roc60"])
        )

        # ── TIER 2 [E]: MULTI-FACTOR BLEND ───────────────────────────
        # Momentum 70% + low-vol 15% + quality 15%.
        # Saat momentum crash, low-vol & quality menahan portfolio.
        # PEAD ditambahkan: momentum tetap dominan, PEAD beri tilt ke
        # saham dengan earnings-surprise drift positif (backtestable proxy).
        FACTOR_MOM, FACTOR_LOWVOL, FACTOR_QUAL, FACTOR_PEAD = 0.65, 0.12, 0.13, 0.10
        df_m["score"] = (
            FACTOR_MOM    * momentum_score        +
            FACTOR_LOWVOL * norm(df_m["low_vol"]) +
            FACTOR_QUAL   * norm(df_m["quality"])  +
            FACTOR_PEAD   * norm(df_m["pead"])
        )

        # ── PHASE 1 CHANGE [5]: DYNAMIC QUALITY GATE ─────────────────
        # Hitung rolling median score 20 minggu terakhir (per regime)
        recent = recent_scores_by_regime[regime]
        if len(recent) >= 5:
            # Hanya enforce gate setelah cukup data historis
            median_threshold = float(np.median(recent[-QUALITY_GATE_WINDOW:]))
            # Saring: hanya kandidat dengan score di atas median threshold
            df_qualified = df_m[df_m["score"] >= median_threshold]
        else:
            df_qualified = df_m  # belum ada data cukup, ambil semua

        if len(df_qualified) < 3:
            # Kualitas minggu ini di bawah norma historis → CASH
            rows.append({
                "week_start": wk.strftime("%Y-%m-%d"),
                "symbol": "CASH", "sector": "market",
                "score": 0.0, "roc20": 0.0, "rs": 0.0,
                "adx": 0.0, "regime": regime,
            })
            cash_weeks += 1
            # Update rolling scores dengan nilai rendah
            recent_scores_by_regime[regime].append(float(df_m["score"].median()))
            continue

        # ── V31 FIX: DYNAMIC SECTOR CAP + EMERGING LEADER PRIORITY ───
        # Masalah lama: MAX_PER_SECTOR=3 membuang NVDA di AI boom 2023
        # (sektor tech penuh, NVDA tergeser). Tapi cap ADA alasannya
        # (2022 tech crash). Solusi: cap ADAPTIF berdasarkan breadth sektor.
        #
        # Logika: kalau satu sektor punya BANYAK saham lolos gate (breadth
        # kuat = sektor sedang memimpin), naikkan cap-nya. Data-driven,
        # bukan hardcode "tech 2023". Kalau 2027 energy memimpin, sama saja.
        df_qualified = df_qualified.copy()
        df_qualified["sector"] = df_qualified["sym"].apply(get_sector)
        df_sorted = df_qualified.sort_values("score", ascending=False)

        # Hitung breadth per sektor (berapa saham lolos gate per sektor)
        sector_breadth = df_qualified["sector"].value_counts().to_dict()

        def sector_cap_for(sec: str) -> int:
            """Cap dinamis: sektor dengan breadth kuat dapat cap lebih tinggi."""
            breadth = sector_breadth.get(sec, 0)
            if breadth >= 8:      # sektor sangat dominan (AI boom style)
                return 6
            elif breadth >= 5:    # sektor memimpin
                return 5
            else:                 # normal → cap ketat (proteksi rotasi)
                return 3

        MAX_TOTAL = 18  # naik dari 15 (universe lebih besar)
        sector_count: dict[str, int] = defaultdict(int)
        selected_rows = []

        # PRIORITAS 1: emerging leaders masuk dulu (kebal sector cap)
        emerging = df_sorted[df_sorted.get("is_emerging", False) == True] \
                   if "is_emerging" in df_sorted.columns else df_sorted.iloc[0:0]
        for _, r in emerging.iterrows():
            selected_rows.append(r)
            sector_count[r["sector"]] += 1
            if len(selected_rows) >= MAX_TOTAL:
                break

        # PRIORITAS 2: sisanya by score dengan sector cap dinamis
        selected_syms = {r["sym"] for r in selected_rows}
        for _, r in df_sorted.iterrows():
            if r["sym"] in selected_syms:
                continue
            sec = r["sector"]
            if sector_count[sec] < sector_cap_for(sec):
                selected_rows.append(r)
                sector_count[sec] += 1
            if len(selected_rows) >= MAX_TOTAL:
                break

        final = pd.DataFrame(selected_rows)

        # Update rolling score buffer
        if not df_m.empty:
            recent_scores_by_regime[regime].append(float(df_m["score"].median()))

        for _, r in final.iterrows():
            rows.append({
                "week_start": wk.strftime("%Y-%m-%d"),
                "symbol":  r["sym"],
                "sector":  r["sector"],
                "score":   round(r["score"],  4),
                "roc20":   round(r["roc20"],  2),
                "roc5":    round(r["roc5"],   2),
                "roc60":   round(r["roc60"],  2),
                "rs":      round(r["rs"],     2),
                "adx":     round(r["adx"],    2),
                "roc_adj": round(r["roc_adj"],3),
                "realized_vol": round(r.get("realized_vol", 0.05), 5),
                "pead": round(r.get("pead", 0.0), 4),
                "regime":  regime,
            })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info(f"✅ Saved {len(rows)} records → {out_csv}")
    logger.info(f"   Coverage  : {rows[0]['week_start']} → {rows[-1]['week_start']}")
    logger.info(f"   CASH weeks: {cash_weeks}/{len(weeks)} "
                f"({cash_weeks/len(weeks)*100:.1f}%)")

    # Ringkasan regime distribution
    df_out = pd.DataFrame(rows)
    if "regime" in df_out.columns:
        regime_dist = df_out.drop_duplicates("week_start")["regime"].value_counts()
        logger.info(f"   Regime dist:\n{regime_dist.to_string()}")


if __name__ == "__main__":
    generate(datetime(2015, 1, 1), datetime.now())
