"""
fundamental_live.py — Fundamental Momentum (LIVE-ONLY)
=======================================================

⚠️  PERINGATAN PENTING — JANGAN DIPAKAI DI BACKTEST ⚠️
-------------------------------------------------------
Modul ini mengambil data fundamental TERKINI via yfinance.
Di backtest historis, ini menyebabkan LOOK-AHEAD BIAS:
data yang dikembalikan adalah angka final/revisi SEKARANG,
bukan angka yang tersedia pada tanggal historis tersebut.

Karena itu modul ini HANYA untuk LIVE / PAPER TRADING,
di mana "data terkini" memang benar-benar point-in-time.

Saat live trading, panggil ini untuk memberi tilt fundamental
ke skor scanner — saham dengan earnings/revenue acceleration
mendapat boost.

Anomali yang dimanfaatkan:
- Earnings growth acceleration (fundamental momentum)
- Revenue growth (durable demand)
- Margin expansion (operating leverage)

Semua terbukti robust puluhan tahun di literatur akademik.
"""

from __future__ import annotations
import yfinance as yf


def get_fundamental_score(symbol: str) -> dict:
    """
    Ambil skor fundamental TERKINI untuk satu saham.
    HANYA untuk live trading — bukan backtest.

    Returns dict:
      earnings_growth : YoY earnings growth (%)
      revenue_growth  : YoY revenue growth (%)
      margin_trend    : perubahan net margin (pp)
      fundamental_score : skor gabungan -1..+1 (untuk tilt scanner)
      available       : bool, apakah data tersedia
    """
    out = {
        "earnings_growth": None, "revenue_growth": None,
        "margin_trend": None, "fundamental_score": 0.0,
        "available": False,
    }
    try:
        tk = yf.Ticker(symbol)
        info = tk.info

        # Earnings & revenue growth (YoY) dari info
        eg = info.get("earningsGrowth")        # fraction, e.g. 0.25 = +25%
        rg = info.get("revenueGrowth")         # fraction
        pm = info.get("profitMargins")         # fraction

        if eg is not None:
            out["earnings_growth"] = eg * 100
        if rg is not None:
            out["revenue_growth"] = rg * 100
        if pm is not None:
            out["margin_trend"] = pm * 100

        # Skor gabungan sederhana (normalisasi kasar)
        score = 0.0
        n = 0
        if eg is not None:
            # earnings growth: +30% → +1.0, -30% → -1.0
            score += max(-1.0, min(1.0, eg / 0.30)); n += 1
        if rg is not None:
            score += max(-1.0, min(1.0, rg / 0.30)); n += 1
        if pm is not None and pm > 0:
            # margin positif beri sedikit boost
            score += min(0.5, pm / 0.20); n += 1

        if n > 0:
            out["fundamental_score"] = score / n
            out["available"] = True

    except Exception:
        pass
    return out


def apply_fundamental_tilt(candidates: list[dict],
                           tilt_weight: float = 0.15) -> list[dict]:
    """
    Terapkan tilt fundamental ke skor kandidat (LIVE-ONLY).

    candidates : list dict dengan minimal {"symbol"/"sym", "composite"}
    tilt_weight: bobot fundamental dalam skor akhir (default 15%)

    Modifikasi 'composite' in-place dan tambah 'fundamental_score'.
    Saham dengan fundamental kuat naik peringkat, yang lemah turun.
    """
    for c in candidates:
        sym = c.get("symbol") or c.get("sym")
        if not sym or sym == "CASH":
            continue
        fund = get_fundamental_score(sym)
        c["fundamental_score"] = fund["fundamental_score"]
        c["fundamental_available"] = fund["available"]
        if fund["available"]:
            base = c.get("composite", 0.0)
            # Blend: (1-w)*base + w*fundamental, fundamental sudah -1..+1
            c["composite"] = (1 - tilt_weight) * base + tilt_weight * fund["fundamental_score"]
    return candidates


if __name__ == "__main__":
    # Smoke test — HANYA untuk cek koneksi, ini data live
    print("⚠️  Modul LIVE-ONLY — jangan pakai di backtest!\n")
    for sym in ["NVDA", "AAPL", "INTC"]:
        f = get_fundamental_score(sym)
        if f["available"]:
            print(f"{sym}: earnings_growth={f['earnings_growth']:.1f}% "
                  f"revenue_growth={f['revenue_growth']:.1f}% "
                  f"score={f['fundamental_score']:+.2f}")
        else:
            print(f"{sym}: data tidak tersedia")
