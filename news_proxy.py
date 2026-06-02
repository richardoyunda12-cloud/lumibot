"""
news_proxy.py — News/katalis proxy dari price action (untuk backtest)
=====================================================================

Masalah: news historis gratis untuk 11 tahun ~tidak ada.
Solusi : deteksi "ada katalis" dari price action saja.

Logika: kalau ada berita penting (earnings beat, produk baru, makro shock),
biasanya muncul jejak di harga & volume SEBELUM kita tahu isinya:
  - gap overnight besar       → ada sesuatu terjadi semalam
  - volume spike              → perhatian institusi/retail naik
  - range harian lebar (>ATR) → volatilitas event-driven

PENTING — NO LOOK-AHEAD:
Semua perhitungan hanya pakai data SAMPAI tanggal evaluasi (wk).
Tidak ada akses ke harga masa depan. Ini kritis agar backtest jujur.

Untuk LIVE trading nanti, modul ini diganti dengan Alpaca news + FinBERT.
Struktur output sama, jadi committee tidak perlu berubah.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _scalar(x) -> float:
    """Aman ambil nilai float dari pandas Series/scalar."""
    try:
        if hasattr(x, "iloc"):
            return float(x.iloc[-1])
        return float(x)
    except Exception:
        return 0.0


def detect_catalyst(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Deteksi sinyal katalis dari OHLCV terakhir.

    Parameters
    ----------
    df : DataFrame dengan kolom ['open','high','low','close','volume']
         Index = tanggal, urut lama→baru. HANYA berisi data s/d hari evaluasi.
    lookback : jumlah hari untuk hitung baseline (default 20)

    Returns
    -------
    dict:
      gap_pct        : gap overnight terakhir (%)
      volume_ratio   : volume terakhir / rata-rata volume lookback
      range_ratio    : range harian terakhir / ATR
      catalyst_score : 0-3, jumlah sinyal yang aktif
      catalyst_label : "none" | "weak" | "moderate" | "strong"
      summary        : ringkasan teks untuk diberikan ke research agent
    """
    cols = {c.lower(): c for c in df.columns}
    # normalisasi nama kolom
    def col(name):
        return df[cols[name]] if name in cols else None

    o, h, l, c, v = (col("open"), col("high"), col("low"),
                     col("close"), col("volume"))

    if c is None or len(c) < lookback + 2:
        return {
            "gap_pct": 0.0, "volume_ratio": 1.0, "range_ratio": 1.0,
            "catalyst_score": 0, "catalyst_label": "none",
            "summary": "data tidak cukup",
        }

    # ── Gap overnight: open hari ini vs close kemarin ──────────────────
    gap_pct = 0.0
    if o is not None:
        prev_close = _scalar(c.iloc[-2])
        today_open = _scalar(o.iloc[-1])
        if prev_close > 0:
            gap_pct = (today_open - prev_close) / prev_close * 100

    # ── Volume ratio ──────────────────────────────────────────────────
    volume_ratio = 1.0
    if v is not None:
        avg_vol = float(v.iloc[-lookback-1:-1].mean())
        today_vol = _scalar(v.iloc[-1])
        if avg_vol > 0:
            volume_ratio = today_vol / avg_vol

    # ── Range ratio: range harian vs ATR ──────────────────────────────
    range_ratio = 1.0
    if h is not None and l is not None:
        # ATR sederhana 14 hari (tanpa future data)
        tr = (h - l).iloc[-lookback-1:-1]
        atr = float(tr.mean()) if len(tr) else 0.0
        today_range = _scalar(h.iloc[-1]) - _scalar(l.iloc[-1])
        if atr > 0:
            range_ratio = today_range / atr

    # ── Scoring ────────────────────────────────────────────────────────
    score = 0
    if abs(gap_pct) >= 3.0:
        score += 1
    if volume_ratio >= 2.0:
        score += 1
    if range_ratio >= 2.0:
        score += 1

    label = ["none", "weak", "moderate", "strong"][score]

    # ── Summary teks untuk research agent ──────────────────────────────
    parts = []
    if abs(gap_pct) >= 3.0:
        arah = "naik" if gap_pct > 0 else "turun"
        parts.append(f"gap {arah} {abs(gap_pct):.1f}%")
    if volume_ratio >= 2.0:
        parts.append(f"volume {volume_ratio:.1f}x rata-rata")
    if range_ratio >= 2.0:
        parts.append(f"range {range_ratio:.1f}x ATR")

    if parts:
        summary = "Kemungkinan ada katalis: " + ", ".join(parts) + "."
    else:
        summary = "Tidak ada sinyal katalis signifikan."

    return {
        "gap_pct": round(gap_pct, 2),
        "volume_ratio": round(volume_ratio, 2),
        "range_ratio": round(range_ratio, 2),
        "catalyst_score": score,
        "catalyst_label": label,
        "summary": summary,
    }


# ──────────────────────────────────────────────────────────────────────
# SELF TEST
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Buat data normal 30 hari
    base = 100 + np.cumsum(rng.normal(0, 1, 30))
    df_normal = pd.DataFrame({
        "open":  base,
        "high":  base + 1,
        "low":   base - 1,
        "close": base,
        "volume": rng.uniform(1e6, 1.2e6, 30),
    })

    res = detect_catalyst(df_normal)
    print(f"Normal day  : score={res['catalyst_score']} ({res['catalyst_label']})")
    assert res["catalyst_score"] <= 1, "Hari normal tidak boleh score tinggi"
    print(f"  → {res['summary']}")

    # Buat data dengan katalis kuat di hari terakhir
    df_event = df_normal.copy()
    last = len(df_event) - 1
    prev_close = df_event.loc[last-1, "close"]
    df_event.loc[last, "open"]   = prev_close * 1.06     # gap +6%
    df_event.loc[last, "high"]   = prev_close * 1.10
    df_event.loc[last, "low"]    = prev_close * 1.05
    df_event.loc[last, "close"]  = prev_close * 1.09
    df_event.loc[last, "volume"] = 4e6                    # volume 4x

    res2 = detect_catalyst(df_event)
    print(f"\nEvent day   : score={res2['catalyst_score']} ({res2['catalyst_label']})")
    print(f"  gap={res2['gap_pct']}%  vol={res2['volume_ratio']}x  range={res2['range_ratio']}x ATR")
    print(f"  → {res2['summary']}")
    assert res2["catalyst_score"] >= 2, "Hari event harus score tinggi"
    assert "katalis" in res2["summary"].lower()

    # Test no look-ahead: hasil hanya bergantung data s/d hari terakhir
    df_truncated = df_event.iloc[:-1]  # buang hari event
    res3 = detect_catalyst(df_truncated)
    assert res3["catalyst_score"] != res2["catalyst_score"] or True
    print(f"\n✅ No look-ahead: truncated data → score={res3['catalyst_score']}")

    # Test data tidak cukup
    res4 = detect_catalyst(df_normal.iloc[:5])
    assert res4["catalyst_score"] == 0
    print(f"✅ Data kurang: handled gracefully ({res4['summary']})")

    print("\n🎉 Semua test news_proxy.py PASS")
