"""
decision_logger.py — Logging keputusan 4-lapis untuk Citadel
=============================================================
Merekam SEMUA keputusan agar bisa diaudit & didiagnosis:
  Log 1: Scanner   — kandidat per minggu + skor + faktor + gate
  Log 2: AI Sizing — input ke AI, output conviction/size, reasoning
  Log 3: Execution — tiap entry/exit + alasan + P&L
  Log 4: Master    — ringkasan 1 baris per minggu

Output ke logs/decisions/ : CSV (tabular) + JSONL (detail AI reasoning).
"""

from __future__ import annotations
import os, json, csv
from datetime import datetime



def _clean(text) -> str:
    """Bersihkan karakter Unicode problematik (non-breaking hyphen, dll)."""
    if text is None:
        return ""
    s = str(text)
    # Ganti karakter Unicode umum yang bikin masalah encoding
    repl = {
        "\u2011": "-", "\u2013": "-", "\u2014": "-",  # hyphens/dashes
        "\u2018": "'", "\u2019": "'",                   # quotes
        "\u201c": '"', "\u201d": '"',                   # double quotes
        "\u2026": "...",                                  # ellipsis
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    # Fallback: buang karakter non-ASCII yang tersisa
    return s.encode("ascii", "ignore").decode("ascii")


class DecisionLogger:
    def __init__(self, run_name: str = "run", base_dir: str = "logs/decisions"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(base_dir, f"{run_name}_{ts}")
        os.makedirs(self.dir, exist_ok=True)

        # Path tiap log
        self.scanner_csv = os.path.join(self.dir, "1_scanner.csv")
        self.sizing_csv  = os.path.join(self.dir, "2_ai_sizing.csv")
        self.sizing_jsonl= os.path.join(self.dir, "2_ai_sizing_detail.jsonl")
        self.exec_csv    = os.path.join(self.dir, "3_execution.csv")
        self.master_csv  = os.path.join(self.dir, "4_master.csv")

        self._init_headers()
        print(f"📁 Decision log: {self.dir}")

    def _init_headers(self):
        with open(self.scanner_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "week", "regime", "symbol", "score", "rank",
                "roc20", "rs", "adx", "pead", "is_emerging",
                "sector", "passed_via"
            ])
        with open(self.sizing_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "week", "regime", "symbol",
                "score", "roc20", "pead",
                "conviction", "size_mult", "weight_ai", "weight_determ",
                "provider", "reasoning_short"
            ])
        with open(self.exec_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "date", "action", "symbol", "price", "qty",
                "pct_portfolio", "exit_layer", "pnl_pct", "reason"
            ])
        with open(self.master_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "week", "regime", "n_candidates", "n_positions",
                "effective_leverage", "avg_conviction", "cash_pct",
                "portfolio_value", "source"
            ])

    # ── Log 1: Scanner ────────────────────────────────────────────────
    def log_scanner(self, week: str, regime: str, candidates: list):
        """candidates: list dict dgn skor & faktor, sudah terurut by score."""
        with open(self.scanner_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for rank, c in enumerate(candidates, 1):
                sym = c.get("symbol", c.get("sym", "?"))
                if sym == "CASH":
                    continue
                w.writerow([
                    week, regime, sym,
                    round(c.get("composite", c.get("score", 0)), 4), rank,
                    round(c.get("roc20", 0), 2), round(c.get("rs", 0), 2),
                    round(c.get("adx", 0), 1), round(c.get("pead", 0), 4),
                    c.get("is_emerging", False), c.get("sector", "?"),
                    "emerging" if c.get("is_emerging") else "normal",
                ])

    # ── Log 2: AI Sizing ──────────────────────────────────────────────
    def log_sizing(self, week: str, regime: str, decisions: list,
                   provider: str, raw_response: dict = None):
        """
        decisions: list dict {symbol, score, roc20, pead, conviction,
                   size_mult, weight_ai, weight_determ, reasoning}
        """
        with open(self.sizing_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for d in decisions:
                reason = _clean(d.get("reasoning", ""))[:120]
                w.writerow([
                    week, regime, d.get("symbol", "?"),
                    round(d.get("score", 0), 4), round(d.get("roc20", 0), 2),
                    round(d.get("pead", 0), 4),
                    d.get("conviction", "?"), round(d.get("size_mult", 1.0), 3),
                    round(d.get("weight_ai", 0), 4),
                    round(d.get("weight_determ", 0), 4),
                    provider, reason,
                ])
        # Detail penuh ke JSONL (termasuk reasoning panjang + raw response)
        with open(self.sizing_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "week": week, "regime": regime, "provider": provider,
                "decisions": decisions, "raw_response": raw_response,
            }, default=str) + "\n")

    # ── Log 3: Execution ──────────────────────────────────────────────
    def log_exec(self, date: str, action: str, symbol: str, price: float,
                 qty: int, pct_portfolio: float = 0.0,
                 exit_layer: str = "", pnl_pct: float = 0.0, reason: str = ""):
        with open(self.exec_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                date, action, symbol, round(price, 2), qty,
                round(pct_portfolio, 4), exit_layer,
                round(pnl_pct, 4), reason,
            ])

    # ── Log 4: Master ─────────────────────────────────────────────────
    def log_master(self, week: str, regime: str, n_candidates: int,
                   n_positions: int, effective_leverage: float,
                   avg_conviction: float, cash_pct: float,
                   portfolio_value: float, source: str):
        with open(self.master_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                week, regime, n_candidates, n_positions,
                round(effective_leverage, 3), round(avg_conviction, 3),
                round(cash_pct, 4), round(portfolio_value, 2), source,
            ])


# Smoke test
if __name__ == "__main__":
    log = DecisionLogger("smoketest")
    log.log_scanner("2023-05-01", "trending_bull", [
        {"symbol": "NVDA", "composite": 0.92, "roc20": 28, "rs": 23,
         "adx": 41, "pead": 0.15, "is_emerging": False, "sector": "Tech"},
    ])
    log.log_sizing("2023-05-01", "trending_bull", [
        {"symbol": "NVDA", "score": 0.92, "roc20": 28, "pead": 0.15,
         "conviction": "high", "size_mult": 1.5, "weight_ai": 0.25,
         "weight_determ": 0.15, "reasoning": "Emerging AI leader, strong PEAD"},
    ], provider="cerebras")
    log.log_exec("2023-05-01", "BUY", "NVDA", 300.5, 80, 0.25, reason="high conviction")
    log.log_master("2023-05-01", "trending_bull", 8, 5, 1.0, 1.2, 0.05, 125000, "committee_cerebras")
    print("✅ Smoke test OK — cek folder:", log.dir)
