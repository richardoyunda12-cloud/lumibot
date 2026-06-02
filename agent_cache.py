"""
agent_cache.py — Cache & audit log untuk AI Agent Committee
============================================================

Dua fungsi sekaligus:
1. CACHE  : simpan verdict per minggu → run ke-2 dst tidak panggil API lagi
2. AUDIT  : simpan reasoning lengkap agar bisa dipelajari kenapa untung/rugi

Aturan penyimpanan (sesuai keputusan):
- Minggu ADA AKSI (buy/exit)  → simpan LENGKAP (semua output + raw prompt/response)
- Minggu CASH / no action     → simpan RINGKAS (week, regime, alasan singkat)

Master CSV mencatat SEMUA minggu (1 baris/minggu) untuk overview.

Key cache = sha256(week + sorted(candidate_symbols) + regime).
Kalau scanner/kandidat berubah → key berubah → auto re-run agent minggu itu.
"""

from __future__ import annotations
import os
import json
import hashlib
import csv
from datetime import datetime
from typing import Optional


class AgentCache:
    def __init__(self, cache_dir: str = "cache/agent_verdicts"):
        self.cache_dir = cache_dir
        self.master_csv = os.path.join(cache_dir, "_master_log.csv")
        os.makedirs(cache_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────
    # KEY GENERATION
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def make_key(week: str, candidate_symbols: list[str], regime: str) -> str:
        """
        Key deterministik dari minggu + kandidat + regime.
        sorted() memastikan urutan simbol tidak mempengaruhi key.
        """
        syms = ",".join(sorted(candidate_symbols))
        raw = f"{week}|{syms}|{regime}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    # ──────────────────────────────────────────────────────────────────
    # READ
    # ──────────────────────────────────────────────────────────────────
    def get(self, key: str) -> Optional[dict]:
        """Baca verdict dari cache. Return None kalau belum ada."""
        path = self._path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    # ──────────────────────────────────────────────────────────────────
    # WRITE
    # ──────────────────────────────────────────────────────────────────
    def save(self, key: str, record: dict) -> None:
        """
        Simpan verdict + audit ke cache.
        record harus minimal punya: week, regime, chairman_verdict.

        Penyimpanan selektif:
        - ada aksi  → simpan record penuh
        - no action → strip raw_prompts & raw_responses & per-agent output
        """
        verdict = record.get("chairman_verdict", {})
        has_action = bool(verdict.get("buy") or verdict.get("exit"))

        if has_action:
            stored = record  # lengkap
        else:
            # ringkas — buang yang berat
            stored = {
                "week":          record.get("week"),
                "regime":        record.get("regime"),
                "n_candidates":  record.get("n_candidates", 0),
                "portfolio_value": record.get("portfolio_value"),
                "current_dd":    record.get("current_dd"),
                "decision":      "no_action",
                "reason":        (verdict.get("reasoning") or "")[:200],
                "source":        record.get("source", "committee"),
            }

        stored["_has_action"] = has_action
        stored["_saved_at"]   = datetime.now().isoformat(timespec="seconds")

        with open(self._path(key), "w", encoding="utf-8") as f:
            json.dump(stored, f, indent=2, ensure_ascii=False)

        self._append_master(record, has_action)

    # ──────────────────────────────────────────────────────────────────
    # MASTER CSV — 1 baris per minggu, semua minggu
    # ──────────────────────────────────────────────────────────────────
    def _append_master(self, record: dict, has_action: bool) -> None:
        verdict = record.get("chairman_verdict", {})
        buy  = verdict.get("buy", [])
        exit_ = verdict.get("exit", [])

        buy_syms  = ";".join(
            b["symbol"] if isinstance(b, dict) else str(b) for b in buy
        )
        size_mults = ";".join(
            f"{b.get('size_mult', 1.0):.2f}" if isinstance(b, dict) else "1.00"
            for b in buy
        )

        row = {
            "week":            record.get("week", ""),
            "regime":          record.get("regime", ""),
            "n_candidates":    record.get("n_candidates", 0),
            "n_buy":           len(buy),
            "n_exit":          len(exit_),
            "buy_symbols":     buy_syms,
            "size_mults":      size_mults,
            "exit_symbols":    ";".join(str(e) for e in exit_),
            "portfolio_value": record.get("portfolio_value", ""),
            "current_dd":      record.get("current_dd", ""),
            "source":          record.get("source", "committee"),
            "chairman_summary": (verdict.get("reasoning") or "")[:300],
        }

        file_exists = os.path.exists(self.master_csv)
        with open(self.master_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    # ──────────────────────────────────────────────────────────────────
    # UTIL
    # ──────────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        """Ringkasan isi cache — berguna untuk cek progress."""
        files = [f for f in os.listdir(self.cache_dir) if f.endswith(".json")]
        n_action = 0
        for fn in files:
            try:
                with open(os.path.join(self.cache_dir, fn)) as f:
                    if json.load(f).get("_has_action"):
                        n_action += 1
            except Exception:
                pass
        return {
            "total_weeks_cached": len(files),
            "action_weeks":       n_action,
            "cash_weeks":         len(files) - n_action,
        }


# ──────────────────────────────────────────────────────────────────────
# SELF TEST
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, shutil

    tmp = tempfile.mkdtemp()
    cache = AgentCache(cache_dir=tmp)

    # Test 1: key deterministik
    k1 = cache.make_key("2020-03-27", ["NVDA", "AMD"], "vshape")
    k2 = cache.make_key("2020-03-27", ["AMD", "NVDA"], "vshape")  # urutan beda
    assert k1 == k2, "Key harus sama meski urutan simbol beda"
    print(f"✅ Key deterministik: {k1}")

    # Test 2: minggu ada aksi → simpan lengkap
    action_record = {
        "week": "2020-03-27", "regime": "vshape", "n_candidates": 15,
        "portfolio_value": 98500, "current_dd": -0.12,
        "research_output": {"evidence": {"NVDA": "gap +5% vol 3x"}},
        "bull_output": {"top_picks": ["NVDA"], "reasoning": "recovery kuat"},
        "bear_output": {"avoid": [], "exit_now": [], "reasoning": "ok"},
        "risk_output": {"size_multiplier": 0.8, "reasoning": "dd masih aman"},
        "chairman_verdict": {
            "buy": [{"symbol": "NVDA", "size_mult": 0.8}],
            "exit": [], "reasoning": "NVDA recovery, masuk"
        },
        "raw_prompts": {"research": "long prompt..."},
        "raw_responses": {"research": "long response..."},
        "source": "committee",
    }
    cache.save(k1, action_record)
    loaded = cache.get(k1)
    assert loaded["_has_action"] is True
    assert "raw_prompts" in loaded, "Minggu aksi harus simpan raw prompt"
    print("✅ Minggu aksi: raw prompt tersimpan")

    # Test 3: minggu CASH → simpan ringkas
    cash_record = {
        "week": "2022-06-12", "regime": "BEAR", "n_candidates": 0,
        "portfolio_value": 90000, "current_dd": -0.08,
        "chairman_verdict": {"buy": [], "exit": [], "reasoning": "Bear, semua CASH"},
        "raw_prompts": {"research": "long prompt..."},
        "source": "committee",
    }
    k3 = cache.make_key("2022-06-12", [], "BEAR")
    cache.save(k3, cash_record)
    loaded_cash = cache.get(k3)
    assert loaded_cash["_has_action"] is False
    assert "raw_prompts" not in loaded_cash, "Minggu CASH tidak simpan raw prompt"
    assert loaded_cash["decision"] == "no_action"
    print("✅ Minggu CASH: ringkas, raw prompt dibuang")

    # Test 4: master CSV
    assert os.path.exists(cache.master_csv)
    with open(cache.master_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2, "Master CSV harus catat semua minggu"
    print(f"✅ Master CSV: {len(rows)} baris (semua minggu tercatat)")

    # Test 5: stats
    s = cache.stats()
    assert s["total_weeks_cached"] == 2
    assert s["action_weeks"] == 1
    assert s["cash_weeks"] == 1
    print(f"✅ Stats: {s}")

    shutil.rmtree(tmp)
    print("\n🎉 Semua test agent_cache.py PASS")
