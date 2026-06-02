"""
ai_sizing.py — AI Position Sizing Committee
============================================
TUGAS BARU AI (beda dari committee lama):
  AI TIDAK memilih saham (scanner sudah pilih, deterministik & terbukti).
  AI menilai CONVICTION tiap kandidat → size multiplier.

  Tujuan: konsentrasi di pemenang sejati (emerging leader + PEAD kuat),
  porsi kecil untuk momentum biasa. Menyerang kelemahan kita di bull:
  terlalu terdiversifikasi, tidak konsentrasi di pemenang.

INPUT (backtest = teknikal + PEAD, JUJUR tanpa look-ahead):
  - momentum (roc20, rs, adx), pead proxy, emerging-leader flag, score
LIVE (nanti): + fundamental (earnings/revenue growth) + news/catalyst

OUTPUT: conviction (high/medium/low) + size_mult (0.5-1.5) per saham.
size_mult mengali bobot deterministik, lalu di-renormalisasi.

Multi-provider fallback: Cerebras → Mistral → Groq → deterministik (size_mult=1.0).
"""

from __future__ import annotations
import os, json, time


SYSTEM_SIZING = """You are a portfolio position-sizing specialist. You do NOT pick stocks.
The candidates are ALREADY selected by a proven quantitative scanner. Your ONLY job:
assign conviction to each, so capital concentrates in the strongest leaders.

You receive per candidate: momentum (roc20), relative strength (rs), trend strength
(adx), PEAD signal (price reaction to earnings — proxy for earnings surprise),
and whether it's flagged as an emerging leader (early-stage breakout).

Judgment:
- HIGH conviction (size_mult 1.3-1.5): emerging leader with strong PEAD + high RS.
  A genuine leader breaking out with earnings-driven momentum (the next NVDA profile).
- MEDIUM conviction (size_mult 0.9-1.1): solid momentum but not exceptional.
- LOW conviction (size_mult 0.5-0.8): momentum present but weak PEAD or extended/late.
  These might be momentum traps (pump without earnings substance — the PTON profile).

Key insight: PEAD distinguishes real leaders (earnings-backed) from pumps (no substance).
High RS + emerging + strong PEAD = lean in. Weak PEAD + extended = size down.

Return ONLY valid JSON (no markdown), array of objects:
[
  {"symbol": "X", "conviction": "high|medium|low", "size_mult": 0.5-1.5,
   "reasoning": "1 short sentence"}
]
Every input symbol must appear exactly once. size_mult must be 0.5-1.5."""


def _build_user(candidates: list, regime: str) -> str:
    lines = [f"Market regime: {regime}", "", "Candidates to size:"]
    for c in candidates:
        sym = c.get("symbol", c.get("sym"))
        if sym == "CASH":
            continue
        lines.append(
            f"- {sym}: roc20={c.get('roc20',0):.0f}% rs={c.get('rs',0):.0f} "
            f"adx={c.get('adx',0):.0f} pead={c.get('pead',0):.3f} "
            f"emerging={c.get('is_emerging',False)}"
        )
    return "\n".join(lines)


SYSTEM_SIZING_LIVE = """You are a portfolio position-sizing specialist. You do NOT pick stocks.
Candidates are ALREADY selected by a proven scanner. Assign conviction so capital
concentrates in GENUINE leaders, not momentum traps.

You receive per candidate: technical signals (momentum, RS, ADX, PEAD, emerging flag),
PLUS fundamental data (earnings/revenue growth) and recent news headlines.

Key judgment — distinguish real leaders from pumps:
- GENUINE LEADER (size_mult 1.3-1.5): strong momentum + accelerating fundamentals
  (earnings/revenue growth) OR a clear forward catalyst in news (new product, demand
  surge, sector inflection). The NVDA-in-early-2023 profile: technicals strong, news
  shows AI inflection, even if last reported earnings still lagging.
- SOLID (size_mult 0.9-1.1): good momentum, stable fundamentals, no red flags.
- TRAP (size_mult 0.5-0.8): momentum WITHOUT fundamental support — guidance cuts,
  margin deterioration, no growth catalyst, hype-only news. The PTON profile.

Weigh FORWARD-LOOKING signals (news catalysts, fundamental trajectory), not just price.
A stock can be a leader before earnings confirm it, if news shows a real catalyst.

Return ONLY valid JSON array:
[{"symbol":"X","conviction":"high|medium|low","size_mult":0.5-1.5,"reasoning":"1 sentence"}]
Every input symbol exactly once. size_mult 0.5-1.5."""


def _build_user_live(candidates: list, regime: str) -> str:
    lines = [f"Market regime: {regime}", "", "Candidates to size:"]
    for c in candidates:
        sym = c.get("symbol", c.get("sym"))
        if sym == "CASH":
            continue
        line = (f"- {sym}: roc20={c.get('roc20',0):.0f}% rs={c.get('rs',0):.0f} "
                f"adx={c.get('adx',0):.0f} pead={c.get('pead',0):.3f} "
                f"emerging={c.get('is_emerging',False)}")
        # Fundamental kalau ada
        eg, rg = c.get("fund_earnings_growth"), c.get("fund_revenue_growth")
        if eg is not None or rg is not None:
            line += f"\n  FUNDAMENTAL: earnings_growth={eg}% revenue_growth={rg}%"
        # News kalau ada
        if c.get("news_titles"):
            line += "\n  NEWS: " + " | ".join(c["news_titles"][:3])
        lines.append(line)
    return "\n".join(lines)


class AISizingCommittee:
    def __init__(self, verbose: bool = True, live_mode: bool = False):
        self.verbose = verbose
        self.live_mode = live_mode   # True = enrich dgn fundamental+news (LIVE)
        self.providers = self._setup_providers()

    def _setup_providers(self):
        provs = []
        # Cerebras (primary)
        if os.environ.get("CEREBRAS_API_KEY"):
            provs.append(("cerebras", "https://api.cerebras.ai/v1",
                          os.environ["CEREBRAS_API_KEY"], "gpt-oss-120b"))
        # Mistral (secondary)
        if os.environ.get("MISTRAL_API_KEY"):
            provs.append(("mistral", "https://api.mistral.ai/v1",
                          os.environ["MISTRAL_API_KEY"], "mistral-small-latest"))
        # Groq (tertiary)
        if os.environ.get("GROQ_API_KEY"):
            provs.append(("groq", None, os.environ["GROQ_API_KEY"],
                          "llama-3.3-70b-versatile"))
        return provs

    def _log(self, msg):
        if self.verbose:
            print(f"  [ai_sizing] {msg}")

    def _call(self, system, user, base_url, api_key, model, is_groq):
        try:
            if is_groq:
                from groq import Groq
                client = Groq(api_key=api_key)
            else:
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url=base_url)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=1500, temperature=0.3, timeout=30,
            )
            txt = resp.choices[0].message.content.strip()
            txt = txt.replace("```json", "").replace("```", "").strip()
            return json.loads(txt)
        except Exception as e:
            self._log(f"call gagal: {str(e)[:60]}")
            return None

    def decide(self, candidates: list, regime: str) -> tuple[list, str]:
        """
        Returns (sizing_list, provider).
        sizing_list: list dict {symbol, conviction, size_mult, reasoning}
        Fallback: semua size_mult=1.0 (= deterministik) kalau semua provider gagal.
        """
        syms = [c.get("symbol", c.get("sym")) for c in candidates
                if c.get("symbol", c.get("sym")) != "CASH"]
        if not syms:
            return [], "no_candidates"

        # ── LIVE MODE: enrich dengan fundamental + news (real-time) ──
        # Hanya saat live_mode — di backtest ini look-ahead, jadi dilewati.
        system_prompt = SYSTEM_SIZING
        if self.live_mode:
            candidates = self._enrich_live(candidates)
            system_prompt = SYSTEM_SIZING_LIVE
            user = _build_user_live(candidates, regime)
        else:
            user = _build_user(candidates, regime)

        for name, base_url, api_key, model in self.providers:
            for attempt in range(2):
                result = self._call(system_prompt, user, base_url, api_key,
                                    model, is_groq=(name == "groq"))
                if result and isinstance(result, list):
                    # Validasi & clamp
                    valid = {}
                    for item in result:
                        s = item.get("symbol")
                        if s in syms:
                            sm = float(item.get("size_mult", 1.0))
                            valid[s] = {
                                "symbol": s,
                                "conviction": item.get("conviction", "medium"),
                                "size_mult": max(0.5, min(1.5, sm)),
                                "reasoning": item.get("reasoning", ""),
                            }
                    # Pastikan semua simbol ada (yg hilang → size_mult 1.0)
                    for s in syms:
                        if s not in valid:
                            valid[s] = {"symbol": s, "conviction": "medium",
                                        "size_mult": 1.0, "reasoning": "default (missing from AI)"}
                    self._log(f"{regime}: {len(syms)} saham sized via {name}")
                    return list(valid.values()), name
                time.sleep(1.5 * (attempt + 1))

        # Semua gagal → deterministik
        self._log(f"semua provider gagal → deterministik (size_mult=1.0)")
        fallback = [{"symbol": s, "conviction": "medium", "size_mult": 1.0,
                     "reasoning": "fallback deterministik"} for s in syms]
        return fallback, "fallback_determ"

    def _enrich_live(self, candidates: list) -> list:
        """
        LIVE-ONLY: tambah fundamental (yfinance) + news (yfinance.news) ke tiap
        kandidat. Di live, data terkini = point-in-time (jujur). JANGAN di backtest.
        """
        try:
            from fundamental_live import get_fundamental_score
        except Exception:
            get_fundamental_score = None

        import yfinance as yf
        for c in candidates:
            sym = c.get("symbol", c.get("sym"))
            if not sym or sym == "CASH":
                continue
            # Fundamental
            if get_fundamental_score:
                try:
                    f = get_fundamental_score(sym)
                    if f.get("available"):
                        c["fund_earnings_growth"] = f.get("earnings_growth")
                        c["fund_revenue_growth"]  = f.get("revenue_growth")
                except Exception:
                    pass
            # News (judul terbaru via yfinance)
            try:
                tk = yf.Ticker(sym)
                news = getattr(tk, "news", []) or []
                titles = []
                for n in news[:3]:
                    t = n.get("title") or n.get("content", {}).get("title", "")
                    if t:
                        titles.append(t)
                if titles:
                    c["news_titles"] = titles
            except Exception:
                pass
        return candidates


if __name__ == "__main__":
    # Smoke test (fallback mode kalau tidak ada API key)
    committee = AISizingCommittee()
    cands = [
        {"symbol": "NVDA", "roc20": 28, "rs": 23, "adx": 41, "pead": 0.15, "is_emerging": True},
        {"symbol": "XYZ",  "roc20": 12, "rs": 3,  "adx": 22, "pead": -0.02, "is_emerging": False},
    ]
    sizing, prov = committee.decide(cands, "trending_bull")
    print(f"\nProvider: {prov}")
    for s in sizing:
        print(f"  {s['symbol']}: {s['conviction']} size={s['size_mult']} — {s['reasoning']}")
