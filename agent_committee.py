"""
agent_committee.py — AI Agent Committee dengan Multi-Provider Fallback
=======================================================================

Architecture:
  2 API call per minggu (bukan 5):
  - Call 1: Research + Bull + Bear (digabung, 1 prompt)
  - Call 2: Risk + Chairman (1 prompt)

  Ini memotong token usage ~60% dibanding 5 call terpisah.

Provider priority (semua free tier):
  1. Cerebras  — 1M token/hari, Llama 3.1 70B, paling generous
  2. Groq      — 100K token/hari, Llama 3.3 70B, tercepat
  3. Gemini    — 1.500 req/hari, Gemini 2.5 Flash, frontier model
  4. Fallback  — scanner top-N deterministik (0 API call)

Cache: verdict disimpan per minggu, run ke-2 dst = 0 API call.
"""

from __future__ import annotations
import os, json, time, hashlib
from typing import Optional

from agent_cache import AgentCache


# ── PROVIDER CONFIG ────────────────────────────────────────────────────
PROVIDERS = [
    {
        "name":     "cerebras",
        "env_key":  "CEREBRAS_API_KEY",
        "base_url": "https://api.cerebras.ai/v1",
        "model":    "gpt-oss-120b",   # model terkuat Cerebras saat ini
        "pkg":      "openai",
    },
    {
        "name":     "mistral",
        "env_key":  "MISTRAL_API_KEY",
        "base_url": "https://api.mistral.ai/v1",
        "model":    "mistral-small-latest",  # hemat token, cukup untuk reasoning
        "pkg":      "openai",
    },
    {
        "name":     "groq",
        "env_key":  "GROQ_API_KEY",
        "base_url": None,
        "model":    "llama-3.3-70b-versatile",
        "pkg":      "groq",
    },
    {
        "name":     "gemini",
        "env_key":  "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model":    "gemini-2.0-flash",
        "pkg":      "openai",
    },
]


# ── PROMPT 1: Research + Bull + Bear (digabung) ────────────────────────
SYSTEM_CALL1 = """You are an investment committee analyst. Given candidate stocks and market data, produce THREE analyses in ONE JSON response.

Return ONLY valid JSON (no markdown):
{
  "evidence": {"SYM": "one neutral sentence on momentum/rs/adx/catalyst"},
  "bull": {
    "top_picks": ["SYM1","SYM2"],
    "conviction": {"SYM1":"high","SYM2":"medium"},
    "reasoning": "2 sentences why"
  },
  "bear": {
    "avoid": ["SYM"],
    "exit_now": ["SYM"],
    "reasoning": "2 sentences on key risks"
  }
}

Rules:
- Only use symbols from the candidate list. Never invent symbols.
- Bear can have empty lists if conditions are good.
- In BEAR/high_vol regime: bull top_picks should be empty or 1-2 max.
- In BULL regime with strong momentum: bull top_picks should be 3-5."""


# ── PROMPT 2: Risk + Chairman (digabung) ──────────────────────────────
SYSTEM_CALL2 = """You are the risk manager and chairman of a trading committee. Make the FINAL trading decision.

Return ONLY valid JSON (no markdown):
{
  "buy": [{"symbol":"SYM","size_mult":0.9}],
  "exit": ["SYM"],
  "reasoning": "2-3 sentences explaining decision and how you weighed bull vs bear"
}

Critical rules:
- size_mult range: 0.5 to 1.2
- Never exceed max_new_positions from risk assessment
- Always honor bear exit_now list
- IMPORTANT: In strong BULL regime with no red flags, buy ALL bull top_picks at size_mult 0.9-1.1. Do NOT be conservative in clear bull markets — missing bull runs is costly.
- In BEAR/high_vol: max 1-2 buys or zero, size_mult 0.4-0.6
- In vshape recovery: buy bull picks at size_mult 0.7-0.9 (opportunity but volatile)
- Only use symbols from the candidate list."""


# ── TIER 3: MACRO / PORTFOLIO MANAGER AGENT ───────────────────────────
SYSTEM_MACRO = """You are the chief portfolio manager (macro). You do NOT pick stocks.
You set the PORTFOLIO-LEVEL risk dial for this week based on market context.

You receive: regime, SPY trend, market breadth (% of candidates with positive
momentum), recent realized volatility, cross-sectional dispersion, current drawdown.

Your judgment: even in a BULL regime, if breadth is narrowing, volatility creeping
up, or dispersion widening (signs of a fragile market), reduce leverage. In a clean
broad-based uptrend, lean in.

Output a leverage multiplier that ADJUSTS the base regime leverage:
- leverage_mult 1.0 = use base regime leverage as-is
- leverage_mult > 1.0 (max 1.3) = lean in (clean broad uptrend, low vol)
- leverage_mult < 1.0 (min 0.4) = pull back (narrow breadth, rising vol, fragile)

Return ONLY valid JSON (no markdown):
{
  "regime_confidence": 0.0-1.0,
  "leverage_mult": 0.4-1.3,
  "risk_posture": "aggressive" | "neutral" | "defensive",
  "reasoning": "2 sentences on market health and why this dial"
}"""


def _build_macro_user(regime: str, spy_trend: str, breadth: float,
                      recent_vol: float, dispersion: float,
                      current_dd: float) -> str:
    return (
        f"Regime: {regime}\n"
        f"SPY trend: {spy_trend}\n"
        f"Market breadth (% candidates positive momentum): {breadth:.0%}\n"
        f"Recent daily volatility: {recent_vol:.2%}\n"
        f"Cross-sectional dispersion: {dispersion:.2%}\n"
        f"Current portfolio drawdown: {current_dd:.1%}"
    )


def _build_call1_user(candidates: list, regime: str) -> str:
    lines = [f"Market regime: {regime}", "", "Candidates:"]
    for c in candidates:
        sym = c.get("sym", c.get("symbol", "?"))
        lines.append(
            f"- {sym}: roc5={c.get('roc5',0):.1f}% roc20={c.get('roc20',0):.1f}% "
            f"roc60={c.get('roc60',0):.1f}% rs={c.get('rs',0):.1f} "
            f"adx={c.get('adx',0):.0f} sector={c.get('sector','?')} "
            f"catalyst={c.get('catalyst_summary','none')}"
        )
    return "\n".join(lines)


def _build_call2_user(call1_result: dict, candidates: list[str],
                      holdings: list[str], portfolio_value: float,
                      current_dd: float, regime: str,
                      n_active: int, cash: float) -> str:
    # Hitung max_new_positions dari dd dan regime
    if regime == "BEAR":
        max_new = 0
        size_hint = 0.0
    elif current_dd < -0.15:
        max_new = 2
        size_hint = 0.5
    elif current_dd < -0.08:
        max_new = 3
        size_hint = 0.7
    elif regime == "vshape":
        max_new = 4
        size_hint = 0.8
    else:  # BULL healthy
        max_new = 5
        size_hint = 1.0

    return (
        f"Portfolio: ${portfolio_value:,.0f} | DD: {current_dd:.1%} | "
        f"Regime: {regime} | Active positions: {n_active} | Cash: ${cash:,.0f}\n"
        f"Risk assessment: max_new_positions={max_new}, suggested_size_mult={size_hint}\n"
        f"Available candidates: {candidates}\n"
        f"Current holdings: {holdings}\n\n"
        f"RESEARCH + BULL analysis:\n{json.dumps(call1_result.get('bull',{}), indent=2)}\n\n"
        f"BEAR analysis:\n{json.dumps(call1_result.get('bear',{}), indent=2)}"
    )


class AgentCommittee:
    def __init__(
        self,
        cache_dir: str = "cache/agent_verdicts",
        timeout: int = 30,
        sleep_between_calls: float = 2.0,
        verbose: bool = True,
    ):
        self.cache = AgentCache(cache_dir=cache_dir)
        self.timeout = timeout
        self.sleep_between_calls = sleep_between_calls
        self.verbose = verbose
        self._clients = {}   # lazy init per provider

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [committee] {msg}")

    # ── CLIENT FACTORY ─────────────────────────────────────────────────
    def _get_client(self, provider: dict):
        name = provider["name"]
        if name in self._clients:
            return self._clients[name]

        api_key = os.environ.get(provider["env_key"])
        if not api_key:
            return None

        try:
            if provider["pkg"] == "groq":
                from groq import Groq
                client = Groq(api_key=api_key)
            else:
                from openai import OpenAI
                kwargs = {"api_key": api_key}
                if provider["base_url"]:
                    kwargs["base_url"] = provider["base_url"]
                client = OpenAI(**kwargs)
            self._clients[name] = client
            return client
        except ImportError:
            return None
        except Exception:
            return None

    # ── SINGLE CALL dengan exponential backoff ─────────────────────────
    def _call(self, provider: dict, system: str, user: str) -> Optional[dict]:
        client = self._get_client(provider)
        if client is None:
            return None

        model = provider["model"]
        for attempt in range(2):  # max 1 retry per provider, lalu pindah
            wait = 2 ** attempt * self.sleep_between_calls  # 2s, 4s
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    temperature=0.2,
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content
                parsed = json.loads(raw)
                time.sleep(self.sleep_between_calls)
                return parsed
            except json.JSONDecodeError:
                self._log(f"{provider['name']} JSON invalid (attempt {attempt+1})")
                time.sleep(wait)
            except Exception as e:
                err = str(e)
                if "404" in err or "does not exist" in err.lower():
                    # Model tidak ada di provider ini → langsung pindah, tidak retry
                    self._log(f"{provider['name']} model tidak ada, skip")
                    return None
                if "429" in err or "rate_limit" in err.lower():
                    # Ambil wait time dari error message kalau bisa
                    import re
                    m = re.search(r"try again in ([\d.]+)(m|s)", err)
                    if m:
                        secs = float(m.group(1)) * (60 if m.group(2) == "m" else 1)
                        wait = min(secs + 2, 30)  # cap 30s
                    self._log(f"{provider['name']} rate limit, wait {wait:.0f}s")
                    time.sleep(wait)
                else:
                    self._log(f"{provider['name']} error: {err[:80]}")
                    time.sleep(wait)
        return None

    # ── TRY ALL PROVIDERS ──────────────────────────────────────────────
    def _call_with_fallback(self, system: str, user: str,
                            call_name: str) -> tuple[Optional[dict], str]:
        for provider in PROVIDERS:
            result = self._call(provider, system, user)
            if result is not None:
                return result, provider["name"]
        self._log(f"{call_name}: semua provider gagal")
        return None, "all_failed"

    # ── FALLBACK DETERMINISTIK ─────────────────────────────────────────
    @staticmethod
    def _make_fallback(candidates: list, week: str, regime: str,
                       top_n: int = 5) -> dict:
        syms = [c.get("sym", c.get("symbol")) for c in candidates[:top_n]]
        syms = [s for s in syms if s and s != "CASH"]
        return {
            "week": week, "regime": regime,
            "chairman_verdict": {
                "buy":  [{"symbol": s, "size_mult": 1.0} for s in syms],
                "exit": [],
                "reasoning": "FALLBACK: semua provider gagal, pakai scanner top-N.",
            },
            "source": "fallback_scanner",
        }

    # ── VALIDATE VERDICT ───────────────────────────────────────────────
    @staticmethod
    def _validate(verdict: dict, valid_syms: set) -> bool:
        cv = verdict.get("chairman_verdict", {})
        if not isinstance(cv, dict):
            return False
        if "buy" not in cv:
            return False
        for b in cv.get("buy", []):
            sym = b.get("symbol") if isinstance(b, dict) else b
            if sym not in valid_syms:
                return False
        return True

    # ── MAIN: decide() ─────────────────────────────────────────────────
    def decide(self, candidates: list, week: str, regime: str,
               portfolio_value: float, current_dd: float,
               holdings: list[str], cash: float,
               market_context: dict = None,
               use_cache: bool = True) -> dict:

        symbols = [c.get("sym", c.get("symbol")) for c in candidates]
        symbols = [s for s in symbols if s and s != "CASH"]

        # Default macro verdict (kalau macro gagal / tidak ada context)
        default_macro = {"regime_confidence": 0.7, "leverage_mult": 1.0,
                         "risk_posture": "neutral",
                         "reasoning": "default (macro tidak dievaluasi)"}

        # ── Cache ──────────────────────────────────────────────────────
        key = self.cache.make_key(week, symbols, regime)
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                self._log(f"{week}: cache hit")
                orig = cached.get("source", "committee")
                cv = cached.get("chairman_verdict", {"buy":[],"exit":[],"reasoning":""})
                if not cached.get("_has_action"):
                    cv = {"buy":[],"exit":[],"reasoning":cached.get("reason","")}
                macro = cached.get("macro", default_macro)
                return {"week":week,"regime":regime,
                        "chairman_verdict":cv,
                        "macro": macro,
                        "source":f"{orig}_cached"}

        # ── No candidates ──────────────────────────────────────────────
        if not symbols:
            record = {
                "week": week, "regime": regime, "n_candidates": 0,
                "portfolio_value": portfolio_value, "current_dd": current_dd,
                "chairman_verdict": {"buy":[],"exit":[],
                    "reasoning":"Tidak ada kandidat dari scanner → CASH."},
                "macro": default_macro,
                "source": "no_candidates",
            }
            self.cache.save(key, record)
            return record

        self._log(f"{week} ({regime}): {len(symbols)} kandidat → 3-call committee")

        # ── CALL 0 (MACRO): portfolio-level risk dial ──────────────────
        macro = default_macro
        if market_context:
            macro_user = _build_macro_user(
                regime,
                market_context.get("spy_trend", "?"),
                market_context.get("breadth", 0.5),
                market_context.get("recent_vol", 0.02),
                market_context.get("dispersion", 0.02),
                current_dd,
            )
            macro_result, _ = self._call_with_fallback(SYSTEM_MACRO, macro_user, "macro")
            if macro_result and "leverage_mult" in macro_result:
                # Clamp leverage_mult ke range aman
                lm = float(macro_result.get("leverage_mult", 1.0))
                macro_result["leverage_mult"] = max(0.4, min(1.3, lm))
                macro = macro_result

        # ── CALL 1: Research + Bull + Bear ─────────────────────────────
        c1_user   = _build_call1_user(candidates, regime)
        c1_result, prov1 = self._call_with_fallback(SYSTEM_CALL1, c1_user, "call1")

        if c1_result is None:
            record = self._make_fallback(candidates, week, regime)
            record.update({"n_candidates":len(symbols),
                           "portfolio_value":portfolio_value,
                           "current_dd":current_dd, "macro": macro})
            self.cache.save(key, record)
            return record

        # ── CALL 2: Risk + Chairman ────────────────────────────────────
        c2_user   = _build_call2_user(
            c1_result, symbols, holdings,
            portfolio_value, current_dd, regime,
            len(holdings), cash
        )
        c2_result, prov2 = self._call_with_fallback(SYSTEM_CALL2, c2_user, "call2")

        if c2_result is None:
            record = self._make_fallback(candidates, week, regime)
            record.update({"n_candidates":len(symbols),
                           "portfolio_value":portfolio_value,
                           "current_dd":current_dd, "macro": macro})
            self.cache.save(key, record)
            return record

        # ── Susun record ───────────────────────────────────────────────
        record = {
            "week": week, "regime": regime,
            "n_candidates": len(symbols),
            "portfolio_value": portfolio_value,
            "current_dd": current_dd,
            "candidates_from_scanner": symbols,
            "macro_output": macro,
            "call1_result": c1_result,
            "chairman_verdict": c2_result,
            "macro": macro,
            "source": f"committee_{prov1}+{prov2}",
        }

        # ── Validate ───────────────────────────────────────────────────
        if not self._validate(record, set(symbols)):
            self._log(f"{week}: verdict invalid → fallback")
            record = self._make_fallback(candidates, week, regime)
            record.update({"n_candidates":len(symbols),
                           "portfolio_value":portfolio_value,
                           "current_dd":current_dd, "macro": macro})
            self.cache.save(key, record)
            return record

        self.cache.save(key, record)
        n_buy  = len(c2_result.get("buy", []))
        n_exit = len(c2_result.get("exit", []))
        self._log(f"{week}: OK → buy={n_buy} exit={n_exit} "
                  f"lev_mult={macro['leverage_mult']:.1f} ({macro['risk_posture']})")
        return {"week":week,"regime":regime,
                "chairman_verdict":c2_result,
                "macro": macro,
                "source":f"committee_{prov1}+{prov2}"}


# ── SELF TEST ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, shutil

    tmp = tempfile.mkdtemp()
    committee = AgentCommittee(cache_dir=tmp, verbose=True)

    cands = [
        {"sym":"NVDA","roc5":4.2,"roc20":12.0,"roc60":3.0,
         "rs":5.1,"adx":28,"sector":"Tech","catalyst_summary":"gap +5% vol 3x"},
        {"sym":"AMD","roc5":3.1,"roc20":8.0,"roc60":1.5,
         "rs":3.2,"adx":22,"sector":"Tech","catalyst_summary":"none"},
        {"sym":"MSFT","roc5":2.0,"roc20":6.0,"roc60":4.0,
         "rs":2.8,"adx":25,"sector":"Tech","catalyst_summary":"none"},
    ]

    # Deteksi provider yang tersedia
    available = []
    for p in PROVIDERS:
        if os.environ.get(p["env_key"]):
            available.append(p["name"])
    print(f"Provider tersedia: {available if available else ['none — fallback mode']}")
    print()

    print("── Test 1: decide() ──")
    v = committee.decide(
        candidates=cands, week="2020-03-27", regime="vshape",
        portfolio_value=98500, current_dd=-0.12,
        holdings=["BA"], cash=30000,
    )
    print(f"source  = {v['source']}")
    print(f"buy     = {v['chairman_verdict'].get('buy')}")
    print(f"exit    = {v['chairman_verdict'].get('exit')}")
    print(f"reason  = {v['chairman_verdict'].get('reasoning','')[:150]}")

    print("\n── Test 2: cache hit ──")
    import time as _t
    t0 = _t.time()
    v2 = committee.decide(
        candidates=cands, week="2020-03-27", regime="vshape",
        portfolio_value=98500, current_dd=-0.12,
        holdings=["BA"], cash=30000,
    )
    elapsed = _t.time() - t0
    assert "cached" in v2["source"], f"Expected cached, got: {v2['source']}"
    assert elapsed < 1.0
    print(f"✅ Cache hit: {v2['source']} ({elapsed:.2f}s)")

    print("\n── Test 3: no candidates → CASH ──")
    v3 = committee.decide(
        candidates=[], week="2022-06-12", regime="BEAR",
        portfolio_value=90000, current_dd=-0.08,
        holdings=[], cash=90000,
    )
    assert v3["chairman_verdict"]["buy"] == []
    print(f"✅ No candidates: {v3['chairman_verdict']['reasoning'][:60]}")

    print(f"\nCache stats: {committee.cache.stats()}")
    shutil.rmtree(tmp)
    print("\n🎉 Semua test PASS")