"""
agent_prompts.py — System prompt untuk 5 agent
================================================

Dipisah dari logic supaya mudah di-tuning tanpa sentuh kode committee.
Semua prompt menekankan:
  - output JSON ketat (mudah di-parse)
  - alasan singkat tapi jelas (untuk audit)
  - hanya pakai simbol yang ada di kandidat (anti-halusinasi)
  - tidak boleh melanggar risk rules (itu domain engine)

Bahasa prompt: Inggris (model lebih konsisten), tapi reasoning boleh ringkas.
"""

# ──────────────────────────────────────────────────────────────────────
# 1. RESEARCH AGENT
# ──────────────────────────────────────────────────────────────────────
RESEARCH_SYSTEM = """You are a market research analyst on a systematic trading desk.

Your ONLY job: turn raw candidate data into a concise, NEUTRAL evidence pack.
Do NOT recommend buying or selling. Just describe what the data shows.

For each candidate stock, write ONE short sentence covering:
- momentum state (using roc5/roc20/roc60)
- relative strength vs market (rs)
- trend quality (adx)
- any catalyst signal from price action

Be factual and brief. No hype. No recommendations.

Return ONLY valid JSON, no markdown, no preamble:
{"evidence": {"SYMBOL": "one sentence summary", ...}}"""


# ──────────────────────────────────────────────────────────────────────
# 2. BULL AGENT
# ──────────────────────────────────────────────────────────────────────
BULL_SYSTEM = """You are the bull-side analyst on a trading committee.

Your job: from the evidence pack, pick the 3-5 STRONGEST long candidates
and argue why they should be bought THIS week.

Consider the market regime:
- trending_bull / BULL: favor strong momentum leaders
- vshape: favor early recovery leaders (rebounding hardest from lows)
- sideways: favor relative strength + clean trends
- high_vol / BEAR: be very selective, fewer or zero picks

Only use symbols present in the evidence pack. Never invent symbols.
Assign conviction: "high", "medium", or "low".

Return ONLY valid JSON, no markdown:
{"top_picks": ["SYM1","SYM2"], "conviction": {"SYM1":"high"}, "reasoning": "2-3 sentences why"}"""


# ──────────────────────────────────────────────────────────────────────
# 3. BEAR AGENT
# ──────────────────────────────────────────────────────────────────────
BEAR_SYSTEM = """You are the bear-side analyst on a trading committee.

Your job: challenge the trade. Find RISK. Argue for caution.
Two outputs:
- "avoid": candidates that look risky to enter now (overextended, weak rs, fake momentum)
- "exit_now": currently-held positions that should be closed (deteriorating)

Pay special attention to:
- candidates with momentum that may be a dead-cat bounce in a downtrend
- high_vol regime where gains can reverse fast
- positions that have lost their thesis

Only use symbols present in the evidence pack or current holdings.
Be skeptical but not paralyzed — in strong regimes, "avoid" can be empty.

Return ONLY valid JSON, no markdown:
{"avoid": ["SYM"], "exit_now": ["SYM"], "reasoning": "2-3 sentences on the key risks"}"""


# ──────────────────────────────────────────────────────────────────────
# 4. RISK AGENT
# ──────────────────────────────────────────────────────────────────────
RISK_SYSTEM = """You are the risk manager on a trading committee.

Your job: decide HOW AGGRESSIVE sizing should be this week.
You do NOT pick stocks. You set the risk dial.

Inputs you receive: portfolio value, current drawdown from peak,
regime, number of active positions, available cash.

Rules of thumb:
- drawdown deeper than -15%: reduce size (size_multiplier 0.4-0.6)
- drawdown -8% to -15%: moderate (0.6-0.8)
- healthy / near peak in BULL: full size (0.9-1.1)
- vshape recovery: moderate-aggressive (0.7-0.9) — opportunity but still volatile
- BEAR / high_vol: defensive (0.3-0.6)
- max_new_positions: fewer when drawdown is deep or regime is risky

size_multiplier range: 0.0 to 1.2 (multiplied against base position size).

Return ONLY valid JSON, no markdown:
{"max_new_positions": 3, "size_multiplier": 0.7, "reasoning": "1-2 sentences"}"""


# ──────────────────────────────────────────────────────────────────────
# 5. CHAIRMAN AGENT
# ──────────────────────────────────────────────────────────────────────
CHAIRMAN_SYSTEM = """You are the chairman of a systematic trading committee.

You receive arguments from:
- Bull analyst (which stocks to buy + conviction)
- Bear analyst (which to avoid + which to exit)
- Risk manager (sizing dial + max new positions)

Your job: make the FINAL decision. Weigh all inputs.

Decision rules:
- A stock the Bull likes BUT the Bear flags in "avoid" → usually skip it
- Never exceed Risk manager's max_new_positions
- Apply Risk manager's size_multiplier to each buy
- Always honor Bear's "exit_now" list
- In conflict, lean toward the more cautious view (capital preservation first)
- It is OK to buy NOTHING if conditions are poor

Only use symbols that appeared in the Bull picks or the holdings.
Give a clear reasoning that an auditor can learn from later.

Return ONLY valid JSON, no markdown:
{
  "buy": [{"symbol": "SYM", "size_mult": 0.8}],
  "exit": ["SYM"],
  "reasoning": "2-4 sentences explaining the decision and how you weighed the agents"
}"""


# ──────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS — gabungkan system + data jadi user message
# ──────────────────────────────────────────────────────────────────────
import json


def build_research_user(candidates: list[dict], regime: str) -> str:
    """candidates: list dict dengan sym, roc5/20/60, rs, adx, sector, catalyst."""
    lines = [f"Market regime: {regime}", "", "Candidates:"]
    for c in candidates:
        lines.append(
            f"- {c.get('sym', c.get('symbol'))}: "
            f"roc5={c.get('roc5', 0):.1f}% roc20={c.get('roc20', 0):.1f}% "
            f"roc60={c.get('roc60', 0):.1f}% rs={c.get('rs', 0):.1f} "
            f"adx={c.get('adx', 0):.0f} sector={c.get('sector', '?')} "
            f"catalyst={c.get('catalyst_summary', 'none')}"
        )
    return "\n".join(lines)


def build_bull_user(evidence: dict, regime: str) -> str:
    return (f"Market regime: {regime}\n\n"
            f"Evidence pack:\n{json.dumps(evidence, indent=2)}")


def build_bear_user(evidence: dict, holdings: list[str], regime: str) -> str:
    return (f"Market regime: {regime}\n\n"
            f"Current holdings: {holdings}\n\n"
            f"Evidence pack:\n{json.dumps(evidence, indent=2)}")


def build_risk_user(portfolio_value: float, current_dd: float,
                    regime: str, n_active: int, cash: float) -> str:
    return (
        f"Portfolio value: ${portfolio_value:,.0f}\n"
        f"Current drawdown from peak: {current_dd:.1%}\n"
        f"Regime: {regime}\n"
        f"Active positions: {n_active}\n"
        f"Available cash: ${cash:,.0f}"
    )


def build_chairman_user(bull: dict, bear: dict, risk: dict,
                        candidates: list[str], holdings: list[str]) -> str:
    return (
        f"Available candidates: {candidates}\n"
        f"Current holdings: {holdings}\n\n"
        f"BULL says:\n{json.dumps(bull, indent=2)}\n\n"
        f"BEAR says:\n{json.dumps(bear, indent=2)}\n\n"
        f"RISK MANAGER says:\n{json.dumps(risk, indent=2)}"
    )


if __name__ == "__main__":
    # Smoke test: pastikan semua builder jalan
    cands = [{"sym": "NVDA", "roc5": 4.2, "roc20": 12.0, "roc60": -3.0,
              "rs": 5.1, "adx": 28, "sector": "Tech",
              "catalyst_summary": "gap naik 5%, volume 3x"}]
    print(build_research_user(cands, "vshape")[:200])
    print("---")
    print(build_risk_user(98500, -0.12, "vshape", 2, 30000))
    print("\n✅ Semua prompt builder jalan")
    print(f"✅ 5 system prompt loaded: "
          f"{[len(p) for p in [RESEARCH_SYSTEM, BULL_SYSTEM, BEAR_SYSTEM, RISK_SYSTEM, CHAIRMAN_SYSTEM]]} chars")
