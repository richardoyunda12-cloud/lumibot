"""
check_setup.py — Cek semua yang dibutuhkan sebelum run ablation test
=====================================================================
Jalankan: python check_setup.py
"""

import sys, os, subprocess

OK   = "✅"
WARN = "⚠️ "
ERR  = "❌"
SEP  = "─" * 60

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def check(label, ok, note=""):
    icon = OK if ok else ERR
    print(f"  {icon} {label}" + (f"  ← {note}" if note else ""))
    return ok

all_ok = True

# ── 1. PYTHON VERSION ─────────────────────────────────────────────────
section("1. Python Version")
v = sys.version_info
ok = v.major == 3 and v.minor >= 10
all_ok &= check(f"Python {v.major}.{v.minor}.{v.micro}", ok,
                "butuh 3.10+" if not ok else "")

# ── 2. PACKAGE DEPENDENCIES ───────────────────────────────────────────
section("2. Package Dependencies")
packages = {
    "lumibot":    "pip install lumibot",
    "pandas":     "pip install pandas",
    "numpy":      "pip install numpy",
    "pandas_ta":  "pip install pandas-ta",
    "yfinance":   "pip install yfinance",
    "groq":       "pip install groq",
    "openai":     "pip install openai",
}
for pkg, install_cmd in packages.items():
    try:
        __import__(pkg)
        ok = True
    except ImportError:
        ok = False
    all_ok &= check(pkg, ok, install_cmd if not ok else "")

# ── 3. SCRIPT FILES ───────────────────────────────────────────────────
section("3. Script Files (di folder ini)")
scripts = {
    # Scanner
    "generate_universe_v27.py":  "Scanner baseline (momentum-only)",
    "generate_universe_v28.py":  "Scanner Tier1 (+ mid-cap)",
    "generate_universe_v29.py":  "Scanner Tier2 (+ multi-factor)",
    # Engine
    "citadel_v8s30_tier123.py":  "Engine utama (Tier 1+2+3)",
    # Committee
    "agent_committee.py":        "AI Committee (Groq/Cerebras/Mistral)",
    "agent_cache.py":            "Cache & audit log",
    "agent_prompts.py":          "System prompts 5 agent",
    "news_proxy.py":             "News proxy dari price action",
    # Test
    "ablation_test.py":          "Ablation test runner",
    "test_v8s30_critical.py":    "Critical period test",
}
for fname, desc in scripts.items():
    exists = os.path.exists(fname)
    all_ok &= check(f"{fname:<40} {desc}", exists,
                    "FILE TIDAK ADA — download dari Claude" if not exists else "")

# ── 4. UNIVERSE CSV ───────────────────────────────────────────────────
section("4. Universe CSV Cache")
csv_files = {
    "cache/weekly_universe_history_v27.csv": "Scanner baseline — generate_universe_v27.py",
    "cache/weekly_universe_history_v28.csv": "Scanner Tier1  — generate_universe_v28.py",
    "cache/weekly_universe_history_v29.csv": "Scanner Tier2  — generate_universe_v29.py",
}
for fpath, gen_cmd in csv_files.items():
    exists = os.path.exists(fpath)
    if exists:
        import pandas as pd
        try:
            df = pd.read_csv(fpath)
            rows = len(df)
            check(f"{fpath}", True, f"{rows:,} rows")
        except Exception:
            check(f"{fpath}", False, "corrupt — regenerate")
            all_ok = False
    else:
        all_ok &= check(f"{fpath}", False,
                        f"BELUM ADA — jalankan: python {gen_cmd.split('—')[1].strip()}")

# ── 5. API KEYS ───────────────────────────────────────────────────────
section("5. API Keys (environment variables)")
api_keys = {
    "CEREBRAS_API_KEY": "Primary — 1M token/hari (console.cerebras.ai)",
    "MISTRAL_API_KEY":  "Secondary — 1B token/bulan (console.mistral.ai)",
    "GROQ_API_KEY":     "Tertiary — 100K token/hari (console.groq.com)",
    "GEMINI_API_KEY":   "Quaternary — opsional (aistudio.google.com)",
}
has_any_key = False
for key, desc in api_keys.items():
    val = os.environ.get(key)
    present = bool(val)
    if present:
        has_any_key = True
    masked = val[:8] + "..." if val else "TIDAK ADA"
    icon = OK if present else WARN
    print(f"  {icon} {key:<22} {masked:<15} {desc}")

if not has_any_key:
    print(f"\n  {ERR} Tidak ada API key sama sekali → committee akan fallback ke scanner top-N")
    print(f"     Set minimal satu key, contoh Windows CMD:")
    print(f"     set CEREBRAS_API_KEY=csk_xxxxx")
    print(f"     set GROQ_API_KEY=gsk_xxxxx")
    all_ok = False

# ── 6. LOGS & CACHE DIR ───────────────────────────────────────────────
section("6. Folder Output")
dirs = ["logs", "cache", "cache/agent_verdicts"]
for d in dirs:
    os.makedirs(d, exist_ok=True)
    check(f"folder {d}/", True, "OK / dibuat")

# ── 7. TEST KONEKSI API ───────────────────────────────────────────────
section("7. Test Koneksi API (cek 1 call kecil)")
for provider, env_key, pkg, model in [
    ("Cerebras", "CEREBRAS_API_KEY", "openai", "gpt-oss-120b"),
    ("Mistral",  "MISTRAL_API_KEY",  "openai", "mistral-small-latest"),
    ("Groq",     "GROQ_API_KEY",     "groq",   "llama-3.3-70b-versatile"),
]:
    api_key = os.environ.get(env_key)
    if not api_key:
        print(f"  {WARN} {provider:<10} skip (tidak ada key)")
        continue
    try:
        if pkg == "groq":
            from groq import Groq
            client = Groq(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":"reply: OK"}],
                max_tokens=5, timeout=10,
            )
        else:
            from openai import OpenAI
            base = ("https://api.cerebras.ai/v1" if provider == "Cerebras"
                    else "https://api.mistral.ai/v1")
            client = OpenAI(api_key=api_key, base_url=base)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":"reply: OK"}],
                max_tokens=5, timeout=10,
            )
        check(f"{provider:<10} ({model})", True, "connected ✓")
    except Exception as e:
        err = str(e)[:60]
        check(f"{provider:<10} ({model})", False, err)
        all_ok = False

# ── RINGKASAN ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
if all_ok:
    print(f"  {OK} SEMUA CHECK LULUS — siap jalankan ablation_test.py")
    print(f"\n  Urutan run:")
    print(f"    1. python generate_universe_v27.py  (kalau belum ada)")
    print(f"    2. python generate_universe_v28.py  (kalau belum ada)")
    print(f"    3. python generate_universe_v29.py  (kalau belum ada)")
    print(f"    4. python ablation_test.py")
else:
    print(f"  {ERR} ADA YANG KURANG — fix dulu sebelum run")
    print(f"\n  Quick install semua package:")
    print(f"    pip install lumibot pandas numpy pandas-ta yfinance groq openai")
print(f"{'='*60}\n")
