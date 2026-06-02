"""
run_paper_trading.py — Paper Trading via Alpaca + Telegram
===========================================================
Menjalankan Citadel LIVE (paper) dengan:
  - Broker: Alpaca paper account (eksekusi + data)
  - Stock selection + exit: deterministik (terbukti robust)
  - Position sizing: AI dengan fundamental+news real-time (use_ai_sizing)
  - Notifikasi: semua keputusan → Telegram real-time
"""

import os

# ⚠️ PASTIKAN NAMA FILE DI BAWAH INI SESUAI DENGAN ENGINE TERAKHIR ANDA
# Jika Anda menggunakan engine V33 dengan 3-Lapis Exit, ubah importnya menjadi:
# from citadel_v8s33_1x import CitadelLevelQuantV8S33 as StrategyClass
from citadel_baseline_v1 import CitadelLevelQuantV8S30 as StrategyClass

# Import Telegram Notifier yang sudah Anda unggah
from telegram_notifier import TelegramNotifier

def check_env():
    """Pastikan semua kredensial ada sebelum mulai."""
    required = {
        "ALPACA_API_KEY":     "Broker (alpaca.markets → Paper → API Keys)",
        "ALPACA_API_SECRET":  "Broker secret",
    }
    optional = {
        "CEREBRAS_API_KEY":   "AI sizing (tanpa ini → deterministik)",
        "TELEGRAM_BOT_TOKEN": "Notifikasi Telegram",
        "TELEGRAM_CHAT_ID":   "Chat ID Telegram",
    }
    print("=" * 60)
    print("  CEK KREDENSIAL")
    print("=" * 60)
    missing = []
    for k, desc in required.items():
        v = os.environ.get(k)
        print(f"  {'OK ' if v else 'XX '} {k:<20} {desc}")
        if not v:
            missing.append(k)
    print("  ── opsional ──")
    for k, desc in optional.items():
        v = os.environ.get(k)
        print(f"  {'OK ' if v else '-- '} {k:<20} {desc}")
    print()
    return len(missing) == 0

def main():
    if not check_env():
        print("❌ Kredensial broker (ALPACA_API_KEY / SECRET) wajib. Set dulu.")
        return

    if not os.path.exists("cache/weekly_universe_history_v32.csv"):
        print("❌ universe v32 belum ada. Jalankan: python generate_universe_v32.py")
        return

    # ── Setup broker Alpaca (paper) ──────────────────────────────────
    from lumibot.brokers import Alpaca
    from lumibot.traders import Trader

    ALPACA_CONFIG = {
        "API_KEY":    os.environ["ALPACA_API_KEY"],
        "API_SECRET": os.environ["ALPACA_API_SECRET"],
        "PAPER":      True,   # WAJIB True — paper trading, bukan uang nyata
    }

    broker = Alpaca(ALPACA_CONFIG)

    # ── Parameter strategi untuk paper ───────────────────────────────
    params = dict(StrategyClass.parameters)
    params.update({
        "use_agent":       False,   
        "use_ai_sizing":   True,    
        "use_leverage":    False,   
        "bear_mode":       "cash",
        "log_decisions":   True,
        "use_telegram":    True,    
        "telegram_level":  "all",   
        "run_name":        "paper",
        "universe_csv":    "cache/weekly_universe_history_v32.csv",
    })

    strategy = StrategyClass(
        broker=broker,
        parameters=params,
    )

    # ── SUNTIKKAN TELEGRAM NOTIFIER KE ENGINE ────────────────────────
    if params.get("use_telegram"):
        strategy.notifier = TelegramNotifier(
            is_live=True, 
            level=params.get("telegram_level", "all"), 
            verbose=True
        )
    else:
        strategy.notifier = None

    trader = Trader()
    trader.add_strategy(strategy)

    print("=" * 60)
    print("  🚀 PAPER TRADING DIMULAI")
    print("=" * 60)
    print("  Broker      : Alpaca (PAPER)")
    print("  Stock select: deterministik (scanner v32 + exit 3-lapis)")
    print("  Sizing      : AI (fundamental+news real-time)")
    print("  Notifikasi  : Telegram (level=all)")
    print("  Stop        : Ctrl+C")
    print("=" * 60)

    # Kirim notif start ke Telegram menggunakan fungsi safe check
    if getattr(strategy, "notifier", None) and strategy.notifier.enabled:
        strategy.notifier.notify_custom(
            "🟢 *CITADEL PAPER TRADING DIMULAI*\n"
            "Mode: Deterministik + AI Sizing\n"
            "Semua keputusan eksekusi akan dikirim ke sini."
        )

    trader.run_all()

if __name__ == "__main__":
    main()