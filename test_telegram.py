"""
test_telegram.py — Verifikasi koneksi bot Telegram
Jalankan sekali untuk pastikan bot bekerja sebelum trading.
  set TELEGRAM_BOT_TOKEN=8796xxxxx:AAFxxxxx
  set TELEGRAM_CHAT_ID=600366409
  python test_telegram.py
"""
import os
from telegram_notifier import TelegramNotifier

def main():
    print("Cek environment variable...")
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat  = os.environ.get("TELEGRAM_CHAT_ID", "")
    print(f"  TELEGRAM_BOT_TOKEN: {'ada (' + token[:12] + '...)' if token else 'TIDAK ADA'}")
    print(f"  TELEGRAM_CHAT_ID:   {chat if chat else 'TIDAK ADA'}")
    if not token or not chat:
        print("\nSet dulu kedua variable, jalankan lagi.")
        return
    tg = TelegramNotifier(verbose=True)
    print("\nMengirim pesan tes...")
    ok = tg._send("*Citadel Bot* terhubung!\nNotifikasi trading akan dikirim ke sini.")
    if ok:
        print("BERHASIL - cek Telegram kamu.")
        tg.buy("NVDA", 480.50, 50, 0.25, "high", "Emerging AI leader")
        tg.exit("PTON", 32.10, 100, -0.18, "momentum break")
        tg.regime_change("BULL", "BEAR", "SPY below SMA200")
        tg.daily_summary("2026-06-01", 128500, 0.012, 5, 0.05, ["NVDA","META","AVGO"])
        print("5 pesan contoh dikirim.")
    else:
        print("GAGAL. Cek: 1) token benar 2) chat_id benar 3) sudah /start bot di Telegram?")

if __name__ == "__main__":
    main()
