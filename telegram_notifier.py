"""
telegram_notifier.py — Notifikasi Telegram untuk Citadel (LIVE/PAPER ONLY)
===========================================================================
Otomatis OFF di backtest. Format rapi & informatif untuk dibaca di HP.

Setup:
  set TELEGRAM_BOT_TOKEN=8796xxxxx:AAFxxxxx
  set TELEGRAM_CHAT_ID=600366409
level: "all" | "actions" | "summary"
"""

from __future__ import annotations
import os, urllib.request, urllib.parse, time


def _esc(s):
    """Escape karakter Markdown yang bisa bikin format rusak."""
    if s is None:
        return ""
    s = str(s)
    for ch in ["_", "*", "`", "["]:
        s = s.replace(ch, "")
    return s


class TelegramNotifier:
    def __init__(self, is_live: bool = False, level: str = "all", verbose: bool = True):
        self.token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        self.verbose = verbose
        self.level   = level
        self.enabled = is_live and bool(self.token) and bool(self.chat_id)
        self._last_send = 0.0
        self.min_interval = 0.4
        self._fail_count = 0
        if is_live and not self.enabled:
            print("WARNING Telegram: token/chat_id tidak ada -> OFF")
        elif self.enabled:
            print(f"Telegram notifier AKTIF (level={level})")

    def _send(self, text: str):
        if not self.enabled:
            if self.verbose:
                print(f"[tg-off] {text[:70]}")
            return False
        elapsed = time.time() - self._last_send
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": self.chat_id, "text": text,
            "parse_mode": "Markdown", "disable_web_page_preview": "true",
        }).encode()
        try:
            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req, timeout=10) as resp:
                self._last_send = time.time(); self._fail_count = 0
                return resp.status == 200
        except Exception as e:
            self._fail_count += 1
            if self.verbose and self._fail_count <= 3:
                print(f"[tg-fail] {str(e)[:60]}")
            return False

    def _lvl(self, needed):
        order = {"all": 3, "actions": 2, "summary": 1}
        return order.get(self.level, 3) >= needed

    # ── BUY ───────────────────────────────────────────────────────────
    def notify_buy(self, symbol, price, qty, pct_portfolio, conviction="", reasoning=""):
        if not self._lvl(2): return
        conv_tag = {"high": "🔥 HIGH", "medium": "▫️ MED", "low": "⬇️ LOW"}.get(
            str(conviction).lower(), str(conviction).upper())
        value = price * qty
        txt = (f"🟢 *BELI {_esc(symbol)}*  ({conv_tag})\n"
               f"`Entry ${price:,.2f} × {qty} = ${value:,.0f}`\n"
               f"Porsi: *{pct_portfolio:.1%}* portfolio")
        if reasoning:
            txt += f"\n💡 {_esc(reasoning)}"
        self._send(txt)

    # ── EXIT ──────────────────────────────────────────────────────────
    def notify_exit(self, symbol, price, qty, pnl_pct, exit_layer="", reason="",
                    entry_price=None, entry_date=""):
        if not self._lvl(2): return
        emoji = "🔴" if pnl_pct < 0 else "🟢"
        tag = "RUGI" if pnl_pct < 0 else "UNTUNG"
        layer_label = {
            "momentum break": "Momentum patah (roc20<0)",
            "trailing stop": "Trailing stop ATR",
            "regime": "Regime BEAR → cash",
        }.get(str(exit_layer).lower(), _esc(exit_layer or reason))

        txt = f"{emoji} *JUAL {_esc(symbol)}*  ({tag} {pnl_pct:+.1%})\n"
        # Entry → Exit lengkap
        if entry_price:
            pnl_dollar = (price - entry_price) * qty
            txt += (f"`Entry ${entry_price:,.2f} → Exit ${price:,.2f}`\n"
                    f"Qty: {qty} · P&L: *${pnl_dollar:+,.0f}* ({pnl_pct:+.1%})")
        else:
            txt += f"`${price:,.2f} × {qty}`"
        # Durasi hold
        if entry_date:
            txt += f"\nMasuk: {_esc(entry_date)}"
        txt += f"\nAlasan: {layer_label}"
        self._send(txt)

    # ── REGIME ────────────────────────────────────────────────────────
    def notify_regime(self, new_regime, old_regime="", detail=""):
        if not self._lvl(2): return
        emoji = {"BULL": "🟢", "TRANS": "🟡", "BEAR": "🔴"}.get(str(new_regime).upper(), "⚪")
        arrow = f"{_esc(old_regime)} → " if old_regime else ""
        txt = f"{emoji} *REGIME: {arrow}{_esc(new_regime)}*"
        if detail:
            txt += f"\n{_esc(detail)}"
        self._send(txt)

    # ── AI SIZING ─────────────────────────────────────────────────────
    def notify_sizing(self, week, regime, decisions, provider):
        if not self._lvl(3): return
        ranked = sorted(decisions, key=lambda x: x.get("size_mult", 1), reverse=True)
        n_high = sum(1 for d in ranked if d.get("conviction") == "high")
        n_low  = sum(1 for d in ranked if d.get("conviction") == "low")

        lines = [f"🤖 *AI SIZING* · {_esc(week)} · {_esc(regime)}",
                 f"_{n_high} high · {n_low} low · via {_esc(provider)}_",
                 "━━━━━━━━━━━━━━━━━━━━"]
        for d in ranked:
            conv = str(d.get("conviction", "?")).lower()
            icon = {"high": "🔥", "medium": "▫️", "low": "⬇️"}.get(conv, "•")
            sym  = _esc(d.get("symbol", "?"))
            mult = d.get("size_mult", 1.0)
            lines.append(f"{icon} *{sym}* `×{mult:.2f}`")
            r = d.get("reasoning")
            if r:
                # Reasoning penuh (tidak dipotong tengah kata), max 2 baris
                r = _esc(r)
                if len(r) > 130:
                    r = r[:127] + "..."
                lines.append(f"   _{r}_")
        self._send("\n".join(lines))

    def notify_scanner(self, week, regime, candidates):
        if not self._lvl(3): return
        real = [c for c in candidates if c.get("symbol", c.get("sym")) != "CASH"]
        top = sorted(real, key=lambda x: x.get("composite", x.get("score", 0)), reverse=True)[:8]
        syms = ", ".join(_esc(c.get("symbol", c.get("sym", "?"))) for c in top)
        self._send(f"📡 *SCANNER* · {_esc(week)} · {_esc(regime)}\n"
                   f"{len(real)} kandidat lolos\nTop: {syms}")

    # ── RINGKASAN HARIAN ──────────────────────────────────────────────
    def notify_daily(self, date, portfolio_value, day_return, n_positions, cash_pct, holdings=None):
        if not self._lvl(1): return
        emoji = "📈" if day_return >= 0 else "📉"
        txt = (f"{emoji} *RINGKASAN HARIAN* · {_esc(date)}\n"
               f"━━━━━━━━━━━━━━━━━━━━\n"
               f"💰 Portfolio: *${portfolio_value:,.0f}*\n"
               f"📊 Hari ini: *{day_return:+.2%}*\n"
               f"📌 Posisi: {n_positions} · Cash: {cash_pct:.0%}")
        if holdings:
            txt += f"\n🏷️ {', '.join(_esc(h) for h in holdings[:8])}"
        self._send(txt)

    def notify_daily_portfolio(self, date, portfolio_value, day_return, cash,
                               cash_pct, positions, total_unrealized, regime):
        """Header asli + tabel posisi code block (fixed width, tidak wrap di HP)."""
        if not self._lvl(1): return
        tag = "UP" if day_return >= 0 else "DOWN"
        regime_emoji = {"BULL": "🟢", "TRANS": "🟡", "BEAR": "🔴"}.get(str(regime), "⚪")
        # Header asli (format yang disukai)
        lines = [
            f"📊 *RINGKASAN HARIAN · {_esc(date)}* [{tag}]",
            f"━━━━━━━━━━━━━━━━━━━━",
            f"💰 Portfolio: *${portfolio_value:,.0f}*",
            f"📈 Hari ini: *{day_return:+.2%}*",
            f"💵 Cash: ${cash:,.0f} ({cash_pct:.0%})",
            f"{regime_emoji} Regime: {_esc(regime)}",
        ]
        if positions:
            pos_sorted = sorted(positions, key=lambda x: x.get("pnl_pct", 0), reverse=True)
            # Tabel code block — fixed width, rapi di HP
            tbl = ["SYM    QTY  ENTRY→NOW       PNL%    $PNL"]
            tbl.append("─" * 42)
            for p in pos_sorted:
                sym     = p.get("sym", "?")[:5]
                qty     = p.get("qty", 0)
                entry   = p.get("entry", 0)
                current = p.get("current", 0)
                pnl_pct = p.get("pnl_pct", 0)
                pnl_usd = p.get("pnl_dollar", 0)
                e_str = f"{entry:,.0f}" if entry >= 100 else f"{entry:.1f}"
                c_str = f"{current:,.0f}" if current >= 100 else f"{current:.1f}"
                sign  = "+" if pnl_pct >= 0 else ""
                sign_d = "+" if pnl_usd >= 0 else ""
                tbl.append(
                    f"{sym:<5}  {qty:>4}  {e_str:>5}→{c_str:<5}  "
                    f"{sign}{pnl_pct*100:.1f}%  {sign_d}${abs(pnl_usd):,.0f}"
                )
            lines.append(f"\n*POSISI ({len(positions)}):*")
            lines.append("```\n" + "\n".join(tbl) + "\n```")
            icon_u = "🟢" if total_unrealized >= 0 else "🔴"
            lines.append(f"{icon_u} *Unrealized: ${total_unrealized:+,.0f}*")
        else:
            lines.append("\n💤 Full cash")
        self._send("\n".join(lines))

    def notify_weekly(self, week, week_return, spy_return, trades_count, portfolio_value):
        if not self._lvl(1): return
        vs = "✅ beat SPY" if week_return > spy_return else "❌ below SPY"
        txt = (f"📊 *RINGKASAN MINGGUAN* · {_esc(week)}\n"
               f"━━━━━━━━━━━━━━━━━━━━\n"
               f"Return: *{week_return:+.2%}* vs SPY {spy_return:+.2%} ({vs})\n"
               f"Trade: {trades_count} · Portfolio: ${portfolio_value:,.0f}")
        self._send(txt)

    def notify_alert(self, message):
        self._send(f"⚠️ *ALERT*\n{_esc(message)}")

    def notify_custom(self, message):
        self._send(message)


if __name__ == "__main__":
    tg = TelegramNotifier(is_live=True, level="all")
    tg.notify_sizing("2026-06-01", "BULL", [
        {"symbol":"MU","conviction":"high","size_mult":1.5,"reasoning":"Strong momentum (roc20=88%) with accelerating fundamentals (earnings growth 756%)"},
        {"symbol":"SNOW","conviction":"low","size_mult":0.7,"reasoning":"Strong momentum but weak fundamentals (no earnings growth) - possible trap"},
    ], "mistral")
    tg.notify_buy("MU", 1026.85, 14, 0.144, "high", "Strong momentum + earnings growth 756%")
    tg.notify_exit("PTON", 32.10, 100, -0.18, "momentum break")
    tg.notify_daily("2026-06-01", 128500, 0.012, 5, 0.05, ["MU","OKTA","FSLR","NUE"])
    print("Test selesai.")
