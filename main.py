# ============================================================
# MEXC QUANT ENGINE — BACK4APP CONTAINERS
# Telegram Bot | EMA Ribbon | S/R | Volume Spike | Backtest
# ============================================================

import os
import time
import math
import logging
import threading
import traceback
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd
import mplfinance as mpf
import telebot
from telebot import types

# ===================== CONFIGURATION =========================
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()

EXCHANGE_ID = "mexc"
TIMEFRAME = "1h"
SCAN_LIMIT = 300
CHUNK_SIZE = 20
REQUEST_DELAY = 0.4
MAX_RETRIES = 5
BACKOFF_BASE = 15
OHLCV_LIMIT = 240
BACKTEST_DAYS = 30
MIN_VOLUME_MULTIPLIER = 1.5
MIN_RR = 1.0
AUTO_SCAN_INTERVAL = 900
MAX_SIGNAL_MESSAGES = 100
ORDERBOOK_LIMIT = 20
# ============================================================

if not BOT_TOKEN:
    raise ValueError("Atur BOT_TOKEN di Environment Variables Back4app.")
if not CHAT_ID:
    raise ValueError("Atur CHAT_ID di Environment Variables Back4app.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("MEXC_QUANT")

bot = telebot.TeleBot(BOT_TOKEN, threaded=True, num_threads=8)

exchange = ccxt.mexc({
    "enableRateLimit": True,
    "timeout": 30000,
    "options": {"defaultType": "spot"},
})

state_lock = threading.RLock()
scan_lock = threading.Lock()
state = {
    "timeframe": TIMEFRAME,
    "auto": False,
    "scan_running": False,
    "last_scan": "Belum pernah",
    "last_symbols": [],
    "signals": {},
    "scan_count": 0,
}
STOP_EVENT = threading.Event()


# ======================== UTILITIES ==========================
def safe_send(chat_id, text, **kwargs):
    try:
        return bot.send_message(
            chat_id, text, disable_web_page_preview=True, **kwargs
        )
    except Exception as exc:
        log.warning("Gagal mengirim pesan Telegram: %s", exc)
        return None


def safe_edit(chat_id, message_id, text, **kwargs):
    try:
        return bot.edit_message_text(text, chat_id, message_id, **kwargs)
    except Exception:
        return None


def is_authorized(message_or_call):
    chat_id = (
        message_or_call.message.chat.id
        if hasattr(message_or_call, "message")
        else message_or_call.chat.id
    )
    return str(chat_id) == str(CHAT_ID)


def fmt_price(value):
    if value is None or not np.isfinite(value):
        return "-"
    value = float(value)
    if value >= 1000:
        return f"{value:,.2f}"
    if value >= 1:
        return f"{value:.5f}"
    return f"{value:.8f}".rstrip("0").rstrip(".")


def chunks(items, size=CHUNK_SIZE):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def exchange_call(func, *args, **kwargs):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except (
            ccxt.RateLimitExceeded,
            ccxt.DDoSProtection,
            ccxt.RequestTimeout,
            ccxt.NetworkError,
            ccxt.ExchangeNotAvailable,
        ) as exc:
            if attempt >= MAX_RETRIES:
                log.error("Request gagal setelah retry: %s", exc)
                return None
            delay = min(BACKOFF_BASE * (2 ** attempt), 180)
            log.warning(
                "Gangguan jaringan/rate limit. Retry %s/%s dalam %ss",
                attempt + 1, MAX_RETRIES, delay
            )
            time.sleep(delay)
        except ccxt.ExchangeError as exc:
            log.warning("Exchange error: %s", exc)
            return None
        except Exception:
            log.exception("Error tidak terduga")
            return None
    return None


# ====================== MARKET UNIVERSE ======================
def load_markets():
    markets = exchange_call(exchange.load_markets)
    return markets or {}


def get_top_symbols(limit=SCAN_LIMIT):
    """Ranking spot USDT berdasarkan quoteVolume 24 jam, bukan market cap."""
    markets = load_markets()
    if not markets:
        return []

    tickers = exchange_call(exchange.fetch_tickers)
    if not tickers:
        return []

    candidates = []
    for symbol, market in markets.items():
        if not market.get("spot") or not market.get("active", True):
            continue
        if market.get("quote") != "USDT" or market.get("base") == "USDT":
            continue

        ticker = tickers.get(symbol, {})
        quote_volume = ticker.get("quoteVolume")
        if quote_volume is None:
            base_volume, last = ticker.get("baseVolume"), ticker.get("last")
            if base_volume is not None and last is not None:
                quote_volume = float(base_volume) * float(last)
        if quote_volume is None:
            continue
        try:
            quote_volume = float(quote_volume)
        except (ValueError, TypeError):
            continue
        if np.isfinite(quote_volume) and quote_volume > 0:
            candidates.append((symbol, quote_volume))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [symbol for symbol, _ in candidates[:limit]]


def fetch_ohlcv_df(symbol, timeframe=None, limit=OHLCV_LIMIT):
    timeframe = timeframe or state["timeframe"]
    time.sleep(REQUEST_DELAY)
    rows = exchange_call(exchange.fetch_ohlcv, symbol, timeframe, None, limit)
    if not rows or len(rows) < 205:
        return None

    df = pd.DataFrame(
        rows, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    )
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("Timestamp").set_index("Timestamp").sort_index()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df if len(df) >= 205 else None


# ======================== INDICATORS =========================
def add_indicators(df):
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["VolMA20"] = df["Volume"].rolling(20).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    return df


# ================= SUPPORT / RESISTANCE =====================
def find_support_resistance(df, window=3, lookback=100):
    data = df.tail(lookback).copy()
    atr = float(data["ATR"].iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return None, None, 0, 0

    highs = data["High"].to_numpy()
    lows = data["Low"].to_numpy()
    close = float(data["Close"].iloc[-1])
    swing_highs, swing_lows = [], []

    for i in range(window, len(data) - window):
        if highs[i] >= max(highs[i-window:i+window+1]):
            swing_highs.append(float(highs[i]))
        if lows[i] <= min(lows[i-window:i+window+1]):
            swing_lows.append(float(lows[i]))

    def cluster_levels(levels):
        groups = []
        for level in sorted(levels):
            if not groups or abs(level - np.mean(groups[-1])) > atr * 0.5:
                groups.append([level])
            else:
                groups[-1].append(level)
        return [float(np.mean(group)) for group in groups]

    resistances = [x for x in cluster_levels(swing_highs) if x > close]
    supports = [x for x in cluster_levels(swing_lows) if x < close]
    resistance = min(resistances) if resistances else None
    support = max(supports) if supports else None

    def touch_count(level):
        if level is None:
            return 0
        return int(
            (data["High"].sub(level).abs() <= atr * 0.5).sum()
            + (data["Low"].sub(level).abs() <= atr * 0.5).sum()
        )

    return support, resistance, touch_count(support), touch_count(resistance)


# ====================== SIGNAL ENGINE ========================
def analyze_symbol(symbol, timeframe=None):
    df = fetch_ohlcv_df(symbol, timeframe)
    if df is None:
        return None
    df = add_indicators(df)
    row = df.iloc[-1]
    close, atr, vol_ma = float(row["Close"]), float(row["ATR"]), float(row["VolMA20"])
    if not np.isfinite(atr) or atr <= 0 or not np.isfinite(vol_ma) or vol_ma <= 0:
        return None

    volume_ratio = float(row["Volume"]) / vol_ma
    support, resistance, support_touches, resistance_touches = find_support_resistance(df)
    recent = df.tail(30)
    swing_low = float(recent["Low"].iloc[:-1].min())
    swing_high = float(recent["High"].iloc[:-1].max())

    ema_bull = row["EMA20"] > row["EMA50"] > row["EMA200"]
    ema_bear = row["EMA20"] < row["EMA50"] < row["EMA200"]
    momentum_bull = close > row["EMA20"] and 52 <= row["RSI"] <= 72
    momentum_bear = close < row["EMA20"] and 28 <= row["RSI"] <= 48
    volume_ok = volume_ratio >= MIN_VOLUME_MULTIPLIER

    side = "BUY" if ema_bull and momentum_bull and volume_ok else (
        "SELL" if ema_bear and momentum_bear and volume_ok else None
    )
    if side is None:
        return None

    entry_low, entry_high, entry = close - 0.15 * atr, close + 0.15 * atr, close
    if side == "BUY":
        sl = min(swing_low, entry - 1.2 * atr)
        risk = entry - sl
        if risk <= 0:
            return None
        tp1, tp2, tp3 = entry + risk, entry + 2*risk, entry + 3*risk
        sr_touches, sr_level = support_touches, support
    else:
        sl = max(swing_high, entry + 1.2 * atr)
        risk = sl - entry
        if risk <= 0:
            return None
        tp1, tp2, tp3 = entry - risk, entry - 2*risk, entry - 3*risk
        sr_touches, sr_level = resistance_touches, resistance

    degradation = "TINGGI" if sr_touches >= 3 else "SEDANG" if sr_touches == 2 else "RENDAH"
    return {
        "symbol": symbol, "side": side, "timeframe": timeframe or state["timeframe"],
        "price": close, "entry_low": entry_low, "entry_high": entry_high,
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "risk": risk, "atr": atr, "rsi": float(row["RSI"]),
        "volume_ratio": volume_ratio, "support": support, "resistance": resistance,
        "sr_level": sr_level, "sr_touches": sr_touches, "degradation": degradation,
        "ema20": float(row["EMA20"]), "ema50": float(row["EMA50"]),
        "ema200": float(row["EMA200"]), "df": df,
    }


# ======================= CHART ENGINE ========================
def create_chart(signal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = signal["df"].tail(100).copy()
    ap = [
        mpf.make_addplot(df["EMA20"], color="dodgerblue", width=1),
        mpf.make_addplot(df["EMA50"], color="orange", width=1),
        mpf.make_addplot(df["EMA200"], color="purple", width=1),
    ]
    entry_mid = (signal["entry_low"] + signal["entry_high"]) / 2
    hlines = dict(
        hlines=[signal["sl"], entry_mid, signal["tp1"], signal["tp2"], signal["tp3"]],
        colors=["red", "blue", "green", "green", "green"],
        linestyle=["--", "-", "--", "--", "--"],
        linewidths=[1.2, 1.4, 1, 1, 1],
    )
    fig, axes = mpf.plot(
        df, type="candle", style="yahoo", volume=True, addplot=ap,
        hlines=hlines, title=f"{signal['symbol']} | {signal['side']} | {signal['timeframe']}",
        ylabel="Price", ylabel_lower="Volume", figsize=(14, 8),
        returnfig=True, tight_layout=True,
    )
    axes[0].axhspan(signal["entry_low"], signal["entry_high"], alpha=0.16, color="blue", label="Entry Zone")
    axes[0].legend(loc="upper left", fontsize=8)
    path = "/tmp/mexc_quant_chart.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ======================= SIGNAL MESSAGE =====================
def signal_message(s):
    emoji = "🟢" if s["side"] == "BUY" else "🔴"
    return (
        f"{emoji} *QUANT SIGNAL — {s['side']}*\n━━━━━━━━━━━━━━━━━━\n"
        f"🪙 Asset: `{s['symbol']}`\n⏱ Timeframe: `{s['timeframe']}`\n"
        f"💵 Harga: `{fmt_price(s['price'])}`\n"
        f"🎯 Entry Zone: `{fmt_price(s['entry_low'])}` – `{fmt_price(s['entry_high'])}`\n"
        f"🛑 Stop Loss: `{fmt_price(s['sl'])}`\n"
        f"🥇 TP1 (1R): `{fmt_price(s['tp1'])}`\n🥈 TP2 (2R): `{fmt_price(s['tp2'])}`\n"
        f"🥉 TP3 (3R): `{fmt_price(s['tp3'])}`\n━━━━━━━━━━━━━━━━━━\n"
        f"📐 RSI: `{s['rsi']:.2f}`\n📊 Volume/MA20: `{s['volume_ratio']:.2f}x`\n"
        f"📈 EMA20: `{fmt_price(s['ema20'])}`\n📈 EMA50: `{fmt_price(s['ema50'])}`\n"
        f"📈 EMA200: `{fmt_price(s['ema200'])}`\n🧱 S/R touches: `{s['sr_touches']}`\n"
        f"⚠️ S/R degradation: `{s['degradation']}`\n━━━━━━━━━━━━━━━━━━\n"
        "_Sinyal berbasis aturan teknikal; bukan jaminan profit._"
    )


def send_signal(chat_id, signal):
    try:
        path = create_chart(signal)
        with open(path, "rb") as image_file:
            bot.send_photo(chat_id, image_file, caption=signal_message(signal), parse_mode="Markdown")
    except Exception as exc:
        log.exception("Chart/sinyal gagal: %s", exc)
        safe_send(chat_id, signal_message(signal), parse_mode="Markdown")


# ===================== SCAN STREAMING ========================
def run_scan(chat_id, only_signals=False):
    if not scan_lock.acquire(blocking=False):
        safe_send(chat_id, "⏳ Scan sedang berjalan. Tunggu sampai selesai.")
        return
    with state_lock:
        state["scan_running"] = True

    found, scanned, errors = [], 0, 0
    safe_send(chat_id, "🔄 [QUANT ENGINE ACTIVE]: Memindai data aset MEXC...")
    try:
        symbols = get_top_symbols(SCAN_LIMIT)
        if not symbols:
            safe_send(chat_id, "❌ Daftar market gagal diambil. Coba lagi nanti.")
            return
        safe_send(chat_id, f"📡 Universe: {len(symbols)} pasar spot USDT berdasarkan volume 24 jam.\n🧩 Chunk: {CHUNK_SIZE} aset.")
        with state_lock:
            state["last_symbols"] = symbols

        for batch_number, batch in enumerate(chunks(symbols, CHUNK_SIZE), start=1):
            safe_send(chat_id, f"🔎 Memproses chunk {batch_number} ({len(batch)} aset)...")
            for symbol in batch:
                try:
                    signal = analyze_symbol(symbol, state["timeframe"])
                    scanned += 1
                    if signal:
                        found.append(signal)
                        with state_lock:
                            state["signals"][symbol] = signal
                        if len(found) <= MAX_SIGNAL_MESSAGES:
                            send_signal(chat_id, signal)
                except Exception as exc:
                    errors += 1
                    log.warning("%s: %s", symbol, exc)
            time.sleep(1.0)

        with state_lock:
            state["last_scan"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            state["scan_count"] += 1

        if not found:
            safe_send(chat_id, "NO TRADE: Tidak ditemukan setup yang memenuhi kriteria EMA Ribbon, momentum, dan volume.")
        elif len(found) > MAX_SIGNAL_MESSAGES:
            safe_send(chat_id, f"⚠️ Total sinyal valid: {len(found)}. {MAX_SIGNAL_MESSAGES} sinyal pertama telah dikirim.")

        buy_count = sum(s["side"] == "BUY" for s in found)
        sell_count = sum(s["side"] == "SELL" for s in found)
        safe_send(
            chat_id,
            "📊 *KESIMPULAN METRIK PASAR GLOBAL MEXC*\n━━━━━━━━━━━━━━━━━━\n"
            f"• Aset diproses: `{scanned}`\n• Sinyal valid: `{len(found)}`\n"
            f"• BUY: `{buy_count}`\n• SELL: `{sell_count}`\n• Error: `{errors}`\n"
            f"• Timeframe: `{state['timeframe']}`\n• Waktu: `{state['last_scan']}`\n"
            "━━━━━━━━━━━━━━━━━━\n_Ranking universe berdasarkan volume, bukan market cap. Hasil scan bukan rekomendasi finansial._",
            parse_mode="Markdown"
        )
    except Exception:
        log.exception("Scan gagal")
        safe_send(chat_id, "❌ Scan berhenti karena error internal.")
    finally:
        with state_lock:
            state["scan_running"] = False
        scan_lock.release()


# ======================= ORDER BOOK ==========================
def orderbook_imbalance(symbol, limit=ORDERBOOK_LIMIT):
    ob = exchange_call(exchange.fetch_order_book, symbol, limit)
    if not ob:
        return None
    bids, asks = ob.get("bids", []), ob.get("asks", [])
    bid_depth = sum(float(p) * float(a) for p, a in bids)
    ask_depth = sum(float(p) * float(a) for p, a in asks)
    total = bid_depth + ask_depth
    if total <= 0:
        return None
    bid_share, ask_share = bid_depth / total, ask_depth / total
    return {
        "symbol": symbol, "bid_depth": bid_depth, "ask_depth": ask_depth,
        "bid_share": bid_share, "ask_share": ask_share,
        "imbalance": bid_share - ask_share,
    }


def scan_orderbooks(chat_id):
    with state_lock:
        candidates = list(state["signals"].keys())[:30]
    if not candidates:
        safe_send(chat_id, "Belum ada sinyal radar. Jalankan Analisis Semua terlebih dahulu.")
        return
    rows = []
    for symbol in candidates:
        result = orderbook_imbalance(symbol)
        if result:
            rows.append(result)
        time.sleep(REQUEST_DELAY)
    rows.sort(key=lambda r: abs(r["imbalance"]), reverse=True)
    lines = ["📊 *ORDERBOOK IMBALANCE*", f"Depth nominal pada {ORDERBOOK_LIMIT} level teratas per sisi.", ""]
    for r in rows[:20]:
        pct = r["imbalance"] * 100
        bias = "BUY WALL" if pct >= 20 else "SELL WALL" if pct <= -20 else "NETRAL"
        confirmation = "⚡ IMBALANCE >70%" if abs(pct) > 70 else ""
        lines.append(
            f"`{r['symbol']}`\nBid: `{r['bid_share']*100:.1f}%` | Ask: `{r['ask_share']*100:.1f}%`\n"
            f"Imbalance: `{pct:+.1f}%` | {bias} {confirmation}\n"
        )
    lines.append("_Order book dapat berubah cepat dan tidak membuktikan identitas atau niat whale._")
    safe_send(chat_id, "\n".join(lines), parse_mode="Markdown")


# ========================== BACKTEST =========================
def backtest_symbol(symbol, days=BACKTEST_DAYS):
    timeframe = "1h"
    limit = min(days * 24 + 220, 1000)
    df = fetch_ohlcv_df(symbol, timeframe, limit)
    if df is None:
        return None
    df = add_indicators(df)
    trades = []

    for i in range(205, len(df) - 1):
        hist, row = df.iloc[:i + 1], df.iloc[i]
        atr, vol_ma = float(row["ATR"]), float(row["VolMA20"])
        if not np.isfinite(atr) or atr <= 0 or not np.isfinite(vol_ma) or vol_ma <= 0:
            continue
        ratio = float(row["Volume"]) / vol_ma
        bull = (row["EMA20"] > row["EMA50"] > row["EMA200"] and row["Close"] > row["EMA20"] and 52 <= row["RSI"] <= 72 and ratio >= MIN_VOLUME_MULTIPLIER)
        bear = (row["EMA20"] < row["EMA50"] < row["EMA200"] and row["Close"] < row["EMA20"] and 28 <= row["RSI"] <= 48 and ratio >= MIN_VOLUME_MULTIPLIER)
        if not bull and not bear:
            continue
        side, entry = ("BUY" if bull else "SELL"), float(row["Close"])
        recent = hist.tail(30)
        if side == "BUY":
            sl = min(float(recent["Low"].iloc[:-1].min()), entry - 1.2 * atr)
            risk, tp = entry - sl, entry + (entry - sl)
        else:
            sl = max(float(recent["High"].iloc[:-1].max()), entry + 1.2 * atr)
            risk, tp = sl - entry, entry - (sl - entry)
        if risk <= 0:
            continue

        outcome = None
        future = df.iloc[i + 1:min(i + 25, len(df))]
        for _, candle in future.iterrows():
            high, low = float(candle["High"]), float(candle["Low"])
            sl_hit, tp_hit = (
                (low <= sl, high >= tp) if side == "BUY"
                else (high >= sl, low <= tp)
            )
            if sl_hit and tp_hit:
                outcome = -1.0
                break
            if sl_hit:
                outcome = -1.0
                break
            if tp_hit:
                outcome = 1.0
                break
        if outcome is None:
            if future.empty:
                continue
            last_close = float(future["Close"].iloc[-1])
            pnl_r = (last_close - entry) / risk if side == "BUY" else (entry - last_close) / risk
            outcome = float(np.clip(pnl_r, -1.0, 1.0))
        trades.append(outcome)

    if not trades:
        return {"symbol": symbol, "trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "max_dd": 0.0, "net_r": 0.0}
    arr = np.asarray(trades, dtype=float)
    equity = np.cumsum(arr)
    running_peak = np.maximum.accumulate(np.concatenate(([0.0], equity)))[1:]
    drawdown = running_peak - equity
    max_dd_r = float(np.max(drawdown)) if len(drawdown) else 0.0
    gross_profit = float(arr[arr > 0].sum())
    gross_loss = float(abs(arr[arr < 0].sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    return {
        "symbol": symbol, "trades": len(arr), "win_rate": float((arr > 0).mean() * 100),
        "profit_factor": pf, "max_dd": max_dd_r, "net_r": float(arr.sum()),
    }


def run_backtest(chat_id):
    safe_send(chat_id, "🧪 Backtest berjalan: lima pasar utama MEXC, timeframe 1h, histori hingga 30 hari...")
    symbols = get_top_symbols(10)
    preferred = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT"]
    selected, available = [], set(symbols)
    for symbol in preferred:
        if symbol in available:
            selected.append(symbol)
    for symbol in symbols:
        if symbol not in selected and len(selected) < 5:
            selected.append(symbol)
    if not selected:
        safe_send(chat_id, "❌ Market untuk backtest tidak tersedia.")
        return
    results = []
    for symbol in selected:
        try:
            result = backtest_symbol(symbol)
            if result:
                results.append(result)
        except Exception as exc:
            log.warning("Backtest %s gagal: %s", symbol, exc)
        time.sleep(REQUEST_DELAY)
    if not results:
        safe_send(chat_id, "❌ Tidak ada hasil backtest yang valid.")
        return
    lines = [
        "*BACKTEST METRICS — HISTORI 30 HARI*", "",
        "| Pair | Trades | WR | PF | Max DD (R) | Net (R) |",
        "|---|---:|---:|---:|---:|---:|"
    ]
    for r in results:
        pf_text = "∞" if math.isinf(r["profit_factor"]) else f"{r['profit_factor']:.2f}"
        lines.append(
            f"| {r['symbol']} | {r['trades']} | {r['win_rate']:.1f}% | {pf_text} | "
            f"{r['max_dd']:.2f} | {r['net_r']:+.2f} |"
        )
    lines += [
        "", "_R = unit risiko per trade. Max DD dihitung dalam unit R, bukan persentase modal._",
        "_Simulasi sederhana; belum memasukkan biaya, slippage, funding, dan keterbatasan fill. Hasil historis bukan prediksi masa depan._"
    ]
    safe_send(chat_id, "\n".join(lines), parse_mode="Markdown")


# ======================== WHALE TRACKER ======================
def whale_tracker(chat_id):
    symbols = get_top_symbols(20)
    if not symbols:
        safe_send(chat_id, "Data market tidak tersedia.")
        return
    rows = []
    for symbol in symbols[:15]:
        df = fetch_ohlcv_df(symbol, state["timeframe"], 80)
        if df is None:
            continue
        df = add_indicators(df)
        row, vol_ma = df.iloc[-1], df["VolMA20"].iloc[-1]
        if pd.isna(vol_ma) or vol_ma <= 0:
            continue
        ratio = float(row["Volume"] / vol_ma)
        if ratio >= MIN_VOLUME_MULTIPLIER:
            rows.append((symbol, ratio, float(row["Close"])))
    rows.sort(key=lambda x: x[1], reverse=True)
    if not rows:
        safe_send(chat_id, "🐋 Belum ada lonjakan volume signifikan.")
        return
    lines = ["🐋 *VOLUME RADAR*", ""]
    for symbol, ratio, price in rows[:10]:
        lines.append(f"`{symbol}` | Volume `{ratio:.2f}x MA20` | Price `{fmt_price(price)}`")
    lines.append("\n_Ini radar volume, bukan bukti transaksi whale._")
    safe_send(chat_id, "\n".join(lines), parse_mode="Markdown")


# ========================= HEATMAP ===========================
def market_heatmap(chat_id):
    symbols = get_top_symbols(30)
    if not symbols:
        safe_send(chat_id, "Data market tidak tersedia.")
        return
    tickers = exchange_call(exchange.fetch_tickers)
    if not tickers:
        safe_send(chat_id, "Gagal mengambil ticker.")
        return
    rows = []
    for symbol in symbols:
        ticker = tickers.get(symbol, {})
        change, volume = ticker.get("percentage"), ticker.get("quoteVolume")
        if change is not None:
            try:
                rows.append((symbol, float(change), float(volume or 0)))
            except (ValueError, TypeError):
                continue
    rows.sort(key=lambda x: x[1], reverse=True)
    lines = ["🔥 *MARKET HEATMAP & VOLUME*"]
    for symbol, change, volume in rows[:20]:
        icon = "🟢" if change >= 0 else "🔴"
        lines.append(f"{icon} `{symbol}` | `{change:+.2f}%` | Vol: `{volume:,.0f} USDT`")
    safe_send(chat_id, "\n".join(lines), parse_mode="Markdown")


# ========================== MENUS ============================
def main_menu():
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("📊 Analisis Semua", callback_data="scan"),
        types.InlineKeyboardButton("🎯 Sinyal BUY/SELL", callback_data="signals"),
    )
    kb.add(
        types.InlineKeyboardButton("🕒 Pilih Timeframe", callback_data="timeframes"),
        types.InlineKeyboardButton("🔔 Auto ON/OFF", callback_data="auto"),
    )
    kb.add(
        types.InlineKeyboardButton("🐋 Whales Tracker", callback_data="whales"),
        types.InlineKeyboardButton("🔥 Heatmap & Vol", callback_data="heatmap"),
    )
    kb.add(
        types.InlineKeyboardButton("🧪 Backtest Metrics", callback_data="backtest"),
        types.InlineKeyboardButton("📊 Orderbook Imbalance", callback_data="orderbook"),
    )
    kb.add(
        types.InlineKeyboardButton("📈 Status Bot", callback_data="status"),
        types.InlineKeyboardButton("ℹ️ Bantuan", callback_data="help"),
    )
    return kb


def timeframe_menu():
    kb = types.InlineKeyboardMarkup(row_width=3)
    kb.add(*[
        types.InlineKeyboardButton(tf, callback_data=f"tf:{tf}")
        for tf in ["5m", "15m", "30m", "1h", "4h", "1d"]
    ])
    kb.add(types.InlineKeyboardButton("⬅️ Menu Utama", callback_data="menu"))
    return kb


# ======================= AUTO SCANNER ========================
def auto_scan_loop():
    while not STOP_EVENT.is_set():
        with state_lock:
            enabled = state["auto"]
        if enabled:
            run_scan(CHAT_ID)
        for _ in range(AUTO_SCAN_INTERVAL):
            if STOP_EVENT.is_set():
                break
            time.sleep(1)


# ======================== TELEGRAM ===========================
def show_status(chat_id):
    with state_lock:
        auto = state["auto"]
        last_scan = state["last_scan"]
        tf = state["timeframe"]
        count = state["scan_count"]
        signals = len(state["signals"])
        running = state["scan_running"]
    safe_send(
        chat_id,
        "📈 *STATUS QUANT ENGINE*\n"
        f"• Exchange: `MEXC`\n• Timeframe: `{tf}`\n"
        f"• Auto scan: `{'ON' if auto else 'OFF'}`\n"
        f"• Scan aktif: `{'YA' if running else 'TIDAK'}`\n"
        f"• Total sesi selesai: `{count}`\n• Radar sinyal tersimpan: `{signals}`\n"
        f"• Scan terakhir: `{last_scan}`",
        parse_mode="Markdown", reply_markup=main_menu()
    )


@bot.message_handler(commands=["start", "menu"])
def start_handler(message):
    if not is_authorized(message):
        return
    bot.send_message(
        message.chat.id,
        "⚡ *MEXC QUANT ENGINE*\nPilih fitur melalui menu di bawah.\n\n"
        "Universe: pasar spot USDT berdasarkan volume 24 jam.\n"
        "Mode: analisis dan simulasi, tanpa eksekusi order.",
        reply_markup=main_menu(), parse_mode="Markdown"
    )


@bot.message_handler(commands=["status"])
def status_handler(message):
    if is_authorized(message):
        show_status(message.chat.id)


@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    if not is_authorized(call):
        bot.answer_callback_query(call.id, "Akses ditolak.")
        return
    data = call.data
    bot.answer_callback_query(call.id)
    chat_id = call.message.chat.id

    if data == "menu":
        safe_send(chat_id, "⚡ Menu utama:", reply_markup=main_menu())
    elif data == "scan":
        threading.Thread(target=run_scan, args=(chat_id,), daemon=True).start()
    elif data == "signals":
        with state_lock:
            signals = list(state["signals"].values())
        if not signals:
            safe_send(chat_id, "Belum ada sinyal tersimpan. Jalankan Analisis Semua.")
        else:
            for signal in signals[:20]:
                safe_send(chat_id, signal_message(signal), parse_mode="Markdown")
    elif data == "timeframes":
        safe_send(chat_id, "Pilih timeframe:", reply_markup=timeframe_menu())
    elif data.startswith("tf:"):
        tf = data.split(":", 1)[1]
        if tf in exchange.timeframes:
            with state_lock:
                state["timeframe"] = tf
            safe_send(chat_id, f"✅ Timeframe diubah ke {tf}.")
        else:
            safe_send(chat_id, "Timeframe tidak didukung exchange.")
    elif data == "auto":
        with state_lock:
            state["auto"] = not state["auto"]
            enabled = state["auto"]
        safe_send(chat_id, f"🔔 Auto scan: {'ON' if enabled else 'OFF'}\nInterval: {AUTO_SCAN_INTERVAL // 60} menit.")
    elif data == "whales":
        threading.Thread(target=whale_tracker, args=(chat_id,), daemon=True).start()
    elif data == "heatmap":
        threading.Thread(target=market_heatmap, args=(chat_id,), daemon=True).start()
    elif data == "backtest":
        threading.Thread(target=run_backtest, args=(chat_id,), daemon=True).start()
    elif data == "orderbook":
        threading.Thread(target=scan_orderbooks, args=(chat_id,), daemon=True).start()
    elif data == "status":
        show_status(chat_id)
    elif data == "help":
        safe_send(
            chat_id,
            "ℹ️ *BANTUAN*\n\n"
            "• Analisis Semua: scan pasar spot USDT.\n• Sinyal BUY/SELL: lihat sinyal tersimpan.\n"
            "• Pilih Timeframe: ubah timeframe scan.\n• Auto ON/OFF: scan berkala.\n"
            "• Whales Tracker: radar lonjakan volume.\n• Heatmap & Vol: perubahan harga dan volume.\n"
            "• Backtest Metrics: simulasi histori.\n• Orderbook Imbalance: perbandingan depth bid/ask.\n\n"
            "SL/TP adalah level analisis, bukan order bursa.\n"
            "Spot SELL bukan otomatis posisi short.\nTidak ada jaminan win rate atau profit.",
            parse_mode="Markdown", reply_markup=main_menu()
        )


# ======================== STARTUP ============================
def main():
    log.info("Memeriksa koneksi MEXC...")
    markets = load_markets()
    if not markets:
        log.warning("Market belum berhasil dimuat; bot tetap berjalan.")
    else:
        log.info("Markets loaded: %s", len(markets))

    threading.Thread(target=auto_scan_loop, daemon=True).start()
    safe_send(
        CHAT_ID,
        "✅ *MEXC QUANT ENGINE ONLINE*\nKoneksi awal selesai.\nKetik /start untuk membuka menu.",
        parse_mode="Markdown", reply_markup=main_menu()
    )
    log.info("Bot Telegram berjalan.")
    bot.infinity_polling(skip_pending=True, timeout=30, long_polling_timeout=30)


if __name__ == "__main__":
    main()
