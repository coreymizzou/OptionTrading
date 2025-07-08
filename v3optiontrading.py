# Elite Options Trade Engine — Multi-Strategy Logic (Full Package)
# Enhanced with RSI-based trend logic, Iron Butterfly filters, and per-strategy diversity control

import requests
import datetime
import numpy as np
import yfinance as yf
from collections import Counter
from tqdm import tqdm  # Add progress bar

QUIVER_API_KEY = "e1c45cb296aab1338edbef3e11fb3b2acd66413b"
TRADIER_API_KEY = "HGW3jJ5ElYbxYrAGW0bZGjNuEeLC"

quiver_headers = {"Authorization": f"Bearer {QUIVER_API_KEY}"}
tradier_headers = {"Authorization": f"Bearer {TRADIER_API_KEY}", "Accept": "application/json"}

ACCOUNT_VALUE = 10000
MAX_TRADE_RISK_PCT = 0.05
MAX_TOTAL_EXPOSURE_PCT = 0.25

strategies = []
total_risked = 0
strategy_counts = Counter()

MAX_PER_STRATEGY = {
    "Iron Butterfly": 3,
    "Iron Condor": 3,
    "Put Credit Spread": 3,
    "Call Credit Spread": 3,
    "Cash-Secured Put": 2,
    "Long Call": 2,
    "Long Put": 2
}

stocks = {
    'Top_Tier': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
    'Mid_Tier': ['SOFI', 'NXT', 'MGNI', 'JOBY', 'SKYE', 'F', 'DKNG', 'U', 'CHPT', 'RUN', 'IONQ', 'PLUG', 'UPST', 'BMBL', 'PINS', 'ROKU', 'W', 'ETSY', 'PTON'],
    'Political': ['BA', 'LMT', 'RTX', 'NOC', 'NANC'],
    'EV': ['TSLA', 'LCID', 'RIVN', 'NIO', 'FISK', 'XPEV'],
    'Energy': ['XOM', 'CVX', 'SLB', 'BP', 'MPC', 'HES'],
    'Tech': ['AMD', 'PLTR', 'INTC', 'CRM', 'ORCL', 'AI', 'PATH', 'SNOW', 'ZS', 'NET', 'DDOG', 'MDB', 'CRWD'],
    'High_Risk': ['GME', 'AMC', 'MARA', 'RIOT', 'HOOD', 'COIN', 'TQQQ', 'SPXL', 'ARKK']
}

def get_underlying_price(ticker):
    url = f"https://api.tradier.com/v1/markets/quotes"
    headers = {
        "Authorization": f"Bearer {TRADIER_API_KEY}",
        "Accept": "application/json"
    }
    params = {"symbols": ticker.upper()}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        quote = data.get("quotes", {}).get("quote")
        if isinstance(quote, dict) and quote.get("last"):
            return float(quote["last"])
    return None

def get_quiver_sentiment(ticker):
    url = f"https://api.quiverquant.com/beta/historical/socialsentiment/{ticker}"
    r = requests.get(url, headers=quiver_headers)
    if r.status_code != 200: return 0
    data = r.json()
    return data[-1].get("SentimentScore", 0) if data else 0

def get_quiver_insiders(ticker):
    url_list = [
        ("Congress", "https://api.quiverquant.com/beta/live/congresstrading"),
        ("Senate", "https://api.quiverquant.com/beta/live/senatetrading"),
        ("Insiders", "https://api.quiverquant.com/beta/live/insiders")
    ]
    score = 0
    weight_map = {"Congress": 0.04, "Senate": 0.04, "Insiders": 0.08}
    details = []

    for label, url in url_list:
        try:
            r = requests.get(url, headers=quiver_headers)
            if r.status_code == 200:
                data = r.json()
                for entry in data:
                    if entry.get('Ticker', '').upper() == ticker.upper():
                        score += weight_map.get(label, 0)
                        details.append(f"{label} trade detected")
        except Exception:
            continue

    return min(score, 0.1), ", ".join(details) if details else "No recent insider trades"

def get_available_expirations(ticker):
    url = f"https://api.tradier.com/v1/markets/options/expirations?symbol={ticker}&includeAllRoots=true&strikes=false"
    r = requests.get(url, headers=tradier_headers)
    if r.status_code == 200:
        return r.json().get("expirations", {}).get("date", [])
    return []

def get_nearest_expiration_in_range(ticker, min_days=25, max_days=60):
    expirations = get_available_expirations(ticker)
    today = datetime.date.today()

    for date_str in expirations:
        try:
            exp_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
            days_out = (exp_date - today).days
            if min_days <= days_out <= max_days and exp_date.weekday() == 4:
                return date_str
        except Exception:
            continue
    return None

def get_tradier_options_chain(ticker, expiration):
    url = f"https://api.tradier.com/v1/markets/options/chains"
    params = {
        "symbol": ticker,
        "expiration": expiration,
        "greeks": "true"
    }
    response = requests.get(url, headers=tradier_headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return data.get("options", {}).get("option", [])
    
    return []

def get_iv_from_tradier_chain(chain):
    raw_ivs = [o.get("greeks", {}).get("mid_iv") for o in chain]
    print(f"[DEBUG] Raw IVs (first 10): {raw_ivs[:10]}")

    ivs = [iv for iv in raw_ivs if isinstance(iv, (float, int)) and iv > 0]

    if not ivs:
        print("[DEBUG] No usable IVs found.")
        return None

    avg_iv = np.mean(ivs) * 100
    print(f"[DEBUG] Average IV: {avg_iv:.2f}%")
    return round(avg_iv, 2)

def select_strikes_around_price(strikes, underlying_price, ticker):
    width = get_dynamic_strike_width(ticker)
    strikes = sorted(strikes)
    center = min(strikes, key=lambda x: abs(x - underlying_price))

    put_short = center
    put_long = max([s for s in strikes if s < center - width / 2], default=None)
    call_short = center
    call_long = min([s for s in strikes if s > center + width / 2], default=None)

    if None in (put_long, call_long):
        return None  # not enough strike range

    return {
        'put_long': put_long,
        'put_short': put_short,
        'call_short': call_short,
        'call_long': call_long
    }

def get_dynamic_strike_width(ticker, fallback=2.5):
    try:
        data = yf.download(ticker, period="14d", interval="1d", progress=False, auto_adjust=False)
        if len(data) < 5:
            return fallback
        data['High-Low'] = data['High'] - data['Low']
        atr = data['High-Low'].rolling(window=5).mean().iloc[-1]
        return round(max(atr * 1.5, fallback), 0.5)
    except Exception:
        return fallback

def get_option_strikes(ticker, expiration):
    url = f"https://api.tradier.com/v1/markets/options/strikes"
    params = {"symbol": ticker, "expiration": expiration}
    r = requests.get(url, headers=tradier_headers, params=params)
    if r.status_code == 200:
        return r.json().get("strikes", {}).get("strike", [])
    return []

def get_technical_signals(ticker):
    try:
        data = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True)
        sma50 = data['Close'].rolling(window=50).mean()
        sma200 = data['Close'].rolling(window=200).mean()
        rsi = 100 - (100 / (1 + data['Close'].pct_change().dropna().rolling(14).mean()))
        if sma50.iloc[-1] > sma200.iloc[-1] and rsi.iloc[-1] < 70:
            return 'Bullish'
        elif sma50.iloc[-1] < sma200.iloc[-1] or rsi.iloc[-1] > 70:
            return 'Bearish'
        return 'Neutral'
    except:
        return 'Neutral'

def choose_strategy(technical, iv_rank):
    if iv_rank is None:
        return None

    if technical == 'Bullish' and iv_rank >= 70:
        strat = "Cash-Secured Put"
    elif technical == 'Bullish' and iv_rank >= 50:
        strat = "Put Credit Spread"
    elif technical == 'Bullish':
        strat = "Long Call"
    elif technical == 'Bearish' and iv_rank >= 50:
        strat = "Call Credit Spread"
    elif technical == 'Bearish':
        strat = "Long Put"
    elif technical == 'Neutral':
        if iv_rank >= 70:
            strat = "Iron Condor"
        elif iv_rank >= 50:
            strat = "Iron Butterfly"
        else:
            return None
    else:
        return None

    if strategy_counts[strat] >= MAX_PER_STRATEGY.get(strat, 3):
        return None
    return strat

def get_spread_legs(options, spread_width=5):
    for i in range(len(options) - 1):
        leg1 = options[i]
        for j in range(i+1, len(options)):
            leg2 = options[j]
            if abs(leg1['strike'] - leg2['strike']) == spread_width:
                return leg1, leg2
    return None, None

def get_nearest_monthly_expiration(ticker):
    url = f"https://api.tradier.com/v1/markets/options/expirations"
    params = {"symbol": ticker, "includeAllRoots": "true", "strikes": "false"}
    response = requests.get(url, headers=tradier_headers, params=params)
    dates = response.json().get("expirations", {}).get("date", [])
    if not dates:
        return None
    # Find the nearest expiration at least 20 days away
    today = datetime.date.today()
    for d in dates:
        exp_date = datetime.datetime.strptime(d, "%Y-%m-%d").date()
        if (exp_date - today).days >= 20:
            return d
    return dates[0] if dates else None

def evaluate_strategies(ticker):
    global total_risked

    expiration = get_nearest_expiration_in_range(ticker)
    if not expiration:
        print(f"No valid expiration found for {ticker}")
        return

    available_strikes = get_option_strikes(ticker, expiration)
    if not available_strikes:
        print(f"No available strikes for {ticker} on {expiration}")
        return

    options_chain = get_tradier_options_chain(ticker, expiration)
    iv_rank = get_iv_from_tradier_chain(options_chain)
    if iv_rank is None:
        print(f"No IV data for {ticker}")
        return

    sentiment = get_quiver_sentiment(ticker)
    insider_score, insider_note = get_quiver_insiders(ticker)
    technical = get_technical_signals(ticker)

    # Calculate IV Skew: avg IV of puts - avg IV of calls
    puts_iv = [o['implied_volatility'] for o in options_chain if o['option_type'] == 'put' and o.get('implied_volatility')]
    calls_iv = [o['implied_volatility'] for o in options_chain if o['option_type'] == 'call' and o.get('implied_volatility')]
    iv_skew = None
    if puts_iv and calls_iv:
        iv_skew = round(sum(puts_iv) / len(puts_iv) - sum(calls_iv) / len(calls_iv), 4)

    # Adjust trend if skew is significant
    if iv_skew is not None:
        if iv_skew > 0.05:
            technical = 'Bearish'
        elif iv_skew < -0.05:
            technical = 'Bullish'

    strategy = choose_strategy(technical, iv_rank)
    if not strategy:
        return

    options = [o for o in options_chain if o.get("volume") and o.get("open_interest") and o.get("bid") and o.get("ask")]
    calls = sorted([o for o in options if o["option_type"] == "call"], key=lambda x: x["strike"])
    puts = sorted([o for o in options if o["option_type"] == "put"], key=lambda x: x["strike"])

    underlying_price = get_underlying_price(ticker)
    if underlying_price is None:
        print(f"Could not retrieve underlying price for {ticker}")
        return

    credit = risk = 0
    trade_details = {
        "Ticker": ticker,
        "Strategy": strategy,
        "Expiration": expiration,
        "IV Rank": iv_rank,
        "Sentiment": sentiment,
    }

    def get_spread_legs(opt_list):
        for i in range(len(opt_list) - 1):
            short = opt_list[i]
            long = opt_list[i + 1]
            if short['bid'] > 0 and long['ask'] > 0:
                return short, long
        return None, None

    if strategy == "Put Credit Spread":
        short, long = get_spread_legs(puts)
        if not short or not long: return
        credit = short['bid'] - long['ask']
        risk = abs(long['strike'] - short['strike']) - credit
        trade_details.update({"Short Put Strike": short['strike'], "Long Put Strike": long['strike']})

    elif strategy == "Call Credit Spread":
        short, long = get_spread_legs(calls)
        if not short or not long: return
        credit = short['bid'] - long['ask']
        risk = abs(long['strike'] - short['strike']) - credit
        trade_details.update({"Short Call Strike": short['strike'], "Long Call Strike": long['strike']})

    if strategy in ["Iron Condor", "Iron Butterfly"]:
        strike_data = select_strikes_around_price(available_strikes, underlying_price, ticker)
        if not strike_data:
            return

        sp = next((p for p in puts if p['strike'] == strike_data['put_short']), None)
        lp = next((p for p in puts if p['strike'] == strike_data['put_long']), None)
        sc = next((c for c in calls if c['strike'] == strike_data['call_short']), None)
        lc = next((c for c in calls if c['strike'] == strike_data['call_long']), None)
        if not all([sp, lp, sc, lc]):
            return

        if strategy == "Iron Condor":
            credit = (sp['bid'] - lp['ask']) + (sc['bid'] - lc['ask'])
            risk = max(abs(lp['strike'] - sp['strike']), abs(lc['strike'] - sc['strike'])) - credit
        else:  # Iron Butterfly
            credit = sp['bid'] + sc['bid'] - lp['ask'] - lc['ask']
            risk = abs(lp['strike'] - sp['strike'])

        trade_details.update({
            "Short Put Strike": sp['strike'],
            "Long Put Strike": lp['strike'],
            "Short Call Strike": sc['strike'],
            "Long Call Strike": lc['strike'],
        })

    elif strategy in ["Long Call", "Long Put", "Cash-Secured Put"]:
        opt_list = calls if strategy == "Long Call" else puts
        selected = next((o for o in opt_list if o['bid'] > 0.1), None)
        if not selected: return
        credit = -selected['ask']
        risk = abs(credit)
        strike_label = "Long Call Strike" if strategy == "Long Call" else ("Long Put Strike" if strategy == "Long Put" else "Short Put Strike")
        trade_details[strike_label] = selected['strike']
    else:
        return

    trade_risk = risk * 100
    if trade_risk > ACCOUNT_VALUE * MAX_TRADE_RISK_PCT or (total_risked + trade_risk) > ACCOUNT_VALUE * MAX_TOTAL_EXPOSURE_PCT:
        return

    base_score = 0.3 + (sentiment/5)*0.1 + (iv_rank/100)*0.15 + insider_score
    confidence = round(base_score * 100, 2)

    trade_details.update({
        "Credit or Cost": round(credit, 2),
        "Confidence Level": confidence,
        "Trade Risk ($)": trade_risk,
        "Account % Used": round((trade_risk / ACCOUNT_VALUE) * 100, 2),
        "Explanation": f"{strategy} selected based on trend={technical}, IV Rank={iv_rank}%. Credit={round(credit,2)}, Risk={round(risk,2)}, Insider Note={insider_note}."
    })

    strategies.append(trade_details)
    total_risked += trade_risk
    strategy_counts[strategy] += 1

def run_engine():
    all_tickers = sum(stocks.values(), [])
    for ticker in tqdm(all_tickers, desc="Evaluating tickers"):
        try:
            evaluate_strategies(ticker)
        except Exception as e:
            print(f"Error evaluating {ticker}: {e}")

    ranked = sorted(strategies, key=lambda x: -x['Confidence Level'])
    if not ranked:
        print("\nNo trades met the criteria today. Market conditions may be unfavorable or thresholds too strict.")
        return

    for i, trade in enumerate(ranked):
        print(f"\nTrade {i + 1}:")
        print(f"Ticker: {trade['Ticker']}")
        print(f"Strategy: {trade['Strategy']}")
        print(f"Expiration: {trade['Expiration']}")
        print(f"IV Rank: {trade['IV Rank']}")
        print(f"Sentiment: {trade['Sentiment']}")

        print("\nStrikes:")
        if 'Short Put Strike' in trade and 'Long Put Strike' in trade:
            print(f"  Put Spread: {trade['Short Put Strike']} / {trade['Long Put Strike']}")
        elif 'Short Put Strike' in trade:
            print(f"  Put Strike: {trade['Short Put Strike']}")
        elif 'Long Put Strike' in trade:
            print(f"  Put Strike: {trade['Long Put Strike']}")

        if 'Short Call Strike' in trade and 'Long Call Strike' in trade:
            print(f"  Call Spread: {trade['Short Call Strike']} / {trade['Long Call Strike']}")
        elif 'Short Call Strike' in trade:
            print(f"  Call Strike: {trade['Short Call Strike']}")
        elif 'Long Call Strike' in trade:
            print(f"  Call Strike: {trade['Long Call Strike']}")

        print(f"\nCredit or Cost: {trade['Credit or Cost']}")
        print(f"Confidence Level: {trade['Confidence Level']}")
        print(f"Trade Risk ($): {trade['Trade Risk ($)']}")
        print(f"Account % Used: {trade['Account % Used']}")
        print(f"Explanation: {trade['Explanation']}")

if __name__ == "__main__":
    run_engine()
