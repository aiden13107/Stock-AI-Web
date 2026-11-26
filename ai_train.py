import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from pykrx import stock
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# 1. 환경 변수(.env) 파일 로드 (KRX API 키 등 보호)
load_dotenv()

warnings.filterwarnings("ignore")

# =========================================================
# 설정 값
# =========================================================
KRX_API_KEY = os.getenv("KRX_API_KEY", "YOUR_KRX_API_KEY_HERE")
KRX_API_URL = "https://data-dbg.krx.co.kr/svc/apis/sto/stk_bydd_trd"

START_DATE = "20160101"
END_DATE = datetime.now().strftime("%Y%m%d")
MARKETS = ["KOSPI", "KOSDAQ"]

INITIAL_CASH = 10_000_000
TRADING_COST = 0.0015
BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45
MIN_TRAIN_YEARS = 5
N_ESTIMATORS = 100
MAX_STOCKS = 30  # 전체 종목 분석 시 None, 테스트 시 숫자로 지정 (예: 30)
HOLDINGS = set()
REQUEST_DELAY = 0.5

BASE_DIR = Path("multi_stock_results")
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "results"
BACKTEST_CACHE_DIR = OUTPUT_DIR / "backtest_cache"
PROGRESS_FILE = OUTPUT_DIR / "progress.csv"

for d in [CACHE_DIR, OUTPUT_DIR, BACKTEST_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =========================================================
# 피처 및 모델 정의
# =========================================================
FEATURE_COLS = [
    "Return_1", "Return_5", "Return_20", "Return_60",
    "MA5_Ratio", "MA20_Ratio", "MA60_Ratio", "MA120_Ratio",
    "MA_Cross", "Long_Trend",
    "RSI", "MACD", "MACD_Diff",
    "Volatility_20", "Volatility_60",
    "Volume_Change", "Volume_Ratio",
    "Daily_Range", "Open_Close_Ratio"
]

def create_model():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

# =========================================================
# 데이터 수집 함수
# =========================================================
def fetch_krx_daily_snapshot(bas_dd):
    if not KRX_API_KEY or KRX_API_KEY == "YOUR_KRX_API_KEY_HERE":
        return pd.DataFrame()

    headers = {"AUTH_KEY": KRX_API_KEY, "Accept": "application/json"}
    params = {"basDd": bas_dd}

    try:
        res = requests.get(KRX_API_URL, headers=headers, params=params, timeout=10)
        if res.status_code != 200:
            return pd.DataFrame()

        data = res.json().get("OutBlock_1", [])
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        rename_map = {
            "ISU_CD": "Ticker", "ISU_SRT_CD": "Ticker",
            "TDD_OPNPRC": "Open", "TDD_HGPRC": "High",
            "TDD_LWPRC": "Low", "TDD_CLSPRC": "Close", "ACC_TRDVOL": "Volume"
        }
        df = df.rename(columns=rename_map)
        required = ["Ticker", "Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required):
            return pd.DataFrame()

        df = df[required].copy()
        df["Ticker"] = df["Ticker"].astype(str).str[-6:].str.zfill(6)

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False).replace("-", "0"),
                errors="coerce"
            ).fillna(0)

        df["Date"] = pd.to_datetime(bas_dd, format="%Y%m%d")
        return df
    except Exception:
        return pd.DataFrame()

def get_all_tickers():
    import FinanceDataReader as fdr
    all_tickers = []
    for market in MARKETS:
        listing = fdr.StockListing(market)
        if listing.empty:
            continue
        market_tickers = listing[["Code", "Name"]].copy()
        market_tickers["Market"] = market
        market_tickers["Code"] = market_tickers["Code"].astype(str).str.zfill(6)
        all_tickers.append(market_tickers)
    return pd.concat(all_tickers, ignore_index=True).drop_duplicates(subset="Code")

def get_stock_data(ticker, krx_today_df=None):
    cache_file = CACHE_DIR / f"{ticker}.csv"
    old_data = pd.read_csv(cache_file, index_col=0, parse_dates=True).sort_index() if cache_file.exists() else pd.DataFrame()
    fetch_start = old_data.index.max().strftime("%Y%m%d") if not old_data.empty else START_DATE

    try:
        new_data = stock.get_market_ohlcv_by_date(fetch_start, END_DATE, ticker)
        if not new_data.empty:
            new_data = new_data.rename(columns={
                "시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"
            })[["Open", "High", "Low", "Close", "Volume"]]
            data = new_data if old_data.empty else pd.concat([old_data, new_data])
        else:
            data = old_data

        if krx_today_df is not None and not krx_today_df.empty:
            today_row = krx_today_df[krx_today_df["Ticker"] == ticker]
            if not today_row.empty:
                today_row = today_row.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
                data = pd.concat([data, today_row])

        if data.empty:
            return None

        data = data[~data.index.duplicated(keep="last")].sort_index()
        data = data[(data["Open"] > 0) & (data["Close"] > 0)]
        data.to_csv(cache_file, encoding="utf-8-sig")
        return data
    except Exception:
        return old_data if not old_data.empty else None

# =========================================================
# 피처 생성 및 분석
# =========================================================
def make_features(data):
    df = data.copy()
    df["Return_1"] = df["Close"].pct_change()
    df["Return_5"] = df["Close"].pct_change(5)
    df["Return_20"] = df["Close"].pct_change(20)
    df["Return_60"] = df["Close"].pct_change(60)

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["MA120"] = df["Close"].rolling(120).mean()

    df["MA5_Ratio"] = df["Close"] / df["MA5"] - 1
    df["MA20_Ratio"] = df["Close"] / df["MA20"] - 1
    df["MA60_Ratio"] = df["Close"] / df["MA60"] - 1
    df["MA120_Ratio"] = df["Close"] / df["MA120"] - 1

    df["MA_Cross"] = df["MA5"] / df["MA20"] - 1
    df["Long_Trend"] = df["MA20"] / df["MA60"] - 1

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Diff"] = df["MACD"] - df["MACD_Signal"]

    df["Volatility_20"] = df["Return_1"].rolling(20).std()
    df["Volatility_60"] = df["Return_1"].rolling(60).std()
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    df["Daily_Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["Open_Close_Ratio"] = df["Close"] / df["Open"] - 1

    df["Target_Return"] = df["Open"].shift(-2) / df["Open"].shift(-1) - 1
    df["Target"] = (df["Target_Return"] > 0).astype(int)

    features = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    valid_features = features.dropna()
    if valid_features.empty:
        return None

    latest_date = valid_features.index[-1]
    trainable = df.loc[df.index.isin(valid_features.index) & (df.index < latest_date)].copy().dropna(subset=FEATURE_COLS + ["Target_Return"])
    if len(trainable) < 300:
        return None

    return df, trainable, valid_features.loc[[latest_date]], latest_date

def backtest_stock(trainable):
    data = trainable.copy()
    data["Year"] = data.index.year
    years = sorted(data["Year"].unique())
    if len(years) <= MIN_TRAIN_YEARS:
        return None

    all_results = []
    position_state = 0

    for year in years[MIN_TRAIN_YEARS:]:
        train = data[data["Year"] < year]
        test = data[data["Year"] == year]
        if len(train) < 300 or len(test) == 0 or train["Target"].nunique() < 2:
            continue

        model = create_model()
        model.fit(train[FEATURE_COLS], train["Target"])
        up_probs = model.predict_proba(test[FEATURE_COLS])[:, list(model.classes_).index(1)]

        result = pd.DataFrame(index=test.index)
        result["Probability"] = up_probs
        result["Actual"] = test["Target"]
        result["Market_Return"] = test["Target_Return"]

        positions = []
        for prob in up_probs:
            if prob >= BUY_THRESHOLD:
                position_state = 1
            elif prob <= SELL_THRESHOLD:
                position_state = 0
            positions.append(position_state)

        result["Position"] = positions
        previous = result["Position"].shift(1, fill_value=0)
        result["Trade_Cost"] = (result["Position"] - previous).abs() * TRADING_COST
        result["AI_Return"] = result["Position"] * result["Market_Return"] - result["Trade_Cost"]
        result["BuyHold_Return"] = result["Market_Return"]
        all_results.append(result)

    return pd.concat(all_results).sort_index() if all_results else None

def save_csv_safely(data, path):
    path = Path(path)
    temp_path = path.with_name(path.stem + "_temp" + path.suffix)
    data.to_csv(temp_path, index=False, encoding="utf-8-sig")
    temp_path.replace(path)

# =========================================================
# 메인 파이프라인 실행
# =========================================================
def run_analysis_pipeline(max_stocks=MAX_STOCKS):
    tickers = get_all_tickers()
    if max_stocks is not None:
        tickers = tickers.head(max_stocks)

    total = len(tickers)
    krx_today_df = fetch_krx_daily_snapshot(END_DATE)
    
    # 기존에 분석된 결과가 있다면 불러오기 (오늘 날짜 기준 중복 실행 방지용)
    existing_signals = {}
    signals_path = OUTPUT_DIR / "latest_signals.csv"
    if signals_path.exists():
        df_old = pd.read_csv(signals_path, dtype={"Ticker": str})
        if not df_old.empty and "Date" in df_old.columns:
            today_str = datetime.now().strftime("%Y-%m-%d")
            for _, row in df_old[df_old["Date"] == today_str].iterrows():
                existing_signals[row["Ticker"]] = row.to_dict()

    signal_dict = existing_signals.copy()  # 기존 분석 결과 유지
    backtest_results = []
    today_str = datetime.now().strftime("%Y-%m-%d")

    for i, row in enumerate(tickers.itertuples(index=False), start=1):
        ticker = str(row.Code).zfill(6)
        name = row.Name
        market = row.Market

        # 오늘 이미 분석된 종목이라면 스킵
        if ticker in signal_dict and signal_dict[ticker].get("Date") == today_str:
            print(f"[{i}/{total}] 이미 분석된 종목 스킵: {name} ({ticker})")
            continue

        print(f"[{i}/{total}] {name} ({ticker}) 최신 데이터 학습 및 예측 중...")
        try:
            data = get_stock_data(ticker, krx_today_df)
            if data is None or len(data) < 400:
                continue

            feature_result = make_features(data)
            if feature_result is None:
                continue

            df, trainable, latest_features, latest_date = feature_result
            if trainable["Target"].nunique() < 2:
                continue

            model = create_model()
            model.fit(trainable[FEATURE_COLS], trainable["Target"])

            up_prob = float(model.predict_proba(latest_features)[0][list(model.classes_).index(1)])
            down_prob = 1 - up_prob
            holding = ticker in HOLDINGS
            signal = "매도 검토" if holding and up_prob <= SELL_THRESHOLD else ("매수 검토" if not holding and up_prob >= BUY_THRESHOLD else ("보유" if holding else "관망"))

            signal_dict[ticker] = {
                "Ticker": ticker, "Name": name, "Market": market,
                "Date": latest_date.strftime("%Y-%m-%d"),
                "Close": int(df.loc[latest_date, "Close"]),
                "Up_Probability": round(up_prob, 4),
                "Down_Probability": round(down_prob, 4),
                "Holding": holding, "Signal": signal,
                "Training_Data": len(trainable)
            }

            backtest = backtest_stock(trainable)
            if backtest is not None:
                backtest["Ticker"] = ticker
                backtest["Name"] = name
                backtest["Market"] = market
                backtest_results.append(backtest)

        except Exception as e:
            print(f"오류 ({ticker}): {e}")

        time.sleep(REQUEST_DELAY)

    if signal_dict:
        signals = pd.DataFrame(list(signal_dict.values())).sort_values("Up_Probability", ascending=False)
        save_csv_safely(signals, OUTPUT_DIR / "latest_signals.csv")

    if backtest_results:
        all_backtests = pd.concat(backtest_results).sort_index()
        stock_metrics = []

        for ticker, group in all_backtests.groupby("Ticker"):
            group = group.sort_index()
            equity = INITIAL_CASH * (1 + group["AI_Return"]).cumprod()
            buyhold_equity = INITIAL_CASH * (1 + group["BuyHold_Return"]).cumprod()
            drawdown = equity / equity.cummax() - 1
            prediction = (group["Probability"] >= BUY_THRESHOLD).astype(int)

            stock_metrics.append({
                "Ticker": ticker, "Name": group["Name"].iloc[0], "Market": group["Market"].iloc[0],
                "AI_Return": round(equity.iloc[-1] / INITIAL_CASH - 1, 4),
                "BuyHold_Return": round(buyhold_equity.iloc[-1] / INITIAL_CASH - 1, 4),
                "MDD": round(drawdown.min(), 4),
                "Accuracy": round(accuracy_score(group["Actual"], prediction), 4),
                "Precision": round(precision_score(group["Actual"], prediction, zero_division=0), 4),
                "Recall": round(recall_score(group["Actual"], prediction, zero_division=0), 4),
                "Trading_Days": int((group["Position"] > 0).sum())
            })

        save_csv_safely(pd.DataFrame(stock_metrics), OUTPUT_DIR / "backtest_by_stock.csv")

        portfolio = all_backtests.groupby(level=0)["AI_Return"].mean().sort_index()
        portfolio_df = pd.DataFrame({
            "Date": portfolio.index,
            "AI_Return": portfolio.values,
            "Equity": (INITIAL_CASH * (1 + portfolio).cumprod()).values
        })
        save_csv_safely(portfolio_df, OUTPUT_DIR / "portfolio_curve.csv")

    print("\n✅ 모든 학습 및 예측 결과 파일 생성 완료!")

def git_auto_push():
    print("\n🚀 깃허브로 결과 파일 자동 업로드 중...")
    try:
        # -f 옵션을 추가하여 .gitignore의 영향을 받지 않고 results 폴더의 CSV만 강제 업로드
        subprocess.run(["git", "add", "-f", "multi_stock_results/results/*.csv"], check=True)
        
        today = datetime.now().strftime("%Y-%m-%d")
        subprocess.run(["git", "commit", "-m", f"Auto-update stock analysis results: {today}"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("✅ 깃허브 자동 업로드 완료!")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ 깃허브 자동 업로드 중 오류 발생 (수동으로 확인해 주세요): {e}")

if __name__ == "__main__":
    run_analysis_pipeline()
    git_auto_push()