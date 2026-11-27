import os
import time
import json
import hashlib
import warnings
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from pykrx import stock
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score

warnings.filterwarnings("ignore")

# =========================================================
# 설정
# =========================================================
load_dotenv()

KRX_API_KEY = os.getenv("KRX_API_KEY", "")
KRX_API_URL = "https://data-dbg.krx.co.kr/svc/apis/sto/stk_bydd_trd"

START_DATE = "20160101"
END_DATE = datetime.now().strftime("%Y%m%d")
MARKETS = ["KOSPI", "KOSDAQ"]

INITIAL_CASH = 10_000_000
TRADING_COST = 0.0015
BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45

MIN_TRAIN_YEARS = 5
MIN_TRAIN_ROWS = 300

# 속도/성능 절충 모델
N_ESTIMATORS = 50
MAX_DEPTH = 6
MIN_SAMPLES_LEAF = 20
MAX_SAMPLES = 0.8

# False: 최신 신호만 계산 (빠름)
# True : 백테스트도 실행
RUN_BACKTEST = False

MAX_STOCKS = None  # 테스트할 때 30 등으로 설정
HOLDINGS = set()
REQUEST_DELAY = 0.2

BASE_DIR = Path("multi_stock_results")
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "results"
BACKTEST_CACHE_DIR = OUTPUT_DIR / "backtest_cache"

for folder in (CACHE_DIR, OUTPUT_DIR, BACKTEST_CACHE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

SIGNALS_FILE = OUTPUT_DIR / "latest_signals.csv"
BACKTEST_METRICS_FILE = OUTPUT_DIR / "backtest_by_stock.csv"
PORTFOLIO_FILE = OUTPUT_DIR / "portfolio_curve.csv"

FEATURE_COLS = [
    "Return_1", "Return_5", "Return_20", "Return_60",
    "MA5_Ratio", "MA20_Ratio", "MA60_Ratio", "MA120_Ratio",
    "MA_Cross", "Long_Trend",
    "RSI", "MACD", "MACD_Diff",
    "Volatility_20", "Volatility_60",
    "Volume_Change", "Volume_Ratio",
    "Daily_Range", "Open_Close_Ratio"
]


# =========================================================
# 모델
# =========================================================
def create_model():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_samples=MAX_SAMPLES,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    
def create_return_model():
    return RandomForestRegressor(
        n_estimators=50,
        max_depth=6,
        min_samples_leaf=20,
        max_samples=0.8,
        random_state=43,
        n_jobs=-1
    )


def model_signature():
    settings = {
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "max_samples": MAX_SAMPLES,
        "buy": BUY_THRESHOLD,
        "sell": SELL_THRESHOLD,
        "cost": TRADING_COST,
        "features": FEATURE_COLS
    }
    return hashlib.md5(
        json.dumps(settings, sort_keys=True).encode("utf-8")
    ).hexdigest()


# =========================================================
# 파일 저장
# =========================================================
def save_csv_safely(data, path, index=False, index_label=None):
    path = Path(path)
    temp_path = path.with_name(path.stem + "_temp" + path.suffix)
    data.to_csv(
        temp_path,
        index=index,
        index_label=index_label,
        encoding="utf-8-sig"
    )
    temp_path.replace(path)


# =========================================================
# KRX 데이터 수집
# =========================================================
def fetch_krx_daily_snapshot(bas_dd):
    if not KRX_API_KEY:
        return pd.DataFrame()

    headers = {"AUTH_KEY": KRX_API_KEY, "Accept": "application/json"}
    params = {"basDd": bas_dd}

    try:
        response = requests.get(
            KRX_API_URL, headers=headers, params=params, timeout=15
        )
        if response.status_code != 200:
            return pd.DataFrame()

        data = response.json().get("OutBlock_1", [])
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        rename_map = {
            "ISU_CD": "Ticker",
            "ISU_SRT_CD": "Ticker",
            "TDD_OPNPRC": "Open",
            "TDD_HGPRC": "High",
            "TDD_LWPRC": "Low",
            "TDD_CLSPRC": "Close",
            "ACC_TRDVOL": "Volume"
        }
        df = df.rename(columns=rename_map)

        required = ["Ticker", "Open", "High", "Low", "Close", "Volume"]
        if not all(column in df.columns for column in required):
            return pd.DataFrame()

        df = df[required].copy()
        df["Ticker"] = df["Ticker"].astype(str).str[-6:].str.zfill(6)

        for column in ["Open", "High", "Low", "Close", "Volume"]:
            df[column] = pd.to_numeric(
                df[column].astype(str)
                .str.replace(",", "", regex=False)
                .replace("-", "0"),
                errors="coerce"
            ).fillna(0)

        df["Date"] = pd.to_datetime(bas_dd, format="%Y%m%d")
        return df

    except Exception as error:
        print(f"KRX snapshot 오류: {error}")
        return pd.DataFrame()


def get_all_tickers():
    import FinanceDataReader as fdr

    listings = []
    for market in MARKETS:
        try:
            listing = fdr.StockListing(market)
            if listing.empty:
                continue

            part = listing[["Code", "Name"]].copy()
            part["Market"] = market
            part["Code"] = part["Code"].astype(str).str.zfill(6)
            listings.append(part)

        except Exception as error:
            print(f"{market} 종목 목록 오류: {error}")

    if not listings:
        return pd.DataFrame(columns=["Code", "Name", "Market"])

    return pd.concat(listings, ignore_index=True).drop_duplicates(
        subset="Code"
    )


def get_stock_data(ticker, krx_today_df=None):
    cache_file = CACHE_DIR / f"{ticker}.csv"

    try:
        old_data = (
            pd.read_csv(cache_file, index_col=0, parse_dates=True).sort_index()
            if cache_file.exists()
            else pd.DataFrame()
        )
    except Exception:
        old_data = pd.DataFrame()

    # 캐시 마지막 날짜부터 다시 조회해 마지막 거래일 수정 데이터도 반영
    fetch_start = (
        old_data.index.max().strftime("%Y%m%d")
        if not old_data.empty
        else START_DATE
    )

    try:
        new_data = stock.get_market_ohlcv_by_date(
            fetch_start, END_DATE, ticker
        )

        if not new_data.empty:
            new_data = new_data.rename(columns={
                "시가": "Open",
                "고가": "High",
                "저가": "Low",
                "종가": "Close",
                "거래량": "Volume"
            })[["Open", "High", "Low", "Close", "Volume"]]

            data = (
                new_data if old_data.empty
                else pd.concat([old_data, new_data])
            )
        else:
            data = old_data

        if krx_today_df is not None and not krx_today_df.empty:
            today_row = krx_today_df[
                krx_today_df["Ticker"] == ticker
            ]
            if not today_row.empty:
                today_row = today_row.set_index("Date")[
                    ["Open", "High", "Low", "Close", "Volume"]
                ]
                data = pd.concat([data, today_row])

        if data.empty:
            return None

        data = data[~data.index.duplicated(keep="last")].sort_index()
        data = data[
            (data["Open"] > 0) &
            (data["Close"] > 0) &
            (data["High"] > 0) &
            (data["Low"] > 0)
        ]

        save_csv_safely(data, cache_file, index=True)
        return data

    except Exception as error:
        print(f"데이터 수집 오류 ({ticker}): {error}")
        return old_data if not old_data.empty else None


# =========================================================
# 피처 생성
# =========================================================
def make_features(data):
    df = data.copy()

    df["Return_1"] = df["Close"].pct_change()
    df["Return_5"] = df["Close"].pct_change(5)
    df["Return_20"] = df["Close"].pct_change(20)
    df["Return_60"] = df["Close"].pct_change(60)

    for window in [5, 20, 60, 120]:
        df[f"MA{window}"] = df["Close"].rolling(window).mean()
        df[f"MA{window}_Ratio"] = df["Close"] / df[f"MA{window}"] - 1

    df["MA_Cross"] = df["MA5"] / df["MA20"] - 1
    df["Long_Trend"] = df["MA20"] / df["MA60"] - 1

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()

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
    df["Volume_Ratio"] = (
        df["Volume"] / df["Volume"].rolling(20).mean()
    )

    df["Daily_Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["Open_Close_Ratio"] = df["Close"] / df["Open"] - 1

    # t일 신호로 t+1 시가 매수, t+2 시가 매도 가정
    df["Target_Return"] = (
        df["Open"].shift(-2) / df["Open"].shift(-1) - 1
    )
    df["Target"] = (df["Target_Return"] > 0).astype(np.int8)

    features = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    valid_features = features.dropna()

    if valid_features.empty:
        return None

    latest_date = valid_features.index[-1]

    trainable = df.loc[
        (df.index.isin(valid_features.index)) &
        (df.index < latest_date)
    ].dropna(subset=FEATURE_COLS + ["Target_Return"])

    if len(trainable) < MIN_TRAIN_ROWS:
        return None

    return df, trainable, valid_features.loc[[latest_date]], latest_date


# =========================================================
# 백테스트
# =========================================================
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

        if (
            len(train) < MIN_TRAIN_ROWS or
            test.empty or
            train["Target"].nunique() < 2
        ):
            continue

        model = create_model()
        X_train = train[FEATURE_COLS].to_numpy(dtype=np.float32)
        y_train = train["Target"].to_numpy(dtype=np.int8)
        X_test = test[FEATURE_COLS].to_numpy(dtype=np.float32)

        model.fit(X_train, y_train)

        class_index = list(model.classes_).index(1)
        up_probs = model.predict_proba(X_test)[:, class_index]

        result = pd.DataFrame(index=test.index)
        result["Probability"] = up_probs
        result["Actual"] = test["Target"]
        result["Market_Return"] = test["Target_Return"]

        positions = []
        for probability in up_probs:
            if probability >= BUY_THRESHOLD:
                position_state = 1
            elif probability <= SELL_THRESHOLD:
                position_state = 0
            positions.append(position_state)

        result["Position"] = positions
        previous = result["Position"].shift(1, fill_value=0)

        result["Trade_Cost"] = (
            (result["Position"] - previous).abs() * TRADING_COST
        )
        result["AI_Return"] = (
            result["Position"] * result["Market_Return"] -
            result["Trade_Cost"]
        )
        result["BuyHold_Return"] = result["Market_Return"]

        all_results.append(result)

    if not all_results:
        return None

    return pd.concat(all_results).sort_index()


def get_cached_backtest(ticker, trainable):
    """
    같은 종목/데이터 끝 날짜/행 수/모델 설정이면 기존 백테스트 재사용.
    데이터가 늘어나거나 설정이 바뀌면 다시 계산.
    """
    cache_file = BACKTEST_CACHE_DIR / f"{ticker}.csv"
    metadata_file = BACKTEST_CACHE_DIR / f"{ticker}.json"

    signature = {
        "last_date": trainable.index.max().strftime("%Y-%m-%d"),
        "rows": int(len(trainable)),
        "model": model_signature()
    }

    if cache_file.exists() and metadata_file.exists():
        try:
            old_signature = json.loads(
                metadata_file.read_text(encoding="utf-8")
            )
            if old_signature == signature:
                cached = pd.read_csv(
                    cache_file, index_col=0, parse_dates=True
                )
                print(f"  백테스트 캐시 사용: {ticker}")
                return cached
        except Exception:
            pass

    result = backtest_stock(trainable)
    if result is None:
        return None

    save_csv_safely(result, cache_file, index=True)
    metadata_file.write_text(
        json.dumps(signature, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return result


# =========================================================
# 메인 분석
# =========================================================
def run_analysis_pipeline(max_stocks=MAX_STOCKS):
    tickers = get_all_tickers()

    if tickers.empty:
        print("종목 목록을 가져오지 못했습니다.")
        return

    if max_stocks is not None:
        tickers = tickers.head(max_stocks)

    total = len(tickers)
    print(f"분석 대상 종목: {total}개")
    print(f"백테스트 실행: {RUN_BACKTEST}")

    krx_today_df = fetch_krx_daily_snapshot(END_DATE)

    # 기존 신호 로드
    signal_dict = {}
    if SIGNALS_FILE.exists():
        try:
            old_signals = pd.read_csv(
                SIGNALS_FILE, dtype={"Ticker": str}
            )
            if not old_signals.empty:
                for _, row in old_signals.iterrows():
                    signal_dict[str(row["Ticker"]).zfill(6)] = row.to_dict()
        except Exception as error:
            print(f"기존 신호 로드 오류: {error}")

    backtest_results = []
    today_str = datetime.now().strftime("%Y-%m-%d")

    for i, row in enumerate(tickers.itertuples(index=False), start=1):
        ticker = str(row.Code).zfill(6)
        name = row.Name
        market = row.Market

        print(f"[{i}/{total}] {name} ({ticker}) 분석")

        start_time = time.time()

        try:
            data = get_stock_data(ticker, krx_today_df)
            if data is None or len(data) < 400:
                print("  데이터 부족")
                continue

            feature_result = make_features(data)
            if feature_result is None:
                print("  피처 또는 학습 데이터 부족")
                continue

            df, trainable, latest_features, latest_date = feature_result

            if trainable["Target"].nunique() < 2:
                print("  Target 클래스 부족")
                continue

            # 최신 신호 학습
            X_train = trainable[FEATURE_COLS].to_numpy(
                dtype=np.float32
            )
            y_train = trainable["Target"].to_numpy(
                dtype=np.int8
            )
            X_latest = latest_features[FEATURE_COLS].to_numpy(
                dtype=np.float32
            )

            # 상승 확률 분류 모델
            model = create_model()
            model.fit(X_train, y_train)

            class_index = list(model.classes_).index(1)

            up_prob = float(
                model.predict_proba(X_latest)[0][class_index]
            )

            down_prob = 1.0 - up_prob


            # 예상 수익률 회귀 모델
            return_model = create_return_model()

            y_return = trainable["Target_Return"].to_numpy(
                dtype=np.float32
            )

            return_model.fit(X_train, y_return)

            expected_return = float(
                return_model.predict(X_latest)[0]
            )

            expected_return = float(
                np.clip(expected_return, -1.0, 1.0)
            )

            holding = ticker in HOLDINGS
            if holding and up_prob <= SELL_THRESHOLD:
                signal = "매도 검토"
            elif not holding and up_prob >= BUY_THRESHOLD:
                signal = "매수 검토"
            elif holding:
                signal = "보유"
            else:
                signal = "관망"

            signal_dict[ticker] = {
                "Ticker": ticker,
                "Name": name,
                "Market": market,
                "Date": latest_date.strftime("%Y-%m-%d"),
                "Close": int(df.loc[latest_date, "Close"]),

                # 상승 및 하락 확률
                "Up_Probability": round(up_prob, 4),
                "Down_Probability": round(down_prob, 4),

                # 예상 수익률
                "Expected_Return": round(expected_return, 6),
                "Expected_Return_Pct": round(expected_return * 100, 2),

                "Holding": holding,
                "Signal": signal,
                "Training_Data": len(trainable)
            }

            # 선택적으로 백테스트 실행
            if RUN_BACKTEST:
                backtest = get_cached_backtest(ticker, trainable)
                if backtest is not None and not backtest.empty:
                    backtest["Ticker"] = ticker
                    backtest["Name"] = name
                    backtest["Market"] = market
                    backtest_results.append(backtest)

            elapsed = time.time() - start_time
            print(
                f"  완료 | 상승확률={up_prob:.3f} | "
                f"학습행={len(trainable)} | {elapsed:.1f}초"
            )

        except Exception as error:
            print(f"  오류 ({ticker}): {error}")

        time.sleep(REQUEST_DELAY)

    # 최신 신호 저장
    if signal_dict:
        signals = pd.DataFrame(list(signal_dict.values()))
        signals["Ticker"] = signals["Ticker"].astype(str).str.zfill(6)
        signals = signals.sort_values(
            "Up_Probability", ascending=False
        )
        save_csv_safely(signals, SIGNALS_FILE)

    # 백테스트 결과 저장
    if RUN_BACKTEST and backtest_results:
        all_backtests = pd.concat(backtest_results).sort_index()
        stock_metrics = []

        for ticker, group in all_backtests.groupby("Ticker"):
            group = group.sort_index()

            equity = INITIAL_CASH * (
                1 + group["AI_Return"]
            ).cumprod()
            buyhold_equity = INITIAL_CASH * (
                1 + group["BuyHold_Return"]
            ).cumprod()

            drawdown = equity / equity.cummax() - 1
            prediction = (
                group["Probability"] >= BUY_THRESHOLD
            ).astype(int)

            stock_metrics.append({
                "Ticker": ticker,
                "Name": group["Name"].iloc[0],
                "Market": group["Market"].iloc[0],
                "AI_Return": round(
                    equity.iloc[-1] / INITIAL_CASH - 1, 4
                ),
                "BuyHold_Return": round(
                    buyhold_equity.iloc[-1] / INITIAL_CASH - 1, 4
                ),
                "MDD": round(drawdown.min(), 4),
                "Accuracy": round(
                    accuracy_score(group["Actual"], prediction), 4
                ),
                "Precision": round(
                    precision_score(
                        group["Actual"], prediction, zero_division=0
                    ), 4
                ),
                "Recall": round(
                    recall_score(
                        group["Actual"], prediction, zero_division=0
                    ), 4
                ),
                "Trading_Days": int(
                    (group["Position"] > 0).sum()
                )
            })

        save_csv_safely(
            pd.DataFrame(stock_metrics),
            BACKTEST_METRICS_FILE
        )

        # 종목별 수익률을 날짜별 평균하여 간단한 포트폴리오 곡선 생성
        portfolio = (
            all_backtests.groupby(level=0)["AI_Return"]
            .mean()
            .sort_index()
        )

        portfolio_df = pd.DataFrame({
            "Date": portfolio.index,
            "AI_Return": portfolio.values,
            "Equity": (
                INITIAL_CASH * (1 + portfolio).cumprod()
            ).values
        })

        save_csv_safely(portfolio_df, PORTFOLIO_FILE)

    print("\n완료! 결과 폴더:", OUTPUT_DIR.resolve())


# =========================================================
# GitHub 자동 업로드
# =========================================================
def git_auto_push():
    print("\nGitHub 결과 업로드 시작")

    try:
        subprocess.run(
            ["git", "add", "multi_stock_results/results/"],
            check=True
        )

        today = datetime.now().strftime("%Y-%m-%d")
        subprocess.run(
            [
                "git", "commit", "--allow-empty",
                "-m", f"Auto-update stock analysis results: {today}"
            ],
            check=True
        )

        subprocess.run(
            ["git", "push", "origin", "main"],
            check=True
        )

        print("GitHub 업로드 완료")

    except subprocess.CalledProcessError as error:
        print(f"GitHub 업로드 오류: {error}")


if __name__ == "__main__":
    run_analysis_pipeline()
    git_auto_push()
