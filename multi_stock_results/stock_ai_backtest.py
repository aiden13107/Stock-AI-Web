import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from pykrx import stock
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. 환경 변수(.env) 파일 로드 (API 키 등 민감한 정보 보호)
load_dotenv()

warnings.filterwarnings("ignore")

# =========================================================
# 1. 사용자 설정 및 환경 변수 연동
# =========================================================

# 코드에 직접 적지 않고 .env 파일이나 시스템 환경변수에서 가져옵니다.
# (.env 파일 예시: KRX_API_KEY=발급받은키값)
KRX_API_KEY = os.getenv("KRX_API_KEY")
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

# None이면 전체 종목 / 테스트 시 10 또는 20으로 제한 가능
MAX_STOCKS = None

# 실제 보유 중인 종목 코드 입력 (예: {"005930", "000660"})
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
# 2. 분석 지표 및 모델 설정
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
# 3. KRX API 일별 전종목 시세 일괄 조회 (실시간 일봉 업데이트용)
# =========================================================

def fetch_krx_daily_snapshot(bas_dd):
    if not KRX_API_KEY or KRX_API_KEY == "YOUR_KRX_API_KEY_HERE":
        return pd.DataFrame()

    headers = {
        "AUTH_KEY": KRX_API_KEY,
        "Accept": "application/json"
    }
    params = {"basDd": bas_dd}

    try:
        res = requests.get(KRX_API_URL, headers=headers, params=params, timeout=10)
        if res.status_code != 200:
            return pd.DataFrame()

        data = res.json()
        items = data.get("OutBlock_1", [])
        if not items:
            return pd.DataFrame()

        df = pd.DataFrame(items)
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
        if not all(col in df.columns for col in required):
            return pd.DataFrame()

        df = df[required].copy()
        df["Ticker"] = df["Ticker"].astype(str).str[-6:].str.zfill(6)

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .replace("-", "0")
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df["Date"] = pd.to_datetime(bas_dd, format="%Y%m%d")
        return df

    except Exception as e:
        print(f"KRX API 조회 오류 ({bas_dd}): {e}")
        return pd.DataFrame()

# =========================================================
# 4. 종목 목록 및 주가 데이터 수집
# =========================================================

def get_all_tickers():
    import FinanceDataReader as fdr

    print("\n[1] 코스피·코스닥 종목 목록 수집")
    all_tickers = []

    for market in MARKETS:
        listing = fdr.StockListing(market)
        if listing.empty:
            continue
        market_tickers = listing[["Code", "Name"]].copy()
        market_tickers["Market"] = market
        market_tickers["Code"] = market_tickers["Code"].astype(str).str.zfill(6)
        all_tickers.append(market_tickers)

    if not all_tickers:
        raise RuntimeError("종목 목록을 가져오지 못했습니다.")

    result = pd.concat(all_tickers, ignore_index=True).drop_duplicates(subset="Code")
    result.to_csv(BASE_DIR / "ticker_list.csv", index=False, encoding="utf-8-sig")
    print(f"총 종목 수: {len(result)}개")
    return result

def get_stock_data(ticker, krx_today_df=None):
    cache_file = CACHE_DIR / f"{ticker}.csv"
    old_data = pd.DataFrame()

    if cache_file.exists():
        try:
            old_data = pd.read_csv(cache_file, index_col=0, parse_dates=True).sort_index()
        except Exception:
            old_data = pd.DataFrame()

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

    except Exception as e:
        print(f"{ticker} 데이터 수집 실패: {str(e)[:80]}")
        return old_data if not old_data.empty else None

# =========================================================
# 5. 기술적 지표 및 피처 생성
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

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Diff"] = df["MACD"] - df["MACD_Signal"]

    # 변동성 및 거래량
    df["Volatility_20"] = df["Return_1"].rolling(20).std()
    df["Volatility_60"] = df["Return_1"].rolling(60).std()
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # 캔들 비율
    df["Daily_Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["Open_Close_Ratio"] = df["Close"] / df["Open"] - 1

    # 타겟: 다음 거래일 시가 -> 그다음 거래일 시가 수익률
    df["Target_Return"] = df["Open"].shift(-2) / df["Open"].shift(-1) - 1
    df["Target"] = (df["Target_Return"] > 0).astype(int)

    features = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    valid_features = features.dropna()

    if valid_features.empty:
        return None

    latest_date = valid_features.index[-1]

    trainable = df.loc[
        df.index.isin(valid_features.index) & (df.index < latest_date)
    ].copy().dropna(subset=FEATURE_COLS + ["Target_Return"])

    if len(trainable) < 300:
        return None

    latest_features = valid_features.loc[[latest_date]]
    return df, trainable, latest_features, latest_date

# =========================================================
# 6. 예측 및 워크포워드 백테스트
# =========================================================

def get_up_probability(model, X):
    probabilities = model.predict_proba(X)[0]
    if 1 not in model.classes_:
        return 0.0
    index = list(model.classes_).index(1)
    return float(probabilities[index])

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

        probabilities = model.predict_proba(test[FEATURE_COLS])
        positive_index = list(model.classes_).index(1)
        up_probs = probabilities[:, positive_index]

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

def analyze_stock(ticker, market, name, krx_today_df=None):
    data = get_stock_data(ticker, krx_today_df)
    if data is None or len(data) < 400:
        return None, None

    feature_result = make_features(data)
    if feature_result is None:
        return None, None

    df, trainable, latest_features, latest_date = feature_result
    if trainable["Target"].nunique() < 2:
        return None, None

    model = create_model()
    model.fit(trainable[FEATURE_COLS], trainable["Target"])

    up_probability = get_up_probability(model, latest_features)
    down_probability = 1 - up_probability
    holding = ticker in HOLDINGS

    if holding:
        signal = "매도 검토" if up_probability <= SELL_THRESHOLD else "보유"
    else:
        signal = "매수 검토" if up_probability >= BUY_THRESHOLD else "관망"

    latest_close = df.loc[latest_date, "Close"]

    signal_data = {
        "Ticker": ticker,
        "Name": name,
        "Market": market,
        "Date": latest_date.strftime("%Y-%m-%d"),
        "Close": int(latest_close),
        "Up_Probability": round(up_probability, 4),
        "Down_Probability": round(down_probability, 4),
        "Holding": holding,
        "Signal": signal,
        "Training_Data": len(trainable)
    }

    backtest = backtest_stock(trainable)
    if backtest is not None:
        backtest["Ticker"] = ticker
        backtest["Name"] = name
        backtest["Market"] = market

    return signal_data, backtest

# =========================================================
# 7. 백엔드 분석 파이프라인 실행 함수
# =========================================================

def run_analysis_pipeline(max_stocks=MAX_STOCKS, reset_progress=False):
    tickers = get_all_tickers()
    if max_stocks is not None:
        tickers = tickers.head(max_stocks)

    total = len(tickers)

    print(f"\n[2] KRX API 당일 시세 조회 ({END_DATE})")
    krx_today_df = fetch_krx_daily_snapshot(END_DATE)
    if not krx_today_df.empty:
        print(f"KRX API 당일 데이터 확보 완료: {len(krx_today_df)}건")
    else:
        print("KRX API 키가 미설정되었거나 장 마감 전입니다. pykrx 데이터로 진행합니다.")

    if reset_progress and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

    signal_dict = {}
    partial_path = OUTPUT_DIR / "latest_signals_partial.csv"
    if partial_path.exists() and not reset_progress:
        df_part = pd.read_csv(partial_path, dtype={"Ticker": str})
        df_part["Ticker"] = df_part["Ticker"].str.zfill(6)
        signal_dict = {r["Ticker"]: r.to_dict() for _, r in df_part.iterrows()}

    completed_status = {}
    if PROGRESS_FILE.exists() and not reset_progress:
        prog_df = pd.read_csv(PROGRESS_FILE, dtype={"Ticker": str})
        for _, r in prog_df.iterrows():
            completed_status[str(r["Ticker"]).zfill(6)] = str(r["Status"])

    completed = set(completed_status.keys())
    backtest_results = []

    for ticker in completed:
        bt_file = BACKTEST_CACHE_DIR / f"{ticker}.csv"
        if bt_file.exists():
            saved_bt = pd.read_csv(bt_file, index_col=0, parse_dates=True)
            if not saved_bt.empty:
                backtest_results.append(saved_bt)

    for i, row in enumerate(tickers.itertuples(index=False), start=1):
        ticker = str(row.Code).zfill(6)
        market = row.Market
        name = row.Name

        if ticker in completed:
            continue

        print(f"[{i}/{total}] {name} ({ticker}) 학습 및 예측 중...")
        try:
            signal, backtest = analyze_stock(ticker, market, name, krx_today_df)
            if signal is not None:
                signal_dict[ticker] = signal
                save_csv_safely(pd.DataFrame(list(signal_dict.values())), partial_path)

            if backtest is not None:
                backtest.reset_index().to_csv(
                    BACKTEST_CACHE_DIR / f"{ticker}.csv", index=False, encoding="utf-8-sig"
                )
                backtest_results.append(backtest)

            completed_status[ticker] = "DONE" if signal is not None else "SKIPPED"
            completed.add(ticker)

            rows = [{"Ticker": k, "Status": v} for k, v in completed_status.items()]
            save_csv_safely(pd.DataFrame(rows), PROGRESS_FILE)

        except Exception as e:
            print(f"오류 발생 ({ticker}): {e}")

        time.sleep(REQUEST_DELAY)

    if signal_dict:
        signals = pd.DataFrame(list(signal_dict.values()))
        signals["Ticker"] = signals["Ticker"].astype(str).str.zfill(6)
        signals = signals.sort_values("Up_Probability", ascending=False)
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
                "Ticker": ticker,
                "Name": group["Name"].iloc[0],
                "Market": group["Market"].iloc[0],
                "AI_Return": round(equity.iloc[-1] / INITIAL_CASH - 1, 4),
                "BuyHold_Return": round(buyhold_equity.iloc[-1] / INITIAL_CASH - 1, 4),
                "MDD": round(drawdown.min(), 4),
                "Accuracy": round(accuracy_score(group["Actual"], prediction), 4),
                "Precision": round(precision_score(group["Actual"], prediction, zero_division=0), 4),
                "Recall": round(recall_score(group["Actual"], prediction, zero_division=0), 4),
                "Trading_Days": int((group["Position"] > 0).sum())
            })

        metrics = pd.DataFrame(stock_metrics)
        save_csv_safely(metrics, OUTPUT_DIR / "backtest_by_stock.csv")

        portfolio = all_backtests.groupby(level=0)["AI_Return"].mean().sort_index()
        portfolio_df = pd.DataFrame({
            "Date": portfolio.index,
            "AI_Return": portfolio.values,
            "Equity": (INITIAL_CASH * (1 + portfolio).cumprod()).values
        })
        save_csv_safely(portfolio_df, OUTPUT_DIR / "portfolio_curve.csv")

    print("\n✅ 모든 학습 및 예측 작업이 완료되었습니다.")

# =========================================================
# 8. Streamlit 웹 앱(대시보드) UI
# =========================================================

def run_streamlit_app():
    import streamlit as st

    st.set_page_config(
        page_title="AI 일봉 주식 예측 대시보드",
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 AI 일봉 데이터 기반 다음 날 상승 유망 종목 추천 앱")
    st.caption("KRX 일별 시세 및 기술적 지표(RandomForest)를 학습하여 다음 거래일 상승 확률을 예측합니다.")

    with st.sidebar:
        st.header("⚙️ 분석 엔진 제어")
        test_limit = st.number_input("분석 종목 수 제한 (0 = 전체)", min_value=0, max_value=3000, value=30, step=10)
        reset_chk = st.checkbox("기존 진행 기록 초기화 후 재학습", value=False)

        if st.button("🚀 최신 일봉 학습 및 예측 실행", use_container_width=True):
            limit_val = None if test_limit == 0 else int(test_limit)
            with st.spinner("일봉 데이터 수집 및 AI 모델 학습 중입니다..."):
                run_analysis_pipeline(max_stocks=limit_val, reset_progress=reset_chk)
            st.success("최신 예측 결과가 업데이트되었습니다!")
            st.rerun()

        st.divider()
        st.markdown("**연동 정보**")
        st.code("KRX API: stk_bydd_trd\n모델: RandomForest\n매수 기준: >= 55.0%\n매도 기준: <= 45.0%")

    signals_file = OUTPUT_DIR / "latest_signals.csv"
    metrics_file = OUTPUT_DIR / "backtest_by_stock.csv"
    curve_file = OUTPUT_DIR / "portfolio_curve.csv"

    if not signals_file.exists():
        st.warning("아직 생성된 분석 결과가 없습니다. 왼쪽 사이드바에서 **[🚀 최신 일봉 학습 및 예측 실행]** 버튼을 눌러주세요.")
        return

    signals = pd.read_csv(signals_file, dtype={"Ticker": str})
    signals["Ticker"] = signals["Ticker"].str.zfill(6)
    signals["상승확률(%)"] = (signals["Up_Probability"] * 100).round(2)

    col1, col2, col3, col4 = st.columns(4)
    buy_candidates = signals[signals["Signal"] == "매수 검토"]

    col1.metric("분석 완료 종목", f"{len(signals):,}개")
    col2.metric("내일 매수 검토 추천", f"{len(buy_candidates):,}개")
    col3.metric("기준 일자", str(signals["Date"].iloc[0]) if not signals.empty else "-")
    top_stock_name = f"{signals.iloc[0]['Name']} ({signals.iloc[0]['상승확률(%)']}%)" if not signals.empty else "-"
    col4.metric("최고 상승 확률 종목", top_stock_name)

    st.divider()

    tab1, tab2, tab3 = st.tabs(["🔥 내일 상승 유망 종목 추천", "📊 워크포워드 모의투자 성과", "🔍 전체 종목 검색"])

    with tab1:
        st.subheader("AI 상승 확률 TOP 30 종목")

        f_col1, f_col2 = st.columns([1, 1])
        with f_col1:
            market_filter = st.multiselect("시장 선택", ["KOSPI", "KOSDAQ"], default=["KOSPI", "KOSDAQ"])
        with f_col2:
            signal_filter = st.multiselect("AI 시그널 필터", ["매수 검토", "보유", "관망", "매도 검토"], default=["매수 검토", "보유"])

        filtered = signals[
            (signals["Market"].isin(market_filter)) &
            (signals["Signal"].isin(signal_filter))
        ].head(30)

        display_cols = ["Ticker", "Name", "Market", "Date", "Close", "상승확률(%)", "Signal"]
        st.dataframe(
            filtered[display_cols].rename(columns={
                "Ticker": "종목코드", "Name": "종목명", "Market": "시장",
                "Date": "기준일", "Close": "현재가(원)", "Signal": "AI 시그널"
            }),
            width="stretch",
            hide_index=True
        )

    with tab2:
        if curve_file.exists() and metrics_file.exists():
            curve_df = pd.read_csv(curve_file)
            metrics_df = pd.read_csv(metrics_file, dtype={"Ticker": str})

            final_equity = curve_df["Equity"].iloc[-1]
            total_ret = (final_equity / INITIAL_CASH - 1) * 100

            st.subheader(f"동일 가중 AI 포트폴리오 누적 자산 추이 (수익률: {total_ret:+.2f}%)")
            st.line_chart(curve_df.set_index("Date")["Equity"])

            st.subheader("백테스트 누적 수익률 상위 20개 종목")
            metrics_df["AI 수익률(%)"] = (metrics_df["AI_Return"] * 100).round(2)
            metrics_df["단순보유 수익률(%)"] = (metrics_df["BuyHold_Return"] * 100).round(2)
            metrics_df["최대낙폭 MDD(%)"] = (metrics_df["MDD"] * 100).round(2)
            metrics_df["정확도(%)"] = (metrics_df["Accuracy"] * 100).round(2)

            top_metrics = metrics_df.sort_values("AI_Return", ascending=False).head(20)
            st.dataframe(
                top_metrics[["Ticker", "Name", "Market", "AI 수익률(%)", "단순보유 수익률(%)", "최대낙폭 MDD(%)", "정확도(%)"]],
                width="stretch",
                hide_index=True
            )
        else:
            st.info("백테스트 결과 파일이 아직 생성되지 않았습니다.")

    with tab3:
        st.subheader("전체 분석 종목 검색")
        search_query = st.text_input("종목명 또는 종목코드 입력", placeholder="예: 삼성전자 또는 005930")
        if search_query:
            res = signals[
                signals["Name"].str.contains(search_query, case=False, na=False) |
                signals["Ticker"].str.contains(search_query, na=False)
            ]
        else:
            res = signals

        st.dataframe(
            res[["Ticker", "Name", "Market", "Date", "Close", "상승확률(%)", "Signal"]],
            width="stretch",
            hide_index=True
        )

# =========================================================
# 9. 실행 진입점
# =========================================================

if __name__ == "__main__":
    if "--analyze" in sys.argv:
        run_analysis_pipeline()
    else:
        run_streamlit_app()