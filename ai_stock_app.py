
import os
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# =========================================================
# 1. 경로 설정
# =========================================================
BASE_DIR = Path("multi_stock_results")
OUTPUT_DIR = BASE_DIR / "results"

SIGNALS_FILE = OUTPUT_DIR / "latest_signals.csv"
METRICS_FILE = OUTPUT_DIR / "backtest_by_stock.csv"
CURVE_FILE = OUTPUT_DIR / "portfolio_curve.csv"

INITIAL_CASH = 10_000_000

# =========================================================
# 2. 데이터 로드 및 정리
# =========================================================
def load_signals():
    if not SIGNALS_FILE.exists():
        return None

    df = pd.read_csv(
        SIGNALS_FILE,
        dtype={"Ticker": str},
        encoding="utf-8-sig"
    )

    # 필수 컬럼 확인
    required_columns = [
        "Ticker",
        "Name",
        "Market",
        "Date",
        "Close",
        "Up_Probability",
        "Expected_Return_Pct"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        st.error(f"CSV에 필요한 컬럼이 없습니다: {missing_columns}")
        return None

    # 문자열 정리
    df["Ticker"] = (
        df["Ticker"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["Name"] = (
        df["Name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["Market"] = (
        df["Market"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # 수치형 변환
    df["Up_Probability"] = pd.to_numeric(
        df["Up_Probability"],
        errors="coerce"
    )

    df["Expected_Return_Pct"] = pd.to_numeric(
        df["Expected_Return_Pct"],
        errors="coerce"
    )

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    # 빈 티커 제거
    df = df[df["Ticker"] != ""].copy()

    # 날짜 정리
    df["Date"] = df["Date"].astype(str)

    # 상승확률이 0~1 범위인 경우 퍼센트 표시용으로 변환
    df["상승확률(%)"] = (
        df["Up_Probability"] * 100
    ).round(2)

    df["예상 상승률(%)"] = (
        df["Expected_Return_Pct"]
    ).round(2)

    return df


# =========================================================
# 3. Streamlit 대시보드
# =========================================================
def run_streamlit_app():

    st.set_page_config(
        page_title="AI 주식 예측 대시보드",
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 AI 상승 유망 종목 대시보드")
    st.caption(
        "국내 주식 및 미국 주식 분석 결과를 표시합니다."
    )

    signals = load_signals()

    if signals is None:
        st.warning(
            "분석 결과 파일을 찾을 수 없거나 "
            "필수 컬럼이 없습니다."
        )

        st.code(
            "python ai_stock_app.py --analyze"
        )
        return

    if signals.empty:
        st.warning("분석 결과가 비어 있습니다.")
        return

    # =====================================================
    # 4. 데이터 진단
    # =====================================================
    with st.expander("데이터 상태 확인", expanded=False):

        st.write("읽어온 파일:")
        st.code(str(SIGNALS_FILE.resolve()))

        st.write("전체 종목 수:", len(signals))

        st.write("시장별 종목 수:")
        st.dataframe(
            signals["Market"]
            .value_counts(dropna=False)
            .rename_axis("Market")
            .reset_index(name="종목 수"),
            hide_index=True
        )

        us_data = signals[
            signals["Market"] == "US"
        ]

        st.write("미국 주식 종목 수:", len(us_data))

        if not us_data.empty:
            st.dataframe(
                us_data[
                    [
                        "Ticker",
                        "Name",
                        "Market",
                        "Date",
                        "Close",
                        "상승확률(%)",
                        "예상 상승률(%)"
                    ]
                ].head(20),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(
                "Market 값이 정확히 US인 행이 없습니다. "
                "CSV의 Market 값을 확인해 주세요."
            )

    # =====================================================
    # 5. 보유 종목 입력 및 시그널 계산
    # =====================================================
    BUY_THRESHOLD = 0.55
    SELL_THRESHOLD = 0.45

    with st.sidebar:
        st.header("💼 보유 종목 설정")

        holdings_input = st.text_input(
            "보유 종목 코드",
            placeholder="예: 005930, 000660, AAPL, MSFT",
            help="국내 또는 미국 티커를 쉼표로 구분해 입력하세요."
        )

        user_holdings = {
            x.strip().upper()
            for x in holdings_input.split(",")
            if x.strip()
        }

        if user_holdings:
            st.success(
                f"등록된 보유 종목: {len(user_holdings)}개"
            )

        st.divider()

        st.markdown("**모델 기준**")
        st.write("매수 검토: 상승확률 ≥ 55%")
        st.write("매도 검토: 상승확률 ≤ 45%")

    signals["Ticker"] = signals["Ticker"].str.upper()

    def dynamic_signal(row):
        ticker = str(row["Ticker"]).strip().upper()
        up_prob = row["Up_Probability"]
        holding = ticker in user_holdings

        if pd.isna(up_prob):
            return "데이터 없음"

        if holding and up_prob <= SELL_THRESHOLD:
            return "매도 검토"

        if not holding and up_prob >= BUY_THRESHOLD:
            return "매수 검토"

        if holding:
            return "보유"

        return "관망"

    signals["Signal"] = signals.apply(
        dynamic_signal,
        axis=1
    )

    signals["보유여부"] = signals["Ticker"].isin(
        user_holdings
    )

    # =====================================================
    # 6. 상단 KPI
    # =====================================================
    buy_candidates = signals[
        signals["Signal"] == "매수 검토"
    ]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "분석 완료 종목",
        f"{len(signals):,}개"
    )

    col2.metric(
        "매수 검토 종목",
        f"{len(buy_candidates):,}개"
    )

    latest_date = (
        signals["Date"].iloc[0]
        if not signals.empty
        else "-"
    )

    col3.metric(
        "기준 일자",
        str(latest_date)
    )

    if not signals.empty:
        top_stock = signals.sort_values(
            "Up_Probability",
            ascending=False
        ).iloc[0]

        top_stock_name = (
            f"{top_stock['Name']} "
            f"({top_stock['상승확률(%)']}%)"
        )
    else:
        top_stock_name = "-"

    col4.metric(
        "최고 상승 확률 종목",
        top_stock_name
    )

    st.divider()

    # =====================================================
    # 7. 탭
    # =====================================================
    tab1, tab2, tab3 = st.tabs(
        [
            "🔥 상승 유망 종목",
            "📊 모의투자 성과",
            "🔍 전체 종목 검색"
        ]
    )

    # =====================================================
    # 탭 1: 상승 유망 종목
    # =====================================================
    with tab1:

        st.subheader("AI 상승 확률 종목")

        f_col1, f_col2 = st.columns(2)

        with f_col1:
            market_options = sorted(
                signals["Market"].dropna().unique().tolist()
            )

            market_filter = st.multiselect(
                "시장 선택",
                options=market_options,
                default=market_options
            )

        with f_col2:
            signal_options = [
                "매수 검토",
                "보유",
                "관망",
                "매도 검토",
                "데이터 없음"
            ]

            signal_filter = st.multiselect(
                "AI 시그널",
                options=signal_options,
                default=[
                    "매수 검토",
                    "보유",
                    "관망",
                    "매도 검토"
                ]
            )

        filtered = signals[
            signals["Market"].isin(market_filter)
            & signals["Signal"].isin(signal_filter)
        ].copy()

        filtered = filtered.sort_values(
            "Up_Probability",
            ascending=False
        ).head(30)

        display_cols = [
            "Ticker",
            "Name",
            "Market",
            "Date",
            "Close",
            "상승확률(%)",
            "예상 상승률(%)",
            "Signal"
        ]

        st.write(
            f"표시 종목: {len(filtered)}개"
        )

        if filtered.empty:
            st.warning(
                "선택한 시장 및 시그널 조건에 해당하는 종목이 없습니다."
            )
        else:
            st.dataframe(
                filtered[display_cols].rename(
                    columns={
                        "Ticker": "종목코드",
                        "Name": "종목명",
                        "Market": "시장",
                        "Date": "기준일",
                        "Close": "종가",
                        "상승확률(%)": "상승확률(%)",
                        "예상 상승률(%)": "예상 상승률(%)",
                        "Signal": "AI 시그널"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )

    # =====================================================
    # 탭 2: 모의투자 성과
    # =====================================================
    with tab2:

        if CURVE_FILE.exists() and METRICS_FILE.exists():

            curve_df = pd.read_csv(CURVE_FILE)

            metrics_df = pd.read_csv(
                METRICS_FILE,
                dtype={"Ticker": str}
            )

            if not curve_df.empty and "Equity" in curve_df.columns:

                final_equity = curve_df["Equity"].iloc[-1]

                total_ret = (
                    final_equity / INITIAL_CASH - 1
                ) * 100

                st.subheader(
                    "포트폴리오 누적 자산 "
                    f"(수익률: {total_ret:+.2f}%)"
                )

                st.line_chart(
                    curve_df.set_index("Date")["Equity"]
                )

            if not metrics_df.empty:

                metrics_df["AI 수익률(%)"] = (
                    metrics_df["AI_Return"] * 100
                ).round(2)

                metrics_df["단순보유 수익률(%)"] = (
                    metrics_df["BuyHold_Return"] * 100
                ).round(2)

                metrics_df["최대낙폭 MDD(%)"] = (
                    metrics_df["MDD"] * 100
                ).round(2)

                metrics_df["정확도(%)"] = (
                    metrics_df["Accuracy"] * 100
                ).round(2)

                top_metrics = metrics_df.sort_values(
                    "AI_Return",
                    ascending=False
                ).head(20)

                st.subheader("백테스트 수익률 상위 종목")

                st.dataframe(
                    top_metrics[
                        [
                            "Ticker",
                            "Name",
                            "Market",
                            "AI 수익률(%)",
                            "단순보유 수익률(%)",
                            "최대낙폭 MDD(%)",
                            "정확도(%)"
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

        else:
            st.info(
                "백테스트 결과 파일이 아직 없습니다."
            )

    # =====================================================
    # 탭 3: 전체 종목 검색
    # =====================================================
    with tab3:

        st.subheader("전체 분석 종목 검색")

        search_query = st.text_input(
            "종목명 또는 티커 검색",
            placeholder="예: 삼성전자, 005930, AAPL, MSFT"
        )

        search_market = st.selectbox(
            "검색 시장",
            options=["전체"] + sorted(
                signals["Market"].dropna().unique().tolist()
            )
        )

        res = signals.copy()

        if search_market != "전체":
            res = res[
                res["Market"] == search_market
            ]

        if search_query.strip():

            query = search_query.strip()

            res = res[
                res["Name"].str.contains(
                    query,
                    case=False,
                    na=False
                )
                |
                res["Ticker"].str.contains(
                    query,
                    case=False,
                    na=False
                )
            ]

        st.write(
            f"검색 결과: {len(res)}개"
        )

        st.dataframe(
            res[
                [
                    "Ticker",
                    "Name",
                    "Market",
                    "Date",
                    "Close",
                    "상승확률(%)",
                    "예상 상승률(%)",
                    "Signal"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# 8. 실행
# =========================================================
if __name__ == "__main__":
    run_streamlit_app()
