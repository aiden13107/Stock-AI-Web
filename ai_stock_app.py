import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
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
# 2. Streamlit 웹 대시보드 UI (조회 전용)
# =========================================================

def run_streamlit_app():
    st.set_page_config(
        page_title="주식 예측 대시보드",
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 상승 유망 종목")
    st.caption("KRX 일별 시세 및 기술적 지표(RandomForest)를 학습한 결과 기반 조회 전용 대시보드입니다.")

    # 사이드바: 정보 안내 (학습 버튼 제거됨)
    with st.sidebar:
        st.header("📌 앱 정보")
        st.info("이 앱은 조회 전용 모드로 동작합니다.\n\n최신 데이터 분석 및 학습은 로컬 PC에서 `--analyze` 명령어로 진행 후 결과 파일을 업로드해 주세요.")
        
        st.divider()
        st.markdown("**모델 사양**")
        st.code("모델: RandomForest\n매수 기준: >= 55.0%\n매도 기준: <= 45.0%")

    # 결과 파일 존재 여부 확인
    if not SIGNALS_FILE.exists():
        st.warning("⚠️ 아직 생성된 분석 결과 파일(`latest_signals.csv`)이 없습니다.")
        st.markdown("""
        ### 💡 해결 방법
        1. 내 PC에서 분석 스크립트를 실행해 결과를 생성해 주세요:
           ```bash
           python ai_stock_app.py --analyze
           ```
        2. 생성된 `multi_stock_results/results/` 폴더 안의 핵심 CSV 파일들을 깃허브 레포지토리의 동일한 경로에 업로드해 주세요.
        """)
        return

    # 데이터 로드
    signals = pd.read_csv(SIGNALS_FILE, dtype={"Ticker": str})
    signals["Ticker"] = signals["Ticker"].str.zfill(6)
    signals["상승확률(%)"] = (signals["Up_Probability"] * 100).round(2)

    # 상단 요약 KPI 카드
    col1, col2, col3, col4 = st.columns(4)
    buy_candidates = signals[signals["Signal"] == "매수 검토"]

    col1.metric("분석 완료 종목", f"{len(signals):,}개")
    col2.metric("내일 매수 검토 추천", f"{len(buy_candidates):,}개")
    col3.metric("기준 일자", str(signals["Date"].iloc[0]) if not signals.empty else "-")
    top_stock_name = f"{signals.iloc[0]['Name']} ({signals.iloc[0]['상승확률(%)']}%)" if not signals.empty else "-"
    col4.metric("최고 상승 확률 종목", top_stock_name)

    st.divider()

    # 탭 구성
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
                "Ticker": "종목코드",
                "Name": "종목명",
                "Market": "시장",
                "Date": "기준일",
                "Close": "현재가(원)",
                "Signal": "AI 시그널"
            }),
            width="stretch",
            hide_index=True
        )

    with tab2:
        if CURVE_FILE.exists() and METRICS_FILE.exists():
            curve_df = pd.read_csv(CURVE_FILE)
            metrics_df = pd.read_csv(METRICS_FILE, dtype={"Ticker": str})

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
            st.info("백테스트 결과 파일(`portfolio_curve.csv` 또는 `backtest_by_stock.csv`)이 아직 없습니다.")

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
# 3. 실행 진입점
# =========================================================
if __name__ == "__main__":
    run_streamlit_app()
