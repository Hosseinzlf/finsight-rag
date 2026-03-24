import os

import requests
import streamlit as st


st.set_page_config(page_title="FinSight", page_icon=":chart_with_upwards_trend:", layout="wide")
st.title("FinSight: Financial Filing Q&A")
st.caption("Ask questions about indexed filing chunks through the FastAPI backend.")

default_api_url = os.getenv("FINSIGHT_API_URL", "http://127.0.0.1:8000")
api_url = st.sidebar.text_input("API Base URL", value=default_api_url).rstrip("/")
available_tickers = ["ALL", "AAPL", "MSFT", "TSLA", "AMZN"]
selected_ticker = st.sidebar.selectbox("Ticker", options=available_tickers, index=0)

question = st.text_area(
    "Question",
    placeholder="What are Apple's main risk factors?",
    height=120,
)

if st.button("Ask", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        payload = {"text": question.strip()}
        if selected_ticker != "ALL":
            payload["ticker"] = selected_ticker

        try:
            with st.spinner("Querying FinSight..."):
                response = requests.post(f"{api_url}/ask", json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to the API. Start backend with: "
                "`uvicorn src.api.main:app --reload`"
            )
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")
        else:
            st.subheader("Answer")
            st.write(data.get("answer", "No answer returned."))

            tickers = data.get("tickers_searched") or []
            if tickers:
                st.caption(f"Tickers searched: {', '.join(tickers)}")

            sources = data.get("sources") or []
            if sources:
                st.subheader("Sources")
                source = sources[0]
                label = source.get("source", "Unknown source")
                source_ticker = source.get("ticker", "UNKNOWN")
                company = source.get("company", "Unknown")
                text = source.get("text", "")

                with st.expander(f"1. {label} ({source_ticker} - {company})"):
                    st.write(text)