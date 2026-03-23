
import streamlit as st
import requests

st.title("FinSight")
st.caption("Ask questions about SEC financial filings")

question = st.text_input("Your question", placeholder="What was Apple's revenue in 2024?")

if st.button("Ask") and question:
    with st.spinner("Searching filings..."):
        response = requests.post(
            "http://localhost:8000/ask",
            json={"text": question}
        )
        data = response.json()

    st.markdown("### Answer")
    st.write(data["answer"])

    st.markdown("### Sources")
    for i, chunk in enumerate(data["sources"]):
        with st.expander(f"Source {i+1}"):
            st.write(chunk)