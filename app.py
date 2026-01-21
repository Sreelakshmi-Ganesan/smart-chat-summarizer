import streamlit as st
import pandas as pd
from transformers import pipeline

from utils_chat import (
    parse_upload_to_df,
    preprocess_df,
    build_daily_summaries,
    build_tasks_df,
)

@st.cache_resource
def get_summarizer():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        tokenizer="sshleifer/distilbart-cnn-12-6",
        device=-1,  # CPU
    )

summarizer = get_summarizer()

st.title("Smart Chat Summarizer – WhatsApp")

uploaded_file = st.file_uploader("Upload WhatsApp .txt export", type=["txt"])

if uploaded_file is None:
    st.info("Upload a WhatsApp chat export (.txt) to see summaries and tasks.")
else:
    st.write(f"Received file: **{uploaded_file.name}**")

    with st.spinner("Processing chat..."):
        df = parse_upload_to_df(uploaded_file.read())
        df_nlp = preprocess_df(df)
        daily_summaries_df = build_daily_summaries(df_nlp, summarizer)
        tasks_df = build_tasks_df(df_nlp)

    st.success("Processing complete.")
    st.write("Daily summaries preview:")
    st.dataframe(daily_summaries_df.head())

    st.write("Tasks preview:")
    st.dataframe(tasks_df.head())
