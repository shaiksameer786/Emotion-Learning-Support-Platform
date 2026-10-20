"""Entry point for the demonstration-only Streamlit application shell."""

import streamlit as st

from els_platform.presentation import render_app


st.set_page_config(
    page_title="Learning Support Prototype",
    page_icon="📘",
    layout="wide",
)
render_app()
