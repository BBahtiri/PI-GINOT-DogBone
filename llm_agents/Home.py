#!/usr/bin/env python3
"""PI-GINOT Agentic Studio — Streamlit entry point."""

import streamlit as st
from pathlib import Path

st.set_page_config(page_title="PI-GINOT Studio", page_icon="🧬", layout="wide")

_PAGE_DIR = Path(__file__).parent / "pages"

pages = [
    st.Page("pages/1_Analyst.py", title="Analyst", default=True),
    st.Page("pages/2_Design_Studio.py", title="Design Studio", icon="🎨"),
]

if (_PAGE_DIR / "3_Demo.py").exists():
    pages.append(st.Page("pages/3_Demo.py", title="Demo", icon="🎬"))

nav = st.navigation(pages)
nav.run()
