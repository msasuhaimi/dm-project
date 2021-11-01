#main.py
import syazwanEDA
import syazwanRegress
import wengken
import hidayatClassify
import streamlit as st
PAGES = {
    "Data Exploration": syazwanEDA,
    "Regression": syazwanRegress,
    "Clustering": wengken,
    "Classification": hidayatClassify
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()