# -*- coding: utf-8 -*-
import streamlit as st

st.title("streamlit Demo")
st.subheader("please enter the details below")

with st.form("inputfield", clear_on_submit=True):
    name = st.text_input("username")
    symptom = st.text_area("enter symptoms")

    submit = st.form_submit_button("submit this form")
