import streamlit as st
import webbrowser

RegURL = 'https://automlv2-6dn5itbyuz35jutyjbmei6.streamlit.app/'
ClassURL = 'https://automlv2-j6xbezlcu7lfjdenpxwbka.streamlit.app/'
DimURL = 'https://automlv2-7u3epmitkv5vhrshzdtwoq.streamlit.app/'

st.set_page_config(page_title="Auto ML v2")

st.write("""
# Auto ML Tool
""")


if st.button('Regression'):
    webbrowser.open_new_tab(RegURL)

if st.button('Classification'):
    webbrowser.open_new_tab(ClassURL)

if st.button('Dimensionality Reduction'):
    webbrowser.open_new_tab(DimURL)