
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(__file__))

from datagraph.app import main as datagraph_main
from lrml.app import main as lrml_main

st.set_page_config(page_title="Data Explorer and ML Modeler", layout="wide")

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Data Explorer", "ML Modeler"])

if selection == "Data Explorer":
    datagraph_main()
elif selection == "ML Modeler":
    lrml_main()
