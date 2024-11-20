import streamlit as st
import torch


st.title("Torch Example")
# st.write(
#     "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
# )

x = torch.randn(size=(10, 10))
st.write(x)