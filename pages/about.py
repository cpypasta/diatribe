import streamlit as st
  
st.set_page_config(layout="wide", page_title="Diatribe About", page_icon="🎧")

st.title(":sparkles: Diatribe About")

st.markdown("""
    ### Diatribe is a tool to create interesting audio dialogues.
""")

st.markdown("""
    The motivation for creating Diatribe was to create a tool for my family. We were having so much
    fun with our cloned voices. At first, I created everything in ElevenLabs and combined and edited
    the dialogue lines in a desktop audio editor. But as family demand for more dialogues continued to grow and
    being the lazy programmer that I am, I decided to create a tool to automate the process. The name `Diatribe`
    was given since the dialogues created tended to be silly and long-winded.
""")

st.info(":bulb: It is worth noting that creating a tool like Diatribe using `Streamlit` has been interesting. While the simplicity of `Streamlit` makes it easy to get started, it can be challenging to build a larger application. Some of the quirks of Diatribe are due to the limitations of `Streamlit`. One thing is for sure, the app definitely feels like a `Streamlit` app.")