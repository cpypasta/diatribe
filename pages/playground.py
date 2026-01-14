import streamlit as st
from diatribe.sidebar import select_audio_provider

st.title("🎧 Diatribe Playground")
st.markdown("## Sound Engine")
with st.container(border=True):
    audio_provider = select_audio_provider()
    if audio_provider:
        audio_provider.define_creds()
        with st.expander("Options"):
            audio_provider_options = audio_provider.define_options() 
        with st.expander("Usage"):
            audio_provider.define_usage()

if audio_provider:  
    st.markdown("## Voice")
    with st.container(border=True): 
        voice = audio_provider.define_voice_explorer()
        if "voice_id" in voice:
            voice_id = voice['voice_id']

    if voice_id:
        audio_file = None
        st.markdown("## Text")
        with st.container(border=True):
            text_to_speak = st.text_area("Text to Speak", height=150)
            if text_to_speak:
                generate_audio = st.button("Generate Audio", key="generate_audio_button")
                if generate_audio:
                    if audio_provider_options:
                        audio_provider_options["test"] = True
                    else:
                        audio_provider_options = {"test": True}
                    audio_file = audio_provider.generate_and_save(
                        text_to_speak, 
                        voice_id, 
                        line=0, 
                        options=audio_provider_options
                    )

        if audio_file:
            st.markdown("## Audio")
            st.audio(audio_file, format="audio/wav")

    