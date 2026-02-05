import streamlit as st
from diatribe.sidebar import select_audio_provider
from diatribe.audio_providers.audio_provider import Location

st.title("🎧 Diatribe Playground")
st.text("Session: " + st.session_state.session_id)

st.markdown("## Sound Engine")
with st.container(border=True):
    location_selection = st.selectbox("Model Location", [location.value for location in Location], index=None)
    if location_selection:
        location = Location(location_selection)
    else:
        location = None
    audio_provider = select_audio_provider(location=location)
    if audio_provider:
        st.caption(audio_provider.description)
        audio_provider.define_creds()
        with st.expander("Options"):
            audio_provider_options = audio_provider.define_options() 
        if audio_provider.has_usage:        
            with st.expander("Usage"):
                audio_provider.define_usage()

if audio_provider:  
    voice_id = None
    st.markdown("## Voice")
    with st.container(border=True): 
        voice = audio_provider.define_voice_explorer()
        if "voice_id" in voice:
            voice_id = voice['voice_id']

    if voice_id:
        audio_file = None
        st.markdown("## Text")
        with st.container(border=True):
            if audio_provider.supports_instructions:
                instructions = st.text_input("Instructions", help="Instruct the voice how you want them to speak")
            else:
                instructions = None

            text_to_speak = st.text_area("Text to Speak", height=150)
            if text_to_speak:
                generate_audio = st.button("Generate Audio", key="generate_audio_button", width="stretch")
                if generate_audio:
                    if audio_provider_options:
                        audio_provider_options["test"] = True
                    else:
                        audio_provider_options = {"test": True}
                    try:
                        audio_file = audio_provider.generate_and_save(
                            text_to_speak, 
                            voice_id, 
                            line=0, 
                            options=audio_provider_options,
                            guidance=instructions
                        )
                    except Exception as e:
                        st.toast(f"Error generating audio: {e}", icon="👎")

        if audio_file:
            st.markdown("## Audio")
            st.audio(audio_file, format="audio/wav")

            with open(audio_file, "rb") as audio_bytes:
                st.download_button(
                    label="Download",
                    data=audio_bytes, 
                    file_name="tts_audio.wav", 
                    mime="audio/wav",
                    width='stretch',
                    type="primary"
                )            

    