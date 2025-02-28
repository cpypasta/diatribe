import os
import streamlit as st
from elevenlabs import Voice, set_api_key
from dataclasses import dataclass
from openai import OpenAI
from streamlit_js_eval import streamlit_js_eval
from diatribe.audio_providers.hume_provider import HumeProvider
from diatribe.audio_providers.el_provider import ElevenLabsProvider
from diatribe.audio_providers.audio_provider import AudioProvider
from typing import Dict

@dataclass
class SidebarData:
  ready: bool
  el_key: str
  hume_key: str  
  audio_provider: AudioProvider
  audio_provider_options: Dict
  voice_names: list[str]
  enable_instructions: bool
  enable_audio_editing: bool
  openai_api_key: str
  openai_model: str
  openai_temp: float
  openai_max_tokens: int

@st.cache_data
def get_models(openai_api_key: str) -> list[str]:
  """Get a list of OpenAI models."""
  client = OpenAI(api_key=openai_api_key)
  response = client.models.list()
  model_ids = [m.id for m in response.data]
  return sorted(model_ids)


def get_env_key(os_name: str, session_name: str) -> str:
    if os.getenv(os_name):
      value = os.getenv(os_name)
    elif session_name in st.session_state:
      value = st.session_state[session_name]
    else:
      value = ""  
    return value

def create_sidebar() -> SidebarData:
  """Create the streamlit sidebar."""
  with st.sidebar:    
    with st.expander("Sound Engine", expanded=True):
      audio_provider = None
      el_key = None
      hume_key = None
      sound_provider = st.selectbox("Provider", ["ElevenLabs", "Hume AI"])
      if sound_provider == "ElevenLabs":
        el_key_value = get_env_key("ELEVENLABS_API_KEY", "el_key_value")
        el_key = st.text_input("ElevenLabs API Key", el_key_value, type="password", key="el_key")  
        if el_key:
          st.session_state["el_key_value"] = el_key  
          set_api_key(el_key)    
          audio_provider = ElevenLabsProvider()
      elif sound_provider == "Hume AI":    
        hume_key_value = get_env_key("HUME_API_KEY", "hume_key_value")
        hume_key = st.text_input("Hume API Key", hume_key_value, type="password", key="hume_key")        
        if hume_key:
          st.session_state["hume_key_value"] = hume_key             
          audio_provider = HumeProvider()

      audio_provider.define_usage()
    
    if audio_provider:        
      with st.expander("Sound Options"):
        audio_provider_options = audio_provider.define_options()                                  
                    
      with st.expander("Voice Explorer"):
        audio_provider.define_voice_explorer()                    
    
      with st.expander("OpenAI Options"):
        if os.getenv("OPENAI_API_KEY"):
          openai_key_value = os.getenv("OPENAI_API_KEY")
        elif "openai_key_value" in st.session_state:
          openai_key_value = st.session_state["openai_key_value"]
        else:
          openai_key_value = ""        

        openai_api_key = st.text_input("API Key _(optional)_", openai_key_value, type="password", key="openai_key")
        if openai_api_key:
          st.session_state["openai_key_value"] = openai_api_key
          openai_models = get_models(openai_api_key)
          try:
            gpt4_index = openai_models.index("gpt-3.5-turbo-16k")
          except:
            gpt4_index = 0
          openai_model = st.selectbox("Model", openai_models, index=gpt4_index)
          openai_temp = st.slider("Temperature", 0.0, 1.5, 1.3, 0.1,  help="The higher the temperature, the more creative the text.")
          openai_max_tokens = st.slider("Max Tokens", 1024, 10000, 3072, 1024, help="Check the official documentation on maximum token size for the selected model.")
        else:
          openai_model = None
          openai_temp = None
          openai_max_tokens = None          
      
      with st.expander("View Options"):          
        show_instructions = st.toggle(
          "Enable Instructions",
          value=False,
          help="Once you get familar with the app you can turn this off."
        )
        edit_audio = st.toggle(
          "Enable Audio Editing", 
          value=True,
          help="Enable audio editing for each dialogue line. This is disabled by default to increase performance."
        )

      clear_dialogue = st.button("Clear Dialogue", help=":warning: Clear everything and start over. :warning:", use_container_width=True)
      if clear_dialogue:
        streamlit_js_eval(js_expressions="parent.window.location.reload()")
        
      return SidebarData( 
        ready=el_key or hume_key,
        el_key=el_key,
        hume_key=hume_key,
        audio_provider=audio_provider,
        audio_provider_options=audio_provider_options,
        voice_names=audio_provider.get_voice_names(),
        enable_instructions=show_instructions,
        enable_audio_editing=edit_audio,        
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_temp=openai_temp,
        openai_max_tokens=openai_max_tokens
      )
    else:
      return SidebarData(
        ready=False,
        el_key="",
        hume_key="",
        audio_provider=None,
        audio_provider_options={},
        voice_names=[],
        enable_instructions=True,
        enable_audio_editing=False,
        openai_api_key="",
        openai_model="",
        openai_temp=1.5,
        openai_max_tokens=4096
      )   

  