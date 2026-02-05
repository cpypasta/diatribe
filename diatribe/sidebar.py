import streamlit as st
from dataclasses import dataclass
from openai import OpenAI
from streamlit_js_eval import streamlit_js_eval
from diatribe.audio_providers.audio_provider import AudioProvider
from diatribe.audio_providers.hume_provider import HumeProvider
from diatribe.audio_providers.el_provider import ElevenLabsProvider
from diatribe.audio_providers.openai_provider import OpenAIProvider
from diatribe.audio_providers.playai_provider import PlayAIProvider
from diatribe.audio_providers.kokoro_provider import KokoroProvider
from diatribe.audio_providers.xtts_provider import XttsProvider
from diatribe.audio_providers.piper_provider import PiperProvider
from diatribe.audio_providers.parler_provider import ParlerProvider
from diatribe.audio_providers.chatterbox_provider import ChatterboxProvider
from typing import Dict
from diatribe.utils import get_env_key
from diatribe.audio_providers.audio_provider import Location

@dataclass
class SidebarData:
  ready: bool
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
def get_openai_models(openai_api_key: str) -> list[str]:
  """Get a list of OpenAI models."""
  client = OpenAI(api_key=openai_api_key)
  response = client.models.list()
  model_ids = [m.id for m in response.data]
  return sorted(model_ids)

def select_audio_provider(location: Location | None = None) -> AudioProvider | None:
    providers = [
      KokoroProvider(),
      XttsProvider(),
      PiperProvider(),
      ParlerProvider(),
      OpenAIProvider(),
      ElevenLabsProvider(),
      HumeProvider(),
      ChatterboxProvider(),
    ]
    if location:
      providers = [provider for provider in providers if provider.location == location]

    provider_names = [provider.name for provider in providers]
    sound_provider = st.selectbox("Engine", provider_names, index=0)
    audio_provider = None
    if sound_provider:
      select_index = provider_names.index(sound_provider)
      audio_provider = providers[select_index]
    return audio_provider

def create_sidebar() -> SidebarData:
  """Create the streamlit sidebar."""
  with st.sidebar:    
    with st.expander("Sound Engine", expanded=True):
      audio_provider = select_audio_provider()        
    
    if audio_provider:  
      audio_provider.define_creds()
      if audio_provider.has_usage:        
        audio_provider.define_usage()            
      with st.expander("Sound Options"):
        audio_provider_options = audio_provider.define_options()                                  
                    
      with st.expander("Voice Explorer"):
        audio_provider.define_voice_explorer()                    
    
      with st.expander("Generative AI Options"):
        openai_key_value =  get_env_key("OPENAI_API_KEY", "openai_key_value")   
        openai_api_key = st.text_input("API Key _(optional)_", openai_key_value, type="password", key="openai_key")

        if openai_api_key:
          st.session_state["openai_key_value"] = openai_api_key
          openai_models = get_openai_models(openai_api_key)
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

      clear_dialogue = st.button("Clear Dialogue", help=":warning: Clear everything and start over. :warning:", width='stretch')
      if clear_dialogue:
        streamlit_js_eval(js_expressions="parent.window.location.reload()")
        
      return SidebarData( 
        ready=bool(audio_provider),
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

  