import os, base64
import streamlit as st
from hume import HumeClient
from hume.tts import PostedUtterance, PostedUtteranceVoiceWithName, ReturnGeneration
from enum import Enum
from typing import List, Dict
from diatribe.audio_providers.audio_provider import AudioProvider
from diatribe.utils import get_env_key

class HumeVoice(Enum):
    WISE_WIZARD = PostedUtteranceVoiceWithName(name="wise wizard", provider="HUME_AI")
    NATURE_DOCUMENTARY = PostedUtteranceVoiceWithName(name="name documentary narrator", provider="HUME_AI")
    TAVERN_TRAVELLER = PostedUtteranceVoiceWithName(name="tavern traveller", provider="CUSTOM_VOICE")
    
def get_voice_from_name(name: str) -> HumeVoice:
    return next((voice for voice in HumeVoice if voice.value.name == name), None)            
    
def generate(
    text: str, 
    voice: HumeVoice, 
    guidance: str = None,
    api_key: str = None
) -> ReturnGeneration:
    client = HumeClient(api_key=api_key)      
    speech = client.tts.synthesize_json(
        utterances=[
            PostedUtterance(
                text=text,
                description=guidance,
                voice=voice.value
            )
        ]
    )
    return speech.generations[0]    
    
class HumeProvider(AudioProvider):
    def get_voice_names(self) -> List[str]:
        voices = [voice for voice in HumeVoice]
        return [voice.value.name for voice in voices]
    
    def get_voice_id(self, name) -> str:
        voice = get_voice_from_name(name)
        if voice is None:
            raise Exception(f"Voice ID not found for voice name: {name}")
        return name
    
    def define_creds(self) -> None:
        hume_key_value = get_env_key("HUME_API_KEY", "hume_key_value")
        hume_key = st.text_input("Hume API Key", hume_key_value, type="password", key="hume_key")        
        if hume_key:
          st.session_state["hume_key_value"] = hume_key 

    def define_options(self) -> Dict:
        st.write("Hume supports speech instructions, so add descriptions to the character and/or to each dialogue line. These descriptions will be provided to Hume.")

        return {
            "api_key": os.getenv("HUME_API_KEY")
        }
        
    def define_voice_explorer(self) -> Dict:
        return {}
    
    def define_usage(self):
        return None
    
    def generate_and_save(
        self,
        text: str,
        voice_id: str,
        line: int,
        options: Dict,
        guidance: str = None
    ) -> str:
        """Generate audio from a dialogue and save it to a file."""
        speech = generate(text, get_voice_from_name(voice_id), api_key=options["api_key"], guidance=guidance)
        audio_data = base64.b64decode(speech.audio)
        audio_file = f"./session/{st.session_state.session_id}/audio/line{line}.wav"
        os.makedirs(os.path.dirname(audio_file), exist_ok=True)
        with open(audio_file, "wb") as f:
            f.write(audio_data)  
        return audio_file    