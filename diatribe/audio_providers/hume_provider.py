import os, base64, json
import streamlit as st
from hume import HumeClient
from hume.core.api_error import ApiError
from hume.tts import PostedUtterance, PostedUtteranceVoiceWithName, ReturnGeneration
from enum import Enum
from typing import List, Dict
from diatribe.audio_providers.audio_provider import AudioProvider, Location
from diatribe.utils import get_env_key
from diatribe.data import AIVoice, Gender, Source

@st.cache_data
def get_voices() -> List[AIVoice]:
    try:
        client = HumeClient(api_key=os.getenv("HUME_API_KEY"))
        hume_response = client.tts.voices.list(provider="HUME_AI")
        hume_voices = list(hume_response)
        custom_response = client.tts.voices.list(provider="CUSTOM_VOICE")
        custom_voices = list(custom_response)
        all_hume_voices = hume_voices + custom_voices
        ai_voices = [
            AIVoice(
                voice.name, 
                voice.id, 
                gender=Gender(voice.tags["GENDER"][0]) if "GENDER" in voice.tags else None, 
                accent=voice.tags["ACCENT"][0] if "ACCENT" in voice.tags else None,
                source=Source.CUSTOM if voice.provider == "CUSTOM_VOICE" else Source.PROVIDED,
                models=voice.compatible_octave_models if voice.compatible_octave_models else []
            )
            for voice in all_hume_voices
        ]
        return ai_voices
    except ApiError as e:
        if e.status_code == 429:
            raise Exception("You have reached your Hume API rate limit. Please try again later.")
        else:
            raise Exception(f"Hume API error: {e.body}")     
    except Exception as e:
        print(e)
        raise Exception("Failed to get Hume voices")            
    
def generate(
    text: str, 
    voice: AIVoice | None, 
    version: str,
    guidance: str | None = None,
    api_key: str | None = None
) -> ReturnGeneration:
    if voice is None:
        raise Exception("Voice not found.")

    client = HumeClient(api_key=api_key)   
    try:   
        speech = client.tts.synthesize_json(
            utterances=[
                PostedUtterance(
                    text=text,
                    description=guidance,
                    voice=PostedUtteranceVoiceWithName(
                        name=voice.name, 
                        provider="CUSTOM_VOICE" if voice.source == Source.CUSTOM else "HUME_AI"
                    )
                )
            ],
            version=version
        )
        return speech.generations[0]  
    except ApiError as e:
        if e.status_code == 429:
            raise Exception("You have reached your Hume API rate limit. Please try again later.")
        else:
            raise Exception(f"Hume API error: {e.body}")  
    
class HumeProvider(AudioProvider):
    @property
    def name(self) -> str:
        return "Hume AI"

    @property
    def description(self) -> str:
        return "Hume AI offers natural and emotive speech with likely billions of parameters."

    @property
    def voices(self) -> List[AIVoice]:
        if not hasattr(self, "hume_voices"):
            try:
                self.hume_voices = get_voices()
            except Exception as ex:
                st.toast(ex, icon="👎")
                self.hume_voices = []                
        return self.hume_voices
    
    @property
    def location(self) -> Location:
        return Location.HOSTED

    def get_voice_id(self, name: str) -> str:
        return super().get_voice_id(name)
    
    def get_voice_names(self) -> List[str]:
        if not hasattr(self, "voice_names"):
            self.voice_names = sorted([v.name for v in self.voices])
        return self.voice_names

    def define_creds(self) -> None:
        hume_key_value = get_env_key("HUME_API_KEY", "hume_key_value")
        hume_key = st.text_input("API Key", hume_key_value, type="password", key="hume_key")        
        if hume_key:
          st.session_state["hume_key_value"] = hume_key 

    def define_options(self) -> Dict:
        model_name = st.selectbox("Speech Model", ["Octave 1", "Octave 2"], index=0)
        if model_name == "Octave 1":
            version = "1"
        else:
            version = "2"
        self.version = version
        return {
            "api_key": os.getenv("HUME_API_KEY"),
            "version": version
        }
        
    def define_voice_explorer(self) -> Dict:
        return self._show_voices(["gender", "accent"], sample_path="hume", model=self.version)
    
    def define_usage(self):
        return None
    
    def generate_and_save(
        self,
        text: str,
        voice_id: str,
        line: int,
        options: Dict,
        guidance: str | None = None
    ) -> str:
        """Generate audio from a dialogue and save it to a file."""
        speech = generate(
            text, 
            self._get_voice_by_id(voice_id), 
            options["version"], 
            api_key=options["api_key"], 
            guidance=guidance
        )        
        audio_data = base64.b64decode(speech.audio)

        if "test" in options:
            audio_file = f"./session/{st.session_state.session_id}/temp/test.wav"
        else:
            audio_file = f"./session/{st.session_state.session_id}/audio/line{line}.wav"

        os.makedirs(os.path.dirname(audio_file), exist_ok=True)
        with open(audio_file, "wb") as f:
            f.write(audio_data)  
        return audio_file    