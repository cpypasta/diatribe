import os, base64, json
import streamlit as st
from hume import HumeClient
from hume.tts.types import VoiceProvider
from hume.core.api_error import ApiError
from hume.tts import PostedUtterance, PostedUtteranceVoiceWithName, ReturnGeneration
from enum import Enum
from typing import List, Dict
from diatribe.audio_providers.audio_provider import AudioProvider
from diatribe.utils import get_env_key
from dataclasses import dataclass

@dataclass
class AIVoice:
    name: str
    id: str
    gender: str | None
    accent: str | None
    provider: VoiceProvider
    versions: List[str] | None

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
                voice.tags["GENDER"][0] if "GENDER" in voice.tags else None, 
                voice.tags["ACCENT"][0] if "ACCENT" in voice.tags else None,
                voice.provider,
                voice.compatible_octave_models
            )
            for voice in all_hume_voices
        ]
        return ai_voices
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
                    voice=PostedUtteranceVoiceWithName(name=voice.name, provider=voice.provider)
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
    def __init__(self):
        try:
            self.voices = get_voices()
        except:
            st.toast("Unable to fetch Hume voices", icon="👎")
            self.voices = []
        self.voice_names = sorted([v.name for v in self.voices])

    def get_voice_from_name(self, name: str) -> AIVoice | None:
        return next((voice for voice in self.voices if voice.name == name), None)

    def get_voice_names(self) -> List[str]:
        return self.voice_names
    
    def get_voice_id(self, name) -> str:
        voice = self.get_voice_from_name(name)
        if voice is None:
            raise Exception(f"Voice ID not found for voice name: {name}")
        return name
    
    def define_creds(self) -> None:
        hume_key_value = get_env_key("HUME_API_KEY", "hume_key_value")
        hume_key = st.text_input("Hume API Key", hume_key_value, type="password", key="hume_key")        
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
        voice_genders = set([self.voice.gender for self.voice in self.voices if self.voice.gender])
        voice_gender_selected = st.selectbox("Gender Filter", voice_genders, index=None)

        voice_accents = sorted(set([voice.accent for voice in self.voices if voice.accent]))
        voice_accent_selected = st.selectbox("Accent Filter", voice_accents, index=None)

        if voice_gender_selected:
            filtered_voices = [voice for voice in self.voices if voice.gender == voice_gender_selected]
        else:
            filtered_voices = self.voices

        if voice_accent_selected:
            filtered_voices = [voice for voice in filtered_voices if voice.accent == voice_accent_selected]

        filtered_voices = [voice for voice in filtered_voices if voice.versions and self.version in voice.versions]

        filtered_voice_names = sorted([voice.name for voice in filtered_voices])
        voice_selected = st.selectbox("Speaker", filtered_voice_names)  
        
        voice_id = None
        if voice_selected:
            try:
                voice_id = self.get_voice_id(voice_selected)
            except:
                st.toast(f"Unable to find voice ID for {voice_selected}")

            sample_path = f"./samples/{voice_id}.wav"
            if os.path.exists(sample_path):
                st.audio(sample_path, format="audio/wav")

        return {
            "voice_id": voice_id
        }
    
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
            self.get_voice_from_name(voice_id), 
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