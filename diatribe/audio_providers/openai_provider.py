import os, streamlit as st
from diatribe.audio_providers.audio_provider import AudioProvider, Location
from diatribe.data import AIVoice, Gender
from typing import List, Dict
from enum import Enum
from openai import OpenAI
from dataclasses import dataclass
from diatribe.utils import get_env_key

all_models = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
latest_models = ["gpt-4o-mini-tts"]
class OpenAIVoice(Enum):
    ALLOY = AIVoice("Alloy", "alloy", gender=Gender.FEMALE, models=all_models)
    ASH = AIVoice("Ash", "ash", gender=Gender.MALE, models=all_models)    
    BALLAD = AIVoice("Ballad", "ballad", gender=Gender.MALE, models=latest_models)
    CORAL = AIVoice("Coral", "coral", gender=Gender.FEMALE, models=all_models)
    ECHO = AIVoice("Echo", "echo", gender=Gender.MALE, models=all_models)
    FABLE = AIVoice("Fable", "fable", gender=Gender.FEMALE, models=all_models)
    NOVA = AIVoice("Nova", "nova", gender=Gender.FEMALE, models=all_models)
    ONYX = AIVoice("Onyx", "onyx", gender=Gender.MALE, models=all_models)
    SAGE = AIVoice("Sage", "sage", gender=Gender.FEMALE, models=all_models)
    SHIMMER = AIVoice("Shimmer", "shimmer", gender=Gender.FEMALE, models=all_models)
    VERSE = AIVoice("Verse", "verse", gender=Gender.MALE, models=latest_models)
    MARIN = AIVoice("Marin", "marin", gender=Gender.FEMALE, models=latest_models)
    CEDAR = AIVoice("Cedar", "cedar", gender=Gender.MALE, models=latest_models)

class OpenAIProvider(AudioProvider):
    def __init__(self) -> None:
        self.openai_voices = [voice.value for voice in OpenAIVoice]
        self.voice_names = sorted([voice.name for voice in self.openai_voices])

    @property
    def name(self) -> str:
        return "Open AI"

    @property
    def description(self) -> str:
        return "Open AI offers fast and natural speech built on the GPT-4o mini model with an estimated 1 billion parameter model."

    @property
    def supports_instructions(self) -> bool:
        return True

    @property
    def voices(self) -> List[AIVoice]:
        return self.openai_voices

    @property
    def location(self) -> Location:
        return Location.HOSTED

    def get_voice_names(self) -> List[str]:
        return self.voice_names
    
    def get_voice_id(self, name: str) -> str:
        return super().get_voice_id(name)
    
    def define_creds(self) -> None:
        openai_key_value = get_env_key("OPENAI_API_KEY", "openai_key_value")
        openai_key = st.text_input("API Key", openai_key_value, type="password", key="openai_provider_key")
        if openai_key:
          st.session_state["openai_key_value"] = openai_key
    
    def define_options(self) -> Dict:
        models = all_models
        model_seleced = st.selectbox("Speech Model", models, index=2)
        self.model = model_seleced

        voice_speed = st.slider("Voice Speed", 0.25, 4.0, 1.0, 0.25)

        return {
            "model_id": model_seleced,
            "api_key": os.getenv("OPENAI_API_KEY"),
            "speed": voice_speed
        }
    
    def define_voice_explorer(self) -> Dict:
        return self._show_voices(["gender"], sample_path="openai", model=self.model)
    
    def define_usage(self) -> Dict:
        return {}
    
    def generate(self, text, voice_id, instructions, api_key, model_id, speed, output_path):
        client = OpenAI(api_key=api_key)
        response = client.audio.speech.create(
            model=model_id,
            voice=voice_id,
            input=text,
            response_format="wav",
            speed=speed,
            instructions=instructions
        )
        response.write_to_file(output_path)        

    def generate_and_save(
        self,
        text: str,
        voice_id: str,
        line: int,
        options: Dict,
        guidance: str = None
    ) -> str:
        api_key = options["api_key"]
        model_id = options["model_id"]
        speed = options["speed"]
        print(guidance)
        if "test" in options:
            audio_file = f"./session/{st.session_state.session_id}/temp/test.wav"
        else:
            audio_file = f"./session/{st.session_state.session_id}/audio/line{line}.wav"

        os.makedirs(os.path.dirname(audio_file), exist_ok=True)
        self.generate(text, voice_id, guidance, api_key, model_id, speed, audio_file)
        return audio_file
    