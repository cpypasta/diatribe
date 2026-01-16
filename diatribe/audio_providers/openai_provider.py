import os, streamlit as st
from diatribe.audio_providers.audio_provider import AudioProvider
from typing import List, Dict
from enum import Enum
from openai import OpenAI
from dataclasses import dataclass
from diatribe.utils import get_env_key

class Gender(Enum):
    MALE = "Male"
    FEMALE = "Female"

@dataclass
class AIVoice:
    name: str
    id: str
    gender: Gender

# TODO: new voices only work with gpt-4o-mini-tts model
class OpenAIVoice(Enum):
    ALLOY = AIVoice("Alloy", "alloy", Gender.FEMALE)
    ASH = AIVoice("Ash", "ash", Gender.MALE)    
    BALLAD = AIVoice("Ballad", "ballad", Gender.MALE) #
    CORAL = AIVoice("Coral", "coral", Gender.FEMALE)
    ECHO = AIVoice("Echo", "echo", Gender.MALE)
    FABLE = AIVoice("Fable", "fable", Gender.FEMALE)
    NOVA = AIVoice("Nova", "nova", Gender.FEMALE)
    ONYX = AIVoice("Onyx", "onyx", Gender.MALE)
    SAGE = AIVoice("Sage", "sage", Gender.FEMALE)
    SHIMMER = AIVoice("Shimmer", "shimmer", Gender.FEMALE)
    VERSE = AIVoice("Verse", "verse", Gender.MALE) #
    MARIN = AIVoice("Marin", "marin", Gender.FEMALE) #
    CEDAR = AIVoice("Cedar", "cedar", Gender.MALE) #
class OpenAIProvider(AudioProvider):
    def get_voice_names(self) -> List[str]:
        voices = [voice for voice in OpenAIVoice]
        return [voice.value.name for voice in voices]
    
    def get_voice_id(self, name: str) -> str:
        voice = next((voice.value.id for voice in OpenAIVoice if voice.value.name == name), None)
        if voice is None:
            raise Exception(f"Voice ID not found for voice name: {name}")
        return voice
    
    def define_creds(self) -> None:
        openai_key_value = get_env_key("OPENAI_API_KEY", "openai_key_value")
        openai_key = st.text_input("OpenAI API Key", openai_key_value, type="password", key="openai_provider_key")
        if openai_key:
          st.session_state["openai_key_value"] = openai_key
    
    def define_options(self) -> Dict:
        models = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
        model_seleced = st.selectbox("Speech Model", models, index=2)

        voice_speed = st.slider("Voice Speed", 0.25, 4.0, 1.0, 0.25)

        return {
            "model_id": model_seleced,
            "api_key": os.getenv("OPENAI_API_KEY"),
            "speed": voice_speed
        }
    
    def define_voice_explorer(self) -> Dict:
        voice_genders = set([voice.value.gender.value for voice in OpenAIVoice])
        voice_gender_selected = st.selectbox("Gender Filter", voice_genders, index=None)

        if voice_gender_selected:
            filtered_voice_names = [voice.value.name for voice in OpenAIVoice if voice.value.gender.value == voice_gender_selected]
        else:
            filtered_voice_names = self.get_voice_names()
            
        voice_selected = st.selectbox("Speaker", filtered_voice_names)        
        if voice_selected:
            voice_id_selected = self.get_voice_id(voice_selected)
            sample_path = f"./samples/openai/{voice_id_selected}.wav"
            if os.path.exists(sample_path):
                st.audio(sample_path, format="audio/wav")

        return {
            "voice_id": voice_id_selected
        }
    
    def define_usage(self) -> Dict:
        return {}
    
    def generate(self, text, voice_id, api_key, model_id, speed, output_path):
        client = OpenAI(api_key=api_key)
        response = client.audio.speech.create(
            model=model_id,
            voice=voice_id,
            input=text,
            response_format="wav",
            speed=speed
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

        if "test" in options:
            audio_file = f"./session/{st.session_state.session_id}/temp/test.wav"
        else:
            audio_file = f"./session/{st.session_state.session_id}/audio/line{line}.wav"

        os.makedirs(os.path.dirname(audio_file), exist_ok=True)
        self.generate(text, voice_id, api_key, model_id, speed, audio_file)
        return audio_file

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    provider = OpenAIProvider()
    provider.generate(
        "this is me", 
        OpenAIVoice.ALLOY.value.id, 
        os.environ["OPENAI_API_KEY"],
        "tts-1-hd",
        "sample.wav"
    )
    