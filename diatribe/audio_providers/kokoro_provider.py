import soundfile as sf, streamlit as st, os
from diatribe.audio_providers.audio_provider import AudioProvider
from typing import List, Dict
from kokoro import KPipeline
from huggingface_hub import HfApi
from enum import Enum
from dataclasses import dataclass

pipeline = KPipeline(lang_code="a", device="cpu")

class Gender(Enum):
    MALE = "Male"
    FEMALE = "Female"

@dataclass
class AIVoice:
    name: str
    id: str
    gender: Gender
    accent: str


def get_accent(voice_id: str):
    first = voice_id[0]
    if first == "e":
        return "Spanish"
    elif first == "f":
        return "French"
    elif first == "h":
        return "Hindi"
    elif first == "i":
        return "Italian"
    elif first == "p":
        return "Brazilian"
    elif first == "a":
        return "American"
    elif first == "b":
        return "British"
    elif first == "j":
        return "Japanese"
    elif first == "z":
        return "Chinese"
    else:
        return ""                            


@st.cache_data
def get_kokoro_voices() -> List[AIVoice]:
    api = HfApi()
    files = api.list_repo_files(repo_id="hexgrad/Kokoro-82M", repo_type="model")
    voice_files = [f for f in files if f.startswith('voices/') and f.endswith('.pt')]
    voice_names = [f.split('/')[-1].replace('.pt', '') for f in voice_files]
    voices = [
        AIVoice(name.split("_")[-1].capitalize() + f" ({get_accent(name)})", name, Gender.MALE if "m_" in name else Gender.FEMALE, get_accent(name))
        for name in voice_names
    ]
    return voices

class KokoroProvider(AudioProvider):
    def __init__(self):
        self.voices = get_kokoro_voices()
        self.voice_names = sorted([v.name for v in self.voices])

    def get_voice_names(self) -> List[str]:
        return self.voice_names
    
    def get_voice_id(self, name: str) -> str:
        voice_id = next((x.id for x in self.voices if name == x.name), None)
        if voice_id is None:
            raise Exception(f"Voice ID not found for voice name: {name}")
        return voice_id
    
    def define_creds(self) -> None:
        pass

    def define_options(self) -> Dict:
        voice_speed = st.slider("Voice Speed", 0.5, 2.0, 1.0, 0.1)

        return {
            "speed": voice_speed
        }
    
    def define_voice_explorer(self) -> Dict:
        voice_genders = set([voice.gender.value for voice in self.voices])
        voice_gender_selected = st.selectbox("Gender Filter", voice_genders, index=None)

        voice_accents = sorted(set([voice.accent for voice in self.voices]))
        voice_accent_selected = st.selectbox("Accent Filter", voice_accents, index=None)


        if voice_gender_selected:
            filtered_voices = [voice for voice in self.voices if voice.gender.value == voice_gender_selected]
        else:
            filtered_voices = self.voices

        if voice_accent_selected:
            filtered_voices = [voice for voice in filtered_voices if voice.accent == voice_accent_selected]
            
        filtered_voice_names = sorted([voice.name for voice in filtered_voices])

        voice_selected = st.selectbox("Speaker", filtered_voice_names)        
        if voice_selected:
            try:
                voice_id_selected = self.get_voice_id(voice_selected)
                st.audio(f"./samples/{voice_id_selected}.wav", format="audio/wav")
            except:
                st.toast(f"No voice sample found for {voice_selected}")

        return {}
    
    def define_usage(self) -> None:
        pass
    
    def generate(self, text, voice_id, output_path, speed):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        generator = pipeline(
            text,
            voice=voice_id,
            speed=speed,
            split_pattern=r'\n+'
        )                   
        for _, (_, _, audio) in enumerate(generator):
            sf.write(output_path, audio, 24000)        

    def generate_and_save(
        self,
        text: str,
        voice_id: str,
        line: int,
        options: Dict,
        guidance: str = None
    ) -> str:
        audio_file = f"./session/{st.session_state.session_id}/audio/line{line}.wav"
        speed = options["speed"] if "speed" in options else 1.0
        self.generate(text, voice_id, audio_file, speed)

if __name__ == "__main__":
    voices = get_kokoro_voices()
    for v in voices:
        print(v)