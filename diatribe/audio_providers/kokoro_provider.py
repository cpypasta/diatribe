import soundfile as sf, streamlit as st
from diatribe.audio_providers.audio_provider import AudioProvider, Location
from typing import List, Dict
from kokoro import KPipeline
from huggingface_hub import HfApi
from diatribe.data import AIVoice, Gender

pipeline = KPipeline(lang_code="a")

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
        AIVoice(
            name.split("_")[-1].capitalize(), 
            name, 
            gender=Gender.MALE if "m_" in name else Gender.FEMALE, 
            accent=get_accent(name)
        )
        for name in voice_names
    ]
    return voices

class KokoroProvider(AudioProvider):
    def __init__(self):
        self.kokoro_voices = get_kokoro_voices()
        self.voice_names = sorted([v.name for v in self.kokoro_voices])

    @property
    def name(self) -> str:
        return "Kokoro"

    @property
    def description(self) -> str:
        return "Kokoro is a 82 million parameter model that balances speed and quality. It is a model based on the StyleTTS 2 architecture."

    @property
    def voices(self) -> List[AIVoice]:
        return self.kokoro_voices

    @property
    def location(self) -> Location:
        return Location.LOCAL

    def get_voice_names(self) -> List[str]:
        return self.voice_names
    
    def get_voice_id(self, name: str) -> str:
        return super().get_voice_id(name)
    
    def define_creds(self) -> None:
        pass

    def define_options(self) -> Dict:
        voice_speed = st.slider("Voice Speed", 0.5, 2.0, 1.0, 0.1)

        return {
            "speed": voice_speed
        }
    
    def define_voice_explorer(self) -> Dict:
        return self._show_voices(["gender", "accent"], sample_path="kokoro")
    
    def define_usage(self) -> None:
        pass

    def generate(self, text, voice_id, output_path, speed):
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
        guidance: str | None = None
    ) -> str:
        audio_file = self._output_file(line, options, st.session_state.session_id)
        speed = options["speed"] if "speed" in options else 1.0
        self.generate(text, voice_id, audio_file, speed)
        return audio_file