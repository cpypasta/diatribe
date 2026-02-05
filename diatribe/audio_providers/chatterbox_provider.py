import streamlit as st, torchaudio as ta
from typing import List, Dict
from diatribe.audio_providers.audio_provider import AudioProvider, LocalProvider, Location
from diatribe.data import AIVoice, Gender
from pathlib import Path
from chatterbox.tts_turbo import ChatterboxTurboTTS

@st.cache_data
def get_chatterbox_voices():
    return [
        AIVoice("Alloy", "alloy", path=Path("samples/openai/alloy.wav"), gender=Gender.FEMALE),
        AIVoice("Ash", "ash", path=Path("samples/openai/ash.wav"), gender=Gender.MALE),
        AIVoice("Ballad", "ballad", path=Path("samples/openai/ballad.wav"), gender=Gender.MALE),
        AIVoice("Coral", "coral", path=Path("samples/openai/coral.wav"), gender=Gender.FEMALE),
        AIVoice("Echo", "echo", path=Path("samples/openai/echo.wav"), gender=Gender.MALE),
        AIVoice("Fable", "fable", path=Path("samples/openai/fable.wav"), gender=Gender.FEMALE),
        AIVoice("Nova", "nova", path=Path("samples/openai/nova.wav"), gender=Gender.FEMALE),
        AIVoice("Onyx", "onyx", path=Path("samples/openai/onyx.wav"), gender=Gender.MALE),
        AIVoice("Sage", "sage", path=Path("samples/openai/sage.wav"), gender=Gender.FEMALE),
        AIVoice("Shimmer", "shimmer", path=Path("samples/openai/shimmer.wav"), gender=Gender.FEMALE),
        AIVoice("Verse", "verse", path=Path("samples/openai/verse.wav"), gender=Gender.MALE),
        AIVoice("Marin", "marin", path=Path("samples/openai/marin.wav"), gender=Gender.FEMALE),
        AIVoice("Cedar", "cedar", path=Path("samples/openai/cedar.wav"), gender=Gender.MALE),
        AIVoice("Talia", "talia", path=Path("samples/parler/talia.wav"), gender=Gender.FEMALE)   
    ]

class ChatterboxProvider(AudioProvider):
    def __init__(self):
        # self.device = LocalProvider.device()
        self.device = "cpu"
        self.chatterbox_voices = get_chatterbox_voices()
        self.voice_names = sorted([voice.name for voice in self.chatterbox_voices])

    @property
    def name(self) -> str:
        return "Chatterbox"
    
    @property
    def description(self) -> str:
        return "Chatterbox Turbo is a model by Resemble AI that produces high quality speech with 350 million parameters. The model also supports adding emotional tags like [cough], [laugh], or [chuckle]."
    
    @property
    def voices(self) -> List[AIVoice]:
        return self.chatterbox_voices
    
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
        temperature = st.slider("Temperature", 0.05, 1.0, 0.8, 0.05, help="Controls the randomness of the model's predictions. Lower values make the output more deterministic and focused, while higher values increase creativity but may reduce stability and coherence.")

        return {
            "temperature": temperature
        }
    
    def define_voice_explorer(self) -> Dict:
        return self._show_voices(["gender"], sample_path="openai")
    
    def define_usage(self) -> None:
        pass

    def generate_and_save(
        self,
        text: str,
        voice_id: str,
        line: int,
        options: Dict,
        guidance: str | None = None
    ) -> str:
        output_file = self._output_file(line, options, st.session_state.session_id) 
        voice = self._get_voice_by_id(voice_id)
        model = ChatterboxTurboTTS.from_pretrained(device=self.device)

        model.prepare_conditionals(
            wav_fpath=voice.path,
            exaggeration=0.5
        )
        wav = model.generate(
            text,
            temperature=options["temperature"]
        )
        ta.save(output_file, wav, model.sr)
        print("saved", output_file)

        return output_file
