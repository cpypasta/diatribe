import streamlit as st, wave
from diatribe.audio_providers.audio_provider import AudioProvider, Location
from piper.voice import PiperVoice
from piper.config import SynthesisConfig
from diatribe.data import AIVoice
from pathlib import Path
from typing import Dict

@st.cache_data
def get_piper_voices() -> list[AIVoice]:
    model_dir = Path("models/piper/en")
    model_paths = list(model_dir.rglob("*.onnx"))
    voices = []
    for model_path in model_paths:
        model_name = model_path.stem
        name = model_name.split("-")[1]
        if "_" in name:
            name = " ".join([part.capitalize() for part in name.split("_")])
        else:
            name = name.capitalize()

        if model_name.startswith("en_US"):
            accent = "American"
        else:
            accent = "British"
        voices.append(AIVoice(name, model_name, path=model_path, accent=accent))
    return voices


class PiperProvider(AudioProvider):
    def __init__(self):
        self.piper_voices = get_piper_voices()
        self.voice_names = sorted([v.name for v in self.voices])

    @property
    def name(self) -> str:
        return "Piper"

    @property
    def description(self) -> str:
        return "Piper is a very fast model, though usually a bit robotic, that ranges from 5-32 million parameters depending on the voice. It is based on the VITS architecture."

    @property
    def voices(self) -> list[AIVoice]:
        return self.piper_voices

    @property
    def location(self) -> Location:
        return Location.LOCAL

    def get_voice_names(self) -> list[str]:
        return self.voice_names
    
    def get_voice_id(self, name: str) -> str:
        return super().get_voice_id(name)
    
    def define_creds(self) -> None:
        pass

    def define_options(self) -> Dict:
        length_scale = st.slider("Length Scale", 0.5, 1.5, 1.2, 0.1, help="Controls the speed of the speech, with lower values being faster and higher being slower")
        noise_scale = st.slider("Noise Scale", 0.1, 2.0, 0.8, 0.1, help="Controls the expressiveness of the speech, with higher values being more expressive/breathy")
        noise_w_scale = st.slider("Noise Width Scale", 0.0, 0.5, 0.2, 0.05, help="Controls the timing variation of the speech, with higher values varying the timing more")
        volume = st.slider("Volume", 0.5, 2.0, 1.0, 0.1, help="Controls the volume of the speech")

        return {
            "length_scale": length_scale,
            "noise_scale": noise_scale,
            "noise_w_scale": noise_w_scale,
            "volume": volume
        }
    
    def define_usage(self) -> None:
        pass

    def define_voice_explorer(self) -> Dict:
        return self._show_voices(["accent"], sample_path="piper")
    
    def generate_and_save(
        self,
        text: str,
        voice_id: str,
        line: int,
        options: Dict,
        guidance: str | None = None
    ) -> str:
        output_file = self._output_file(line, options, st.session_state.session_id)           
        model_path = self._get_voice_by_id(voice_id).path
        voice = PiperVoice.load(model_path)

        config = SynthesisConfig(
            length_scale=options["length_scale"],
            noise_scale=options["noise_scale"],
            noise_w_scale=options["noise_w_scale"],
            volume=options["volume"]
        )

        with wave.open(output_file, "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(voice.config.sample_rate)
            voice.synthesize_wav(text, wav_file, syn_config=config)

        return output_file