import streamlit as st, torch, soundfile as sf
from typing import List, Dict
from diatribe.audio_providers.audio_provider import AudioProvider, LocalProvider, Location
from diatribe.data import AIVoice, Gender
from TTS.api import TTS
from TTS.tts.models.xtts import Xtts

@st.cache_data
def get_xtts_voices():
    return [
        AIVoice("Alloy", "alloy", gender=Gender.FEMALE),
        AIVoice("Ash", "ash", gender=Gender.MALE),
        AIVoice("Ballad", "ballad", gender=Gender.MALE),
        AIVoice("Coral", "coral", gender=Gender.FEMALE),
        AIVoice("Echo", "echo", gender=Gender.MALE),
        AIVoice("Fable", "fable", gender=Gender.FEMALE),
        AIVoice("Nova", "nova", gender=Gender.FEMALE),
        AIVoice("Onyx", "onyx", gender=Gender.MALE),
        AIVoice("Sage", "sage", gender=Gender.FEMALE),
        AIVoice("Shimmer", "shimmer", gender=Gender.FEMALE),
        AIVoice("Verse", "verse", gender=Gender.MALE),
        AIVoice("Marin", "marin", gender=Gender.FEMALE),
        AIVoice("Cedar", "cedar", gender=Gender.MALE)        
    ]

class XttsProvider(AudioProvider):
    def __init__(self):
        self.device = LocalProvider.device()
        self.xtts_voices = get_xtts_voices()
        self.voice_names = sorted([voice.name for voice in self.xtts_voices])

    @property
    def name(self) -> str:
        return "XTTS"

    @property
    def description(self) -> str:
        return "XTTS v2 is a medium sized model that has around 467 million parameters that offers more natural speech but is a bit slower. This is a autoregressive model using a GPT-2 architecture."

    @property
    def voices(self) -> List[AIVoice]:
        return self.xtts_voices

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
        temperature = st.slider("Temperature", 0.05, 1.0, 0.65, 0.05, help="Controls the randomness of the model's predictions. Lower values make the output more deterministic and focused, while higher values increase creativity but may reduce stability and coherence.")
        length_penalty = st.slider("Length Penalty", 0.05, 1.5, 1.0, 0.05, help="Applies an exponential penalty to the length of generated sequences in the autoregressive decoder. Higher values encourage shorter, more concise outputs (terse speech), while lower values allow for longer generations.")
        repetition_penalty = st.slider("Repetition Penalty", 2.0, 12.0, 9.0, 0.5, help="Penalizes the model for repeating tokens or phrases during decoding, helping to avoid issues like long silences, filler sounds, or looping content.")
        top_k = st.slider("Top-K Sampling", 20, 100, 30, 1, help="Limits sampling to the top K most likely tokens at each step. Lower values make outputs more predictable, while higher values allow more diversity.")
        top_p = st.slider("Top-P (Nucleus)", 0.7, 0.9, 0.8, 0.1, help="Lower values focus on more probable outputs (less diverse), while values closer to 1.0 increase variety.")
        voice_speed = st.slider("Voice Speed", 0.5, 2.0, 1.0, 0.1)

        return {
            "temperature": temperature,
            "length_penalty": length_penalty,
            "repetition_penalty": repetition_penalty,
            "top_k": top_k,
            "top_p": top_p,
            "speed": voice_speed
        }

    def define_voice_explorer(self) -> Dict:
        return self._show_voices(["gender"], sample_path="xtts")
    
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

        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        voice_model = f"./models/xtts/{voice_id}.pt"
        voice = torch.load(voice_model, map_location=self.device, weights_only=True)
        voice_latent = voice["gpt_cond_latent"]
        voice_embedding = voice["speaker_embedding"]   

        model: Xtts = tts.synthesizer.tts_model
        wav = model.inference(
            text=text,
            language="en",
            gpt_cond_latent=voice_latent,
            speaker_embedding=voice_embedding,
            temperature=options["temperature"],
            length_penalty=options["length_penalty"],
            repetition_penalty=options["repetition_penalty"],
            top_k=options["top_k"],
            top_p=options["top_p"],
            speed=options["speed"]
        )["wav"]             
        sf.write(output_file, wav, samplerate=tts.synthesizer.output_sample_rate)
        
        return output_file