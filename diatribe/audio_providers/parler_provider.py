import streamlit as st, torch, numpy as np, soundfile as sf
from typing import Dict, List
from pathlib import Path
from diatribe.data import AIVoice, Gender
from diatribe.audio_providers.audio_provider import AudioProvider, LocalProvider, Location
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

@st.cache_data
def parler_voices() -> list[AIVoice]:
    return [
        AIVoice("Elisabeth", "elisabeth", path=Path("models/parler/elisabeth.pt"), gender=Gender.FEMALE),
        AIVoice("Jerry", "jerry", path=Path("models/parler/jerry.pt"), gender=Gender.MALE),
        AIVoice("Talia", "talia", path=Path("models/parler/talia.pt"), gender=Gender.FEMALE),
        AIVoice("Thomas", "thomas", path=Path("models/parler/thomas.pt"), gender=Gender.MALE),
    ]

def calculate_max_tokens(text: str) -> int:
    chars_per_second = 5
    estimated_seconds = len(text) / chars_per_second
    tokens_per_second = 18
    max_new_tokens = int(estimated_seconds * tokens_per_second) + 300
    return max_new_tokens  


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def trim_trailing_silence(input: Path, output: Path, silence_thresh=-50.0, chunk_size=10):
    sound: AudioSegment = AudioSegment.from_wav(input)
    reversed_sound = sound.reverse()
    trim_point = detect_leading_silence(
        reversed_sound, 
        silence_threshold=silence_thresh, 
        chunk_size=chunk_size
    )
    if trim_point > 0:
        trim = reversed_sound[trim_point-100:]        
        trimmed = trim.reverse()
        silence = AudioSegment.silent(duration=200, frame_rate=sound.frame_rate)
        trimmed = trimmed + silence
    else:
        trimmed = sound
    trimmed.export(output, format="wav")


class ParlerProvider(AudioProvider):
    def __init__(self) -> None:
        self.device = LocalProvider.device()
        self.parler_voices = parler_voices()
        self.voice_names = sorted([voice.name for voice in self.parler_voices])

    @property
    def name(self) -> str:
        return "Parler"

    @property
    def description(self) -> str:
        return "Parler is a medium sized model with around 600 million parameters that can produce natural speech but is prone to audio artifacts, so keep the text short. This app uses the mini-expresso version of the model which is an AudioLM style architecture."

    @property
    def voices(self) -> list[AIVoice]:
        return self.parler_voices
    
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
        temp = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
        repetition_penalty = st.slider("Repetition Penalty", 0.1, 2.0, 1.2, 0.1)

        return {
            "temp": temp,
            "repetition_penalty": repetition_penalty
        }
    
    def define_voice_explorer(self) -> Dict:
        return self._show_voices(["gender"], sample_path="parler")
    
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
        model_id = "parler-tts/parler-tts-mini-expresso"
        device = self.device
        voice = self._get_voice_by_id(voice_id)
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        voice_dict = torch.load(voice.path, map_location=device)
        description_ids = voice_dict["input_ids"]
        description_attention_mask = voice_dict["attention_mask"]

        sentences = [text]
        waveforms = []
        for sentence in sentences:
            prompt_input_ids = tokenizer(sentence, return_tensors="pt").input_ids.to(device) 
            max_tokens = calculate_max_tokens(sentence)
            set_seed(42)     
            with torch.inference_mode():
                generation = model.generate(
                    input_ids=description_ids,
                    attention_mask=description_attention_mask,
                    prompt_input_ids=prompt_input_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,       
                    temperature=options["temp"],      
                    max_new_tokens=max_tokens,  
                    repetition_penalty=options["repetition_penalty"], 
                )
            waveforms.append(generation.squeeze().cpu().numpy())

        waveform = np.concatenate(waveforms)
        sample_rate = model.config.sampling_rate 
        temp_path = Path(f"./session/{st.session_state.session_id}/temp/parler.wav")
        sf.write(temp_path, waveform, sample_rate)
        trim_trailing_silence(temp_path, Path(output_file), silence_thresh=-25.0)

        return output_file
