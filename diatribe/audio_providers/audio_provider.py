import torch, os, streamlit as st
from abc import ABC, abstractmethod
from typing import List, Dict
from diatribe.data import AIVoice
from enum import Enum

class Location(Enum):
    LOCAL = "local"
    HOSTED = "hosted"

class LocalProvider:
    _DEVICE = None

    @classmethod
    def get_device(cls) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        cls._DEVICE = device
        return device

    @classmethod
    def device(cls) -> torch.device:
        if cls._DEVICE is None:
            return cls.get_device()
        return cls._DEVICE


class AudioProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    def supports_instructions(self) -> bool:
        return False

    @property
    @abstractmethod
    def voices(self) -> List[AIVoice]:
        pass

    @property
    @abstractmethod
    def location(self) -> Location:
        pass

    @abstractmethod
    def get_voice_names(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_voice_id(self, name: str) -> str:
        voice_id = next((x.id for x in self.voices if name == x.name), None)
        if voice_id is None:
            raise Exception(f"Voice ID not found for voice name: {name}")
        return voice_id

    @abstractmethod
    def define_creds(self) -> None:
        pass

    @abstractmethod
    def define_options(self) -> Dict:
        pass
    
    @abstractmethod
    def define_voice_explorer(self) -> Dict:
        pass

    @property
    def has_usage(self) -> bool:
        return False

    @abstractmethod
    def define_usage(self) -> None:
        pass

    @abstractmethod
    def generate_and_save(
        self,
        text: str,
        voice_id: str,
        line: int,
        options: Dict,
        guidance: str | None = None
    ) -> str:
        pass                

    def _output_file(self, line: int, options: Dict, session_id) -> str:
        if "test" in options:
            audio_file = f"./session/{st.session_state.session_id}/temp/test.wav"
        else:
            audio_file = f"./session/{st.session_state.session_id}/audio/line{line}.wav"        
        os.makedirs(os.path.dirname(audio_file), exist_ok=True)
        return audio_file

    def _get_voice_by_name(self, name: str) -> AIVoice | None:
        for voice in self.voices:
            if voice.name == name:
                return voice
        return None

    def _get_voice_by_id(self, id: str) -> AIVoice | None:
        for voice in self.voices:
            if voice.id == id:
                return voice
        return None

    def _voice_genders(self) -> List[str]:
        return sorted(set([voice.gender.value for voice in self.voices if voice.gender]))

    def _voice_accents(self) -> List[str]:
        return sorted(set([voice.accent for voice in self.voices if voice.accent]))

    def _voice_ages(self) -> List[str]:
        return sorted(set([voice.age for voice in self.voices if voice.age]))

    def _filter_voices(self, 
        gender: str | None = None, 
        accent: str | None = None, 
        age: str | None = None,
        cloned: bool | None = None,
        model: str | None = None
    ) -> List[AIVoice]:
        filtered_voices = self.voices
        if gender:
            filtered_voices = [voice for voice in filtered_voices if voice.gender and voice.gender.value == gender]
        if accent:
            filtered_voices = [voice for voice in filtered_voices if voice.accent and voice.accent == accent]
        if age:
            filtered_voices = [voice for voice in filtered_voices if voice.age and voice.age == age]
        if cloned is not None:
            filtered_voices = [voice for voice in filtered_voices if voice.cloned == cloned]
        if model:
            filtered_voices = [voice for voice in filtered_voices if model in voice.models]
        return sorted(filtered_voices, key=lambda x: x.name)
    

    def _show_voices(self, filters: list[str], sample_path: str = None, model: str = None) -> Dict:
        if "gender" in filters:
            voice_genders = self._voice_genders()
            voice_gender_selected = st.selectbox("Gender", voice_genders, index=None)
        else:
            voice_gender_selected = None

        if "accent" in filters:
            voice_accents = self._voice_accents()
            voice_accent_selected = st.selectbox("Accent", voice_accents, index=None)
        else:
            voice_accent_selected = None

        if "age" in filters:
            voice_ages = self._voice_ages()
            voice_age_selected = st.selectbox("Age", voice_ages, index=None)
        else:
            voice_age_selected = None

        if "cloned" in filters:
            cloned_voices_only = st.toggle("Cloned Voices Only", value=False)
        else:
            cloned_voices_only = None

        filtered_voices = self._filter_voices(
            gender=voice_gender_selected, 
            accent=voice_accent_selected,
            age=voice_age_selected,
            cloned=cloned_voices_only,
            model=model
        )            
        filtered_voice_names = [voice.name for voice in filtered_voices]

        voice_selected = st.selectbox("Speaker", filtered_voice_names)        
        if voice_selected:
            try:
                voice_id_selected = self.get_voice_id(voice_selected)
                selected_voice = self._get_voice_by_name(voice_selected)
                local_sample = f"./samples/{sample_path}/{voice_id_selected}.wav" if sample_path else ""
                if selected_voice and selected_voice.sample_url:
                    st.audio(selected_voice.sample_url, format="audio/mp3")
                elif selected_voice.path and str(selected_voice.path).endswith(".wav"):
                    st.audio(selected_voice.path, format="audio/wav")
                elif os.path.exists(local_sample):
                    st.audio(local_sample, format="audio/wav")
                return {"voice_id": voice_id_selected}
            except:
                st.toast(f"No voice sample found for {voice_selected}")   
                return {}  
        return {}   