import os, requests, streamlit as st
from diatribe.audio_providers.audio_provider import AudioProvider
from diatribe.audio_providers.dialogue_provider import DialogueProvider
from typing import List, Dict
from enum import Enum
from diatribe.dialogues import Dialogue
from pathlib import Path
from dataclasses import dataclass
from diatribe.utils import get_env_key
from diatribe.data import AIVoice, Gender

class PlayAIVoice(Enum):
    ANGELO = AIVoice("Angelo", "s3://voice-cloning-zero-shot/baf1ef41-36b6-428c-9bdf-50ba54682bd8/original/manifest.json", Gender.MALE)
    DEEDEE = AIVoice("Deedee", "s3://voice-cloning-zero-shot/e040bd1b-f190-4bdb-83f0-75ef85b18f84/original/manifest.json", Gender.FEMALE)

class PlayAIProvider(AudioProvider, DialogueProvider):
    def get_voice_names(self) -> List[str]:
        voices = [voice for voice in PlayAIVoice]
        return [voice.value.name for voice in voices]

    def get_voice_id(self, name: str) -> str:
        voice = next((voice.value.id for voice in PlayAIVoice if voice.value.name == name), None)
        if voice is None:
            raise Exception(f"Voice ID not found for voice name: {name}")
        return voice

    def define_creds(self) -> None:
        playai_key_value = get_env_key("PLAYAI_API_KEY", "playai_key_value")
        playai_key = st.text_input("Play AI API Key", playai_key_value, type="password", key="playai_provider_key")
        playai_user_id = get_env_key("PLAYAI_USER_ID", "playai_user_id")
        playai_user_id = st.text_input("Play AI User ID", playai_user_id, type="password", key="playai_user_id")        
        if playai_key and playai_user_id:
          st.session_state["playai_key_value"] = playai_key

    def define_options(self) -> Dict:
        return {
            "api_key": os.getenv("PLAYAI_API_KEY"),
            "user_id": os.getenv("PLAYAI_USER_ID")
        }

    def define_voice_explorer(self) -> Dict:
        return {}

    def define_usage(self):
        return None

    def generate_and_save(
        self,
        text: str,
        voice_id: str,
        line: int,
        options: Dict,
        guidance: str = None
    ) -> str:
        raise Exception("Cannot generate single line with Play AI")
    
    def generate_dialogue(
        self,
        lines: List[Dialogue],
        options: Dict
    ) -> str:
        headers = {
            "AUTHORIZATION": f"Bearer {options['api_key']}",
            "X-USER-ID": options["user_id"]
        } 
        characters = list(set([line.character for line in lines]))
        if len(characters) > 2:
            raise Exception("PlayAI only supports 2 characters")
        
        text = " ".join([f"{line.character.name}: {line.text}" for line in lines])
        character = lines[0].character
        character2 = lines[1].character
        voice = character.voice_id
        voice2 = character2.voice_id

        data = {
            "model": "PlayDialog",
            "text": text,
            "voice": voice,
            "voice2": voice2,
            "outputFormat": "mp3",
            "speed": 1,
            "sampleRate": 48000,
            "seed": None,
            "temperature": None,
            "turnPrefix": f"{character.name}:",
            "turnPrefix2": f"{character2.name}:",
            "prompt": None,
            "prompt2": None,
            "voiceConditioningSeconds": 20,
            "voiceConditioningSeconds2": 20,
            "language": "english"    
        }         

        output_path = f"./session/{st.session_state.session_id}/final/audio/dialogue.mp3"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        response = requests.post("https://api.play.ai/api/v1/tts/stream", headers=headers, json=data)
        if response.ok:
            with open(output_path, "wb") as f:
                f.write(response.content)
            return output_path
        else:
            print("Play AI responded with:", response.status_code)
            raise Exception(f"Failed to generate dialogue") 

