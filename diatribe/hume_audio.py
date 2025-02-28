import os, base64
import streamlit as st
from dotenv import load_dotenv
from hume import HumeClient
from hume.tts import PostedUtterance, PostedUtteranceVoiceWithName, ReturnGeneration, ReturnVoice
from enum import Enum
from typing import List

class HumeVoice(Enum):
    WISE_WIZARD = PostedUtteranceVoiceWithName(name="wise wizard", provider="HUME_AI")
    NATURE_DOCUMENTARY = PostedUtteranceVoiceWithName(name="name documentary narrator", provider="HUME_AI")
    TAVERN_TRAVELLER = PostedUtteranceVoiceWithName(name="tavern traveller", provider="CUSTOM_VOICE")

def save_voice(generation_id: str, voice_name: str) -> ReturnVoice:
    api_key = os.getenv("HUME_API_KEY")
    client = HumeClient(api_key=api_key)    
    return client.tts.voices.create(
        name=voice_name,
        generation_id=generation_id
    )

def get_voices() -> List[HumeVoice]:
    return [voice for voice in HumeVoice]

def get_voice_names(voices: List[HumeVoice]) -> List[str]:
    return [voice.value.name for voice in voices]

def get_voice_from_name(name: str) -> HumeVoice:
    return next((voice for voice in HumeVoice if voice.value.name == name), None)

def generate(
    text: str, 
    voice: HumeVoice, 
    guidance: str = None,
    api_key: str = None
) -> ReturnGeneration:
    api_key = "t3A3bzlqtfnasX7k0m0lSI5qPAEAMcGAT2mcOAnIdSKYJJdC"
    client = HumeClient(api_key=api_key)      
    speech = client.tts.synthesize_json(
        utterances=[
            PostedUtterance(
                text=text,
                description=guidance,
                voice=voice.value
            )
        ]
    )
    return speech.generations[0]

def generate_and_save(
  text: str,
  voice_id: str,
  line: int,
  api_key: str
) -> str:
    """Generate audio from a dialogue and save it to a file."""
    speech = generate(text, get_voice_from_name(voice_id), api_key=api_key)
    audio_data = base64.b64decode(speech.audio)
    audio_file = f"./session/{st.session_state.session_id}/audio/line{line}.wav"
    os.makedirs(os.path.dirname(audio_file), exist_ok=True)
    with open(audio_file, "wb") as f:
        f.write(audio_data)  
    return audio_file

if __name__ == "__main__":
    pass
    # voice = save_voice("d37d5e2c-1f6e-48bc-8089-7068c1d902aa", "tavern traveller") # 18db5456-c927-4f05-a9ce-89ca94ea9e7b
    # print(voice.id)