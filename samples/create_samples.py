import os, requests
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from diatribe.audio_providers.kokoro_provider import KokoroProvider, get_kokoro_voices

load_dotenv()

text = "The End? No, the journey doesn't end here. Death is just another path, one that we all must take. The grey rain curtain of this world rolls back, and all turns to silver glass, and then you see it.",

# def generate(voice_id: str, text: str):
#     audio_file = f"./samples/{voice_id}.wav"
#     if Path(audio_file).exists():
#         return

#     api_key = options["api_key"]
#     model_id = options["model_id"]
#     client = OpenAI(api_key=api_key)
#     response = client.audio.speech.create(
#         model=model_id,
#         voice=voice_id,
#         input=text,
#         response_format="wav"
#     )
#     response.write_to_file(audio_file)

# options = {
#     "model_id": "tts-1-hd",
#     "api_key": os.getenv("OPENAI_API_KEY")
# }

# text = "End? No, the journey doesn't end here. Death is just another path, one that we all must take. The grey rain curtain of this world rolls back, and all turns to silver glass, and then you see it."
# voice_ids = ["alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"]

# for voice_id in voice_ids:
#     generate(voice_id, text)
#     print(f"Generated audio for {voice_id}")

# headers = {
#     "AUTHORIZATION": f"Bearer {os.getenv('PLAYAI_API_KEY')}",
#     "X-USER-ID": os.getenv("PLAYAI_USER_ID")
# }
# data = {
#     "model": "PlayDialog",
#     "text": "Country Mouse: Welcome to my humble home, cousin! Town Mouse: Thank you, cousin. It's quite... peaceful here.",
#     "voice": "s3://voice-cloning-zero-shot/baf1ef41-36b6-428c-9bdf-50ba54682bd8/original/manifest.json",
#     "voice2": "s3://voice-cloning-zero-shot/baf1ef41-36b6-428c-9bdf-50ba54682bd8/original/manifest.json",
#     "outputFormat": "wav",
#     "speed": 1,
#     "sampleRate": 48000,
#     "seed": None,
#     "temperature": None,
#     "turnPrefix": "Country Mouse:",
#     "turnPrefix2": "Town Mouse:",
#     "prompt": None,
#     "prompt2": None,
#     "voiceConditioningSeconds": 20,
#     "voiceConditioningSeconds2": 20,
#     "language": "english"    
# }

# data2 = {
#     "model": "Play3.0-mini",
#     "text": "End? No, the journey doesn't end here. Death is just another path, one that we all must take. The grey rain curtain of this world rolls back, and all turns to silver glass, and then you see it.",
#     "voice": "s3://voice-cloning-zero-shot/baf1ef41-36b6-428c-9bdf-50ba54682bd8/original/manifest.json",
#     "quality": "premium",
#     "outputFormat": "wav",
#     "speed": 1,
#     "sampleRate": 48000,
#     "seed": None,
#     "temperature": None,
#     "voiceGuidance": None,
#     "styleGuidance": 10,
#     "textGuidance": 1,
#     "language": "english"
# }

# response = requests.post("https://api.play.ai/api/v1/tts/stream", headers=headers, json=data2)
# if response.ok:
#     with open("sample.wav", "wb") as f:
#         f.write(response.content)
#     print("Audio generated successfully")
# else:
#     print(response.status_code)
#     print(response.text)

voices = get_kokoro_voices()
provider = KokoroProvider()
for voice in voices:
    provider.generate(text, voice.id, f"./samples/{voice.id}.wav")