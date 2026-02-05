import wave
from piper.voice import PiperVoice
from piper.config import SynthesisConfig
from pathlib import Path
from diatribe.data import AIVoice, Gender

text = "End? No, the journey does not end here. Death is just another path, one that we all must take. The grey rain curtain of this world rolls back and all turns to silver glass. And then you see it."

def download_voices():
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="rhasspy/piper-voices",
        allow_patterns=["en/**/*.onnx", "en/**/*.onnx.json"],
        local_dir=Path("/Users/broller/code/diatribe/models/piper"),
        local_dir_use_symlinks=False,
    )


def generate(model_path: Path, name: str, text: str):
    voice = PiperVoice.load(model_path)

    config = SynthesisConfig(
        length_scale=1.0, # lower is faster, higher is slower
        noise_scale=0.8, # higher is more expressive/breathy
        noise_w_scale=0.2, # higher varies timing
        volume=1.0
    )

    output_path = f"samples/piper/{name}.wav"
    with wave.open(output_path, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(voice.config.sample_rate)
        voice.synthesize_wav(text, wav_file, syn_config=config)

    print(f"Audio saved to {output_path}")

def find_voices() -> list[AIVoice]:
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

for voice in find_voices():
    if voice.path:
        generate(voice.path, voice.id, text)