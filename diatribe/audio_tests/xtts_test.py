import torch
from pathlib import Path
from TTS.api import TTS
import soundfile as sf
from TTS.tts.models.xtts import Xtts

# Detect device: prefer MPS on Apple Silicon, fallback to CPU
# device = "mps" if torch.backends.mps.is_available() else "cpu"

def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# print(TTS().list_models())

device=get_best_device()
print(f"Using device: {device}")

# Load XTTS-v2 (auto-downloads on first run)
tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    progress_bar=True
).to(device)
model: Xtts = tts.synthesizer.tts_model

# Paths & settings
text = "End? No, the journey does not end here. Death is just another path, one that we all must take. The grey rain curtain of this world rolls back and all turns to silver glass. And then you see it."
reference_audio = "/Users/broller/code/diatribe/samples/openai/alloy.wav"  # 6–30s clean WAV of the target voice
output_file = "xtts_output.wav"
language = "en"  # 'en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'ja', 'hu', 'ko', 'hi'

# Generate with zero-shot voice cloning
# print("Generating speech...")
# wav = tts.tts(
#     text=text,
#     speaker_wav=reference_audio,
#     language=language,
#     # Optional tweaks for quality/consistency
#     temperature=0.65,          # lower = more consistent, higher = more variation
#     length_penalty=1.0,
#     repetition_penalty=2.0,
#     top_k=30,
#     top_p=0.85,
#     speed=1.0
# )

def save_sample(voice_model: Path):
    print("Processing", voice_model.name, "...")
    voice = torch.load(voice_model, map_location=device, weights_only=True)
    voice_latent = voice["gpt_cond_latent"]
    voice_embedding = voice["speaker_embedding"]

    wav = model.inference(
        text=text,
        language=language,
        gpt_cond_latent=voice_latent,
        speaker_embedding=voice_embedding,
        # Optional tweaks for quality/consistency
        temperature=0.65,          # lower = more consistent, higher = more variation
        length_penalty=1.0,
        repetition_penalty=8.0,
        top_k=30,
        top_p=0.8,
        speed=1.0
    )["wav"]

    # Save output
    output_file = f"/Users/broller/code/diatribe/samples/xtts/{voice_model.stem}.wav"
    sf.write(output_file, wav, samplerate=tts.synthesizer.output_sample_rate)    

def save_voice(path: Path):
    cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=path,
    )
    torch.save({
        "gpt_cond_latent": cond_latent,
        "speaker_embedding": speaker_embedding
    }, f"/Users/broller/code/diatribe/samples/xtts/{path.stem}.pt", _use_new_zipfile_serialization=False)

# folder = Path("/Users/broller/code/diatribe/models/xtts")
# for file in folder.glob("*.pt"):
#     save_sample(file)

save_sample(Path("/Users/broller/code/diatribe/models/xtts/cedar.pt"))