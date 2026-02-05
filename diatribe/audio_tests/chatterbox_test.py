import soundfile as sf, torchaudio as ta, torch
from chatterbox.tts_turbo import ChatterboxTurboTTS

def generate():
    model = ChatterboxTurboTTS.from_pretrained(device="mps")

    model.prepare_conditionals(
        wav_fpath="samples/openai/alloy.wav",
        exaggeration=0.5
    )

    text = "End [chuckle]? No, the journey does not end here. Death is just another path, one that we all must take [laugh]. The grey rain curtain of this world rolls back and all turns to silver glass. And then you see it."
    wav = model.generate(
        text,
        temperature=0.8
    )
    ta.save("output.wav", wav, model.sr)
    # sf.write("output.wav", wav, model.sr)

generate()