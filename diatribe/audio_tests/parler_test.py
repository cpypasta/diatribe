import torch, numpy as np, nltk
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import detect_leading_silence
from pathlib import Path
from nltk.tokenize import sent_tokenize

# nltk.download('punkt_tab')

device = "mps"
model_id = "parler-tts/parler-tts-mini-expresso" # smalled model
# model_id = "parler-tts/parler-tts-mini-v1"
# model_id = "MoritzLaurer/parler-tts-large-v1"

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Optional speedup (PyTorch 2.0+ compile — can give 20-50% faster on MPS/CPU)
# model = torch.compile(model)

# Jerry (male), Thomas (male), Elisabeth (female), Talia (female)

def chunk_text(text: str, min_chars: int = 30) -> list[str]:
    setences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sent in setences:
        sent = sent.strip()
        if not sent:
            continue
        
        if len(current_chunk) <= min_chars:
            if current_chunk:
                current_chunk += " " + sent
            else:
                current_chunk = sent            
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sent

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

description = (
    "Elisabeth's voice is soft and warm with a gentle female tone, speaking at a moderate pace "
    "with clear enunciation and a hint of friendliness, in a quiet studio recording."
)

# description = (
#     "Thomas' voice is warm and confident with a male tone, speaking at a moderate pace "
#     "with clear enunciation and a hint of friendliness, in a quiet studio recording."
# )

text = (
    "Hello Brian, this is a woman who wants to suck your juicy dick."
)
text = (
    "Hello Brian, my name is not important, but what IS important is what I need to tell you."
)

def calculate_max_tokens(text: str) -> int:
    chars_per_second = 5
    estimated_seconds = len(text) / chars_per_second
    tokens_per_second = 18
    max_new_tokens = int(estimated_seconds * tokens_per_second) + 300
    return max_new_tokens  

def save_voice():
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    description_tokens = tokenizer(
        description, 
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )    
    description_ids = description_tokens.input_ids.to(device)
    description_attention_mask = description_tokens.attention_mask.to(device)     

    voice_dict = {
        "description": description,
        "input_ids": description_ids,
        "attention_mask": description_attention_mask
    }
    torch.save(voice_dict, "parler_voice.pt", _use_new_zipfile_serialization=False)

def trim_trailing_silence(sample_path: Path, silence_thresh=-50.0, chunk_size=10):
    sound: AudioSegment = AudioSegment.from_wav(sample_path)
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
    trimmed.export(sample_path, format="wav")

def generate_sample(model_path: Path, text: str):
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    voice_dict = torch.load(model_path, map_location=device)
    description_ids = voice_dict["input_ids"]
    description_attention_mask = voice_dict["attention_mask"]

    # sentences = chunk_text(text)
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
                temperature=0.7,      
                max_new_tokens=max_tokens,  
                repetition_penalty=1.2, 
            )
        waveforms.append(generation.squeeze().cpu().numpy())

    waveform = np.concatenate(waveforms)
    sample_rate = model.config.sampling_rate 
    sample_path = Path(f"/Users/broller/code/diatribe/samples/parler/{model_path.stem}.wav")    
    sf.write(sample_path, waveform, sample_rate)
    trim_trailing_silence(sample_path, silence_thresh=-25.0)

def generate():
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    description_tokens = tokenizer(
        description, 
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )
    description_ids = description_tokens.input_ids.to(device)
    description_attention_mask = description_tokens.attention_mask.to(device)  
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device) 
    max_tokens = calculate_max_tokens(text)
    set_seed(42)     
    with torch.inference_mode():
        generation = model.generate(
            input_ids=description_ids,
            attention_mask=description_attention_mask,
            prompt_input_ids=prompt_input_ids,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,       
            temperature=0.7,      
            max_new_tokens=max_tokens,  
            repetition_penalty=1.2, 
        )
    sample_rate = model.config.sampling_rate 
    waveform = generation.squeeze().cpu().numpy()
    sf.write("parler_output.wav", waveform, sample_rate)

# save_voice()
# generate()
# generate_sample(Path("parler_voice.pt"))
# trim_trailing_silence(silence_thresh=-25.0)

# text = "End? No, the journey does not end here. Death is just another path, one that we all must take. The grey rain curtain of this world rolls back and all turns to silver glass. And then you see it."
text = "No, the journey does not end here. Death is just another path, one that we all must take."
generate_sample(Path("/Users/broller/code/diatribe/models/parler/thomas.pt"), text)

# for file in Path("/Users/broller/code/diatribe/models/parler").glob("*.pt"):
#     generate_sample(file, text)

# print(chunk_text(text))