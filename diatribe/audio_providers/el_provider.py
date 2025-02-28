import elevenlabs as el, streamlit as st, re, os, traceback, datetime
from typing import List, Dict
from diatribe.audio_providers.audio_provider import AudioProvider
from elevenlabs import Voice, Models, Model, VoiceSettings, User

@st.cache_data
def get_voices() -> list[Voice]:
  voices: list[Voice] = list(el.voices())
  voices.sort(key=lambda x: x.name)
  return voices

@st.cache_data
def get_models() -> List[Model]:
  return list(Models.from_api())

def extract_name(s: str) -> str:
  """Extract the voice name from the voice name with (cloned) suffix."""
  match = re.match(r"(.*?)( \(cloned\))?$", s)
  if match:
    return match.group(1)
  return s  
    
def get_voice_by_name(name: str, voices: list[Voice]) -> Voice:
  """Get a voice by the voice name."""
  name = extract_name(name)
  return next((v for v in voices if v.name == name), None)    
    
def get_voice_id(voice_name: str, voices: list[Voice]) -> str:
  """Get the voice ID from the voice name."""
  voice_name = extract_name(voice_name)
  voice_index = next((i for i, v in enumerate(voices) if v.name == voice_name), None)
  if voice_index is not None:
    return voices[voice_index].voice_id
  else:
    return None    
    
def voice_names_with_filter(
  voices: list[Voice], 
  gender: str, 
  age: str, 
  accent: str, 
  cloned: bool
) -> list[str]:
  """Get a list of voice names filtered by gender, age, accent, and cloned."""
  voice_names = []
  for v in voices:
    v_labels = v.labels
    v_gender = v_labels["gender"] if "gender" in v_labels else None
    v_age = v_labels["age"] if "age" in v_labels else None
    v_acccent = v_labels["accent"] if "accent" in v_labels else None
    match = True
    if gender and v_gender != gender:
      match = False
    if age and v_age != age:
      match = False
    if accent and v_acccent != accent:
      match = False
    if cloned:
      if v.category != "cloned":
        match = False
    if match:
      voice_names.append(f"{v.name} ({v.category})" if v.category == "cloned" else v.name)
  return voice_names    
    
def generate(
  text: str,
  voice_id: str,
  options: Dict 
) -> bytes:
  """Generate audio from a dialogue."""
  try:
    audio = el.generate(
      text=text,
      model=options["model_id"],
      voice = Voice(
        voice_id=voice_id,
        settings=VoiceSettings(
          stability=options["stability"],
          similarity_boost=options["similarity_boost"],
          style=options["style"]
        )
      )
    ) 
  except:
    traceback.print_exc()
    raise Exception("Failed to generate ElevenLabs audio.")
  return audio    
    
@st.cache_data(ttl=900)
def get_usage_percent() -> dict:
  """Get the character usage percent from the Eleven Labs API."""
  user_info = User.from_api()
  percent = user_info.subscription.character_count / user_info.subscription.character_limit * 100
  resets = user_info.subscription.next_character_count_reset_unix
  resets = datetime.datetime.fromtimestamp(resets).strftime("%m/%d")
  return {
    "usage": percent,
    "reset": resets,
    "count": user_info.subscription.character_count,
    "limit": user_info.subscription.character_limit
  }
          
class ElevenLabsProvider(AudioProvider):
    def get_voice_names(self) -> List[str]:
        el_voices = get_voices()
        return [f"{voice.name}{' (cloned)' if voice.category == 'cloned' else ''}" for voice in el_voices]
      
    def get_voice_id(self, name) -> str:
      voices = get_voices()
      voice_id = get_voice_id(name, voices)
      if voice_id is None:
        raise Exception(f"Voice ID not found for voice name: {name}")
      return voice_id
      
    def define_options(self) -> Dict:
        models = get_models()
        model_ids = [m.model_id for m in models]
        model_names = [m.name for m in models]
        try:
          turbo_model_index = model_names.index("Eleven Turbo v2.5")
        except:
          turbo_model_index = 0
        model_name = st.selectbox("Speech Model", model_names, index=turbo_model_index)
        if model_name:
          model_index = model_names.index(model_name)
          model_id = model_ids[model_index]   
                           
        stability = st.slider(
          "Stability", 
          0.0, 
          1.0, 
          value=0.35, 
          help="Increasing stability will make the voice more consistent between re-generations, but it can also make it sounds a bit monotone. On longer text fragments we recommend lowering this value."
        )
        simarlity_boost = st.slider(
          "Clarity + Simalarity Enhancement",
          0.0,
          1.0,
          value=0.80,
          help="High enhancement boosts overall voice clarity and target speaker similarity. Very high values can cause artifacts, so adjusting this setting to find the optimal value is encouraged."
        )
        style = st.slider(
          "Style Exaggeration",
          0.0,
          1.0,
          value=0.0,
          help="High values are recommended if the style of the speech should be exaggerated compared to the uploaded audio. Higher values can lead to more instability in the generated speech. Setting this to 0.0 will greatly increase generation speed and is the default setting."
        )         
        
        return {
          "model_id": model_id,
          "stability": stability,
          "similarity_boost": simarlity_boost,
          "style": style
        }
        
    def define_voice_explorer(self) -> Dict:
        el_voices = get_voices()
        el_voice_accents = set([voice.labels["accent"] for voice in el_voices if "accent" in voice.labels])
        el_voice_ages = set([voice.labels["age"] for voice in el_voices if "age" in voice.labels])
        el_voice_genders = set([voice.labels["gender"] for voice in el_voices if "gender" in voice.labels])
        el_voice_accents = sorted(el_voice_accents, key=lambda x: x.lower())
        
        el_voice_cloned = st.toggle("Cloned Voices Only", value=False)
        el_voice_gender = st.selectbox("Gender Filter", el_voice_genders, index=None)
        el_voice_age = st.selectbox("Age Filter", el_voice_ages, index=None)
        el_voice_accent = st.selectbox("Accent Filter", el_voice_accents, index=None)
        
        speaker_voice_names = voice_names_with_filter(
          el_voices, 
          el_voice_gender, 
          el_voice_age, 
          el_voice_accent,
          el_voice_cloned
        )
        el_voice = st.selectbox("Speaker", speaker_voice_names)
        if el_voice:
          el_voice_details = get_voice_by_name(el_voice, el_voices)
          el_voice_id = el_voice_details.voice_id
          
          if el_voice_details.preview_url:
            st.audio(el_voice_details.preview_url, format="audio/mp3")
        
          st.markdown(f"_Voice ID: {el_voice_id}_")  
        return {}           
      
    def define_usage(self):
      usage = get_usage_percent()
      st.markdown(f"**Character Percent:** {usage['usage']:.1f}%")
      st.markdown(f"**Character Count:** {usage['count']:,}")
      st.markdown(f"**Character Limit:** {usage['limit']:,}")
      st.markdown(f"**Reset:** {usage['reset']}")      
      
    def generate_and_save(
      self,
      text: str,
      voice_id: str,
      line: int,
      options: Dict,
      guidance: str = None
    ) -> str:
      """Generate audio from a dialogue and save it to a file."""
      audio = generate(text, voice_id, options)
      audio_file = f"./session/{st.session_state.session_id}/audio/line{line}.wav"
      os.makedirs(os.path.dirname(audio_file), exist_ok=True)
      with open(audio_file, "wb") as f:
        f.write(audio)  
      return audio_file