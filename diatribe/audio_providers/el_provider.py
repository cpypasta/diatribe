import elevenlabs as el, streamlit as st, re, os, traceback, datetime
from elevenlabs.client import ElevenLabs
from typing import List, Dict
from diatribe.audio_providers.audio_provider import AudioProvider, Location
from diatribe.utils import get_env_key
from elevenlabs import VoiceSettings
from elevenlabs.types import Voice, Model
from diatribe.data import AIVoice, Gender

@st.cache_data
def get_voices(api_key) -> List[AIVoice]:
  voices: list[Voice] = ElevenLabs(api_key=api_key).voices.get_all().voices  
  genders = {m.value for m in Gender}

  def get_gender(labels: Dict | None) -> Gender | None:
    if not labels:
      return None
    value = labels.get("gender")
    return Gender(value.capitalize()) if value and value.capitalize() in genders else None

  def get_label(label: str, labels: Dict | None) -> str | None:
    if not labels:
      return None
    return labels.get(label)

  ai_voices = [
    AIVoice(
      str(voice.name), 
      voice.voice_id,
      gender=get_gender(voice.labels),
      accent=get_label("accent", voice.labels),
      age=get_label("age", voice.labels),
      models=voice.high_quality_base_model_ids if voice.high_quality_base_model_ids else [],
      cloned=(voice.category == "cloned"),
      sample_url=voice.preview_url
    )
    for voice in voices
  ]
  return ai_voices

@st.cache_data
def get_models(api_key) -> List[Model]:
  models = ElevenLabs(api_key=api_key).models.list()
  return models   
    

def generate(
  text: str,
  voice_id: str,
  options: Dict,
  api_key: str
) -> bytes:
  """Generate audio from a dialogue."""
  try:
    audio = ElevenLabs(api_key=api_key).text_to_speech.convert(
      voice_id=voice_id,
      output_format=options["output_format"], # TextToSpeechConvertRequestOutputFormat
      text=text,
      model_id=options["model_id"],
      voice_settings=VoiceSettings(
        stability=options["stability"],
        similarity_boost=options["similarity_boost"],
        style=options["style"]
      )     
    )
  except:
    traceback.print_exc()
    raise Exception("Failed to generate ElevenLabs audio.")
  return b''.join(audio)    
    
@st.cache_data(ttl=900)
def get_usage_percent(api_key) -> dict:
  """Get the character usage percent from the Eleven Labs API."""
  user_info = ElevenLabs(api_key=api_key).user.get()
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
    @property
    def name(self) -> str:
        return "ElevenLabs"    

    @property
    def description(self) -> str:
        return "ElevenLabs is a leading provider of TTS models."

    @property
    def has_usage(self) -> bool:
        return True

    @property
    def voices(self) -> List[AIVoice]:
        if not hasattr(self, "el_voices"):
            self.el_voices = get_voices(self.api_key)
        return self.el_voices

    @property
    def location(self) -> Location:
        return Location.HOSTED

    def get_voice_names(self) -> List[str]:
        if self.voice_names:
            return self.voice_names
        self.el_voices = get_voices(self.api_key)
        self.voice_names = sorted([voice.name for voice in self.voices])
        return self.voice_names
      
    def get_voice_id(self, name) -> str:
      return super().get_voice_id(name)
      
    def define_creds(self) -> None:
      el_key_value = get_env_key("ELEVENLABS_API_KEY", "el_key_value")
      el_key = st.text_input("API Key", el_key_value, type="password", key="el_key")  
      if el_key:
        st.session_state["el_key_value"] = el_key    
      self.api_key = get_env_key("ELEVENLABS_API_KEY", "el_key_value")  

    def define_options(self) -> Dict:
        models = get_models(self.api_key)
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
          self.model_id = model_id

        audio_format = st.selectbox(
          "Audio Format",
          [
            "mp3_44100_128",
            "opus_48000_128",
            "pcm_44100",
            "wav_44100"
          ],
          index=0
        )
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
          "output_format": audio_format,
          "model_id": model_id,
          "stability": stability,
          "similarity_boost": simarlity_boost,
          "style": style
        }
        
    def define_voice_explorer(self) -> Dict:
        return self._show_voices(["gender", "age", "accent", "cloned"], model=self.model_id)
      
    def define_usage(self):
      usage = get_usage_percent(self.api_key)
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
      guidance: str | None = None
    ) -> str:
      """Generate audio from a dialogue and save it to a file."""
      audio = generate(text, voice_id, options, self.api_key)
      if "test" in options:
          audio_file = f"./session/{st.session_state.session_id}/temp/test.wav" 
      else:
        audio_file = f"./session/{st.session_state.session_id}/audio/line{line}.wav"
      os.makedirs(os.path.dirname(audio_file), exist_ok=True)
      with open(audio_file, "wb") as f:
        f.write(audio)  
      return audio_file