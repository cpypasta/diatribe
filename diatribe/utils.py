import re, logging, os
import pandas as pd
import numpy as np
import streamlit as st
    
def extract_name(s: str) -> str:
  """Extract the voice name from the voice name with (cloned) suffix."""
  match = re.match(r"(.*?)( \(cloned\))?$", s)
  if match:
    return match.group(1)
  return s  

def start_index_at_one(df: pd.DataFrame) -> None:
  """Start the index of a dataframe at 1."""
  df.index = np.arange(1, len(df) + 1)
  
@st.cache_data
def get_logger():
  log_level = logging.INFO
  logger = logging.getLogger(__name__)
  logger.setLevel(log_level)
  handler = logging.StreamHandler()
  handler.setLevel(log_level)
  formatter = logging.Formatter("%(levelname)s: %(message)s")
  handler.setFormatter(formatter)
  logger.addHandler(handler)
  return logger

def log(message: str) -> None:
  """Log a message to the console."""
  logger = get_logger()
  logger.info(message)
  

def remove_state(key: str) -> None:
  if key in st.session_state:
    del st.session_state[key]

def get_env_key(os_name: str, session_name: str) -> str:
    if os.getenv(os_name):
      value = os.getenv(os_name)
    elif session_name in st.session_state:
      value = st.session_state[session_name]
    else:
      value = ""  
    return value    