from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

class Gender(Enum):
    MALE = "Male"
    FEMALE = "Female"
    NEUTRAL = "Neutral"

class Source(Enum):
    PROVIDED = "Provided"
    CUSTOM = "Custom"

@dataclass
class AIVoice:
    name: str
    id: str
    path: Path | None = None
    gender: Gender | None = None
    accent: str | None = None
    age: str | None = None
    sample_url: str | None = None
    models: list[str] = field(default_factory=list)
    cloned: bool = field(default=False)
    source: Source = field(default=Source.PROVIDED)