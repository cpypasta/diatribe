
from abc import ABC, abstractmethod
from typing import List, Dict
from diatribe.dialogues import Dialogue

class DialogueProvider(ABC):
    @abstractmethod
    def generate_dialogue(
        self,
        lines: List[Dialogue],
        options: Dict
    ) -> str:
        pass