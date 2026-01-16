from abc import ABC, abstractmethod
from typing import List, Dict

class AudioProvider(ABC):        
    @abstractmethod
    def get_voice_names(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_voice_id(self, name: str) -> str:
        pass

    @abstractmethod
    def define_creds(self) -> None:
        pass
    
    @abstractmethod
    def define_options(self) -> Dict:
        pass
    
    @abstractmethod
    def define_voice_explorer(self) -> Dict:
        pass
    
    @abstractmethod
    def define_usage(self) -> None:
        pass

    @abstractmethod
    def generate_and_save(
        self,
        text: str,
        voice_id: str,
        line: int,
        options: Dict,
        guidance: str | None = None
    ) -> str:
        pass                