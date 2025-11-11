from typing import Dict
from dotenv import load_dotenv
import os


class Config:
    def __init__(self, environment: Dict[str, str]) -> None:
        self.__environments: Dict[str, str] = environment
        self.values: Dict[str, str] = dict()
        self.load_from_env()

    def load_from_env(self) -> None:
        load_dotenv()
        for key, friendly_key in self.__environments.items():
            env_value = os.getenv(key)
            if env_value is None:
                raise RuntimeError(f"Missing {key} environment key")
            self[friendly_key] = env_value

    def __setitem__(self, key: str, value: str) -> None:
        if key not in self.__environments.values():
            raise KeyError(f"{key} is not valid")
        self.values[key] = value

    def __getitem__(self, key: str) -> str:
        return self.values[key]
