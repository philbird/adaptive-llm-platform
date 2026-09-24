"""Full or LoRA student updates share the existing authenticated CPU training loop."""

from pathlib import Path

from adaptive_llm.gateway.identity import Keyring
from adaptive_llm.storage.crypto import PayloadCipher
from adaptive_llm.training.lora import LoraTrainer


class StudentTrainer(LoraTrainer):
    def __init__(
        self,
        data_dir: Path,
        cipher: PayloadCipher,
        keyring: Keyring,
        *,
        full: bool,
        memory_limit_bytes: int = 2_000_000_000,
        time_limit_seconds: float = 300,
    ) -> None:
        super().__init__(
            data_dir,
            cipher,
            keyring,
            memory_limit_bytes=memory_limit_bytes,
            time_limit_seconds=time_limit_seconds,
        )
        self.full_student = full
        self.architecture = "student-full-v1" if full else "lora-peft-v1"
