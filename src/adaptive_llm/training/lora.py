"""Offline CPU LoRA. This is the sole boundary importing the optional training stack.

Base memory estimate: tensor parameters * requested precision bytes * 4 * 1.5.
The four copies budget weights/gradients/Adam moments; 1.5 is a safety factor.
This is an admission estimate, not an OS memory cap. Peak RSS is process-wide.
"""

import hashlib
import hmac
import importlib
import json
import math
import resource
import shutil
import struct
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from adaptive_llm.contracts import (
    AdapterConfig,
    Citation,
    DatasetManifest,
    ModelManifest,
    ResourceUsage,
    TrainingJobSpecification,
    Usage,
)
from adaptive_llm.datasets.artifacts import read_shards
from adaptive_llm.gateway.identity import GatewayError, Keyring
from adaptive_llm.providers import CITATION_PATTERN, ProviderRequest, ProviderResult
from adaptive_llm.storage.crypto import PayloadCipher

CPU_LOCK = RLock()


@dataclass(frozen=True)
class Libraries:
    torch: Any
    transformers: Any
    peft: Any
    tensors: Any


def libraries() -> Libraries:
    try:
        return Libraries(
            importlib.import_module("torch"),
            importlib.import_module("transformers"),
            importlib.import_module("peft"),
            importlib.import_module("safetensors.torch"),
        )
    except (ImportError, OSError):
        raise GatewayError(503, "training_dependencies_unavailable") from None


@contextmanager
def cpu(libs: Libraries, seed: int) -> Iterator[None]:
    # Torch RNG and thread settings are process global; serialize all local tensor work.
    with CPU_LOCK:
        torch = libs.torch
        threads = torch.get_num_threads()
        deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                yield
        finally:
            torch.set_num_threads(threads)
            torch.use_deterministic_algorithms(deterministic)


def encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class BaseFiles:
    files: dict[str, bytes]
    digest: str
    context_limit: int
    parameter_count: int


def base_files(
    data_dir: Path, model_id: str, revision: str, tokenizer_id: str, template: str, licence: str
) -> BaseFiles:
    """Verify the full local inventory before any library sees a path or allocates weights."""
    try:
        if any(Path(v).name != v or v in {".", ".."} for v in (model_id, revision)):
            raise ValueError
        root = data_dir / "base-models"
        path = root / model_id / revision
        if any(p.is_symlink() for p in (root, path.parent, path)):
            raise ValueError
        if not path.is_dir() or any(p.is_symlink() for p in path.rglob("*")):
            raise ValueError
        manifest_bytes = (path / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        if (
            manifest["model_id"] != model_id
            or manifest["revision"] != revision
            or manifest["tokenizer_id"] != tokenizer_id
            or manifest["chat_template_version"] != template
            or manifest["licence"] != licence
        ):
            raise ValueError
        inventory = manifest["files"]
        if {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()} != {
            "manifest.json",
            *inventory,
        }:
            raise ValueError
        if not {"config.json", "LICENSE.txt"} <= set(inventory):
            raise ValueError
        files: dict[str, bytes] = {}
        parameters = 0
        for name, digest in inventory.items():
            relative = Path(name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.suffix not in {".json", ".safetensors", ".txt", ".jinja", ".model"}
            ):
                raise ValueError
            content = (path / relative).read_bytes()
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError
            files[name] = content
            if relative.suffix == ".safetensors":
                size = struct.unpack("<Q", content[:8])[0]
                header = json.loads(content[8 : 8 + size])
                parameters += sum(
                    math.prod(v["shape"]) for k, v in header.items() if k != "__metadata__"
                )
        limit = int(manifest["context_limit"])
        if parameters <= 0 or limit < 2:
            raise ValueError
        return BaseFiles(files, hashlib.sha256(manifest_bytes).hexdigest(), limit, parameters)
    except Exception:
        raise GatewayError(409, "invalid_base_model") from None


def load_base(base: BaseFiles, libs: Libraries, precision: str) -> tuple[Any, Any]:
    # Load only authenticated bytes from a private snapshot, never re-read mutable originals.
    with TemporaryDirectory(prefix="adaptive-base-") as temporary:
        directory = Path(temporary)
        for name, content in base.files.items():
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        tokenizer = libs.transformers.AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False
        )
        model = libs.transformers.AutoModelForCausalLM.from_pretrained(
            directory,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            dtype={
                "fp32": libs.torch.float32,
                "fp16": libs.torch.float16,
                "bf16": libs.torch.bfloat16,
            }[precision],
            attn_implementation="eager",
        ).cpu()
    if not tokenizer.chat_template or tokenizer.eos_token_id is None:
        raise GatewayError(422, "training_configuration_invalid")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return model, tokenizer


def attach(model: Any, config: AdapterConfig, libs: Libraries) -> Any:
    return libs.peft.get_peft_model(
        model,
        libs.peft.LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=config.target_modules,
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            bias="none",
        ),
    )


class TrainingMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class TrainingInputs(BaseModel):
    messages: list[TrainingMessage]
    sources: list[str] = Field(default_factory=list)


class TrainingExample(BaseModel):
    input: TrainingInputs
    target: str


def example_messages(row: TrainingExample) -> list[dict[str, str]]:
    # Shards already contain policy-approved messages (including system/tool roles) and
    # exact-version source boundaries. Do not reconstruct context from a live retriever.
    messages = [
        {"role": m.role, "content": m.content} for m in row.input.messages if m.role != "tool"
    ]
    if row.input.sources:
        messages.append({"role": "user", "content": "\n".join(row.input.sources)})
    messages.extend(
        {"role": m.role, "content": m.content} for m in row.input.messages if m.role == "tool"
    )
    return messages


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


class LoraTrainer:
    architecture = "lora-peft-v1"

    def __init__(
        self,
        data_dir: Path,
        cipher: PayloadCipher,
        keyring: Keyring,
        *,
        memory_limit_bytes: int = 2_000_000_000,
        time_limit_seconds: float = 300,
    ) -> None:
        self.data_dir, self.cipher, self.keyring = data_dir, cipher, keyring
        self.memory_limit_bytes, self.time_limit_seconds = memory_limit_bytes, time_limit_seconds

    def train(
        self,
        specification: TrainingJobSpecification,
        dataset: DatasetManifest,
        directory: Path,
        checkpoint_refs: list[str],
        checkpoint: Callable[[str], None],
        check: Callable[[], None],
    ) -> ResourceUsage:
        try:
            libs = libraries()
            with cpu(libs, specification.seed):
                return self._train(
                    specification, dataset, directory, checkpoint_refs, checkpoint, check, libs
                )
        except GatewayError:
            raise
        except Exception:
            raise GatewayError(503, "training_failed") from None

    def _train(
        self,
        spec: TrainingJobSpecification,
        dataset: DatasetManifest,
        directory: Path,
        checkpoint_refs: list[str],
        checkpoint: Callable[[str], None],
        check: Callable[[], None],
        libs: Libraries,
    ) -> ResourceUsage:
        start = perf_counter()
        if spec.hardware_class != "cpu":
            raise GatewayError(422, "training_configuration_invalid")
        base = base_files(
            self.data_dir,
            spec.base_model_id,
            spec.base_model_revision,
            spec.tokenizer_id,
            spec.chat_template_version,
            spec.base_model_licence,
        )
        precision_bytes = 4 if spec.adapter_config.precision == "fp32" else 2
        estimate = base.parameter_count * precision_bytes * 6
        if estimate > self.memory_limit_bytes:
            raise GatewayError(422, "training_memory_limit")
        if spec.max_sequence_length > base.context_limit:
            raise GatewayError(422, "training_configuration_invalid")
        model, tokenizer = load_base(base, libs, spec.adapter_config.precision)
        model = attach(model, spec.adapter_config, libs)
        model.train()
        model.config.use_cache = False
        rows = read_shards(dataset, self.data_dir, self.cipher, self.keyring)["train"]
        if not rows:
            raise GatewayError(409, "empty_training_split")
        examples: list[tuple[list[int], list[int]]] = []
        for raw in rows:
            row = TrainingExample.model_validate_json(raw)
            messages = example_messages(row)
            prefix = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            full = tokenizer.apply_chat_template(
                [*messages, {"role": "assistant", "content": row.target}], tokenize=True
            )
            if full[: len(prefix)] != prefix:
                raise GatewayError(422, "training_configuration_invalid")
            # Preserve the approved target; truncate the oldest prompt tokens first.
            target = full[len(prefix) :][: spec.max_sequence_length - 1]
            prompt = prefix[-(spec.max_sequence_length - len(target)) :]
            if not prompt or not target:
                raise GatewayError(422, "training_configuration_invalid")
            examples.append((prompt + target, [-100] * len(prompt) + target))
        torch = libs.torch
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=spec.adapter_config.learning_rate,
            foreach=False,
        )
        binding = hashlib.sha256(
            encoded(
                {
                    "specification": spec.model_dump(mode="json", exclude={"job_id"}),
                    "dataset": dataset.content_digest,
                    "base": base.digest,
                }
            )
        ).hexdigest()
        losses: list[float] = []
        tokens_seen = 0
        completed = 0
        refs = {p.name for p in directory.glob("checkpoint-*")}
        if not set(checkpoint_refs) <= refs:
            raise GatewayError(409, "checkpoint_integrity_failed")
        reports: dict[int, tuple[dict[str, Any], dict[str, bytes]]] = {}
        for name in refs:
            report, files = self._read_checkpoint(directory / name, binding)
            step = int(report["steps"])
            if name != f"checkpoint-{step}" or step > spec.steps:
                raise GatewayError(409, "checkpoint_integrity_failed")
            reports[step] = report, files
            if name not in checkpoint_refs:
                checkpoint(name)
        if reports:
            completed = max(reports)
            report, files = reports[completed]
            libs.peft.set_peft_model_state_dict(
                model, libs.tensors.load(files["adapter.safetensors"])
            )
            states = libs.tensors.load(files["optimizer.safetensors"])
            state = optimizer.state_dict()
            for index, _parameter in enumerate(optimizer.param_groups[0]["params"]):
                values = {
                    k: states[f"{index}.{k}"]
                    for k in ("step", "exp_avg", "exp_avg_sq")
                    if f"{index}.{k}" in states
                }
                if values:
                    state["state"][index] = values
            optimizer.load_state_dict(state)
            torch.set_rng_state(states["rng"])
            losses, tokens_seen = report["loss_curve"], report["tokens_seen"]

        def save(step: int) -> None:
            nonlocal reports
            if step in reports:
                return
            adapter = libs.tensors.save(
                {
                    k: v.detach().cpu().contiguous()
                    for k, v in libs.peft.get_peft_model_state_dict(
                        model, save_embedding_layers=False
                    ).items()
                }
            )
            states = {"rng": torch.get_rng_state()}
            for index, values in optimizer.state_dict()["state"].items():
                for key, value in values.items():
                    states[f"{index}.{key}"] = value.cpu().contiguous()
            report = {
                "steps": step,
                "examples": len(rows),
                "loss_curve": losses.copy(),
                "tokens_seen": tokens_seen,
                "base_manifest_digest": base.digest,
                "binding": binding,
                "adapter_digest": hashlib.sha256(adapter).hexdigest(),
            }
            files = {
                "adapter.safetensors": adapter,
                "optimizer.safetensors": libs.tensors.save(states),
                "training_report.json": encoded(report),
            }
            hashes = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
            files["checkpoint.json"] = encoded(
                {"files": hashes, "mac": self.keyring.artifact_mac(encoded(hashes).decode())}
            )
            temp = directory / f".checkpoint-{step}"
            if temp.exists():
                shutil.rmtree(temp)
            temp.mkdir(mode=0o700)
            for name, content in files.items():
                (temp / name).write_bytes(content)
            temp.rename(directory / f"checkpoint-{step}")
            reports[step] = report, files
            checkpoint(f"checkpoint-{step}")

        def boundary(step: int) -> None:
            try:
                check()
                if perf_counter() - start > self.time_limit_seconds:
                    raise GatewayError(503, "training_interrupted")
            except GatewayError:
                save(step)
                raise

        boundary(completed)
        for step in range(completed + 1, spec.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for micro in range(spec.adapter_config.gradient_accumulation):
                offset = (
                    (step - 1) * spec.adapter_config.gradient_accumulation + micro
                ) * spec.batch_size
                selected = [examples[(offset + i) % len(examples)] for i in range(spec.batch_size)]
                width = max(len(ids) for ids, _ in selected)
                ids = torch.tensor(
                    [x + [tokenizer.pad_token_id] * (width - len(x)) for x, _ in selected]
                )
                labels = torch.tensor([y + [-100] * (width - len(y)) for _, y in selected])
                mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in selected])
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                if not bool(torch.isfinite(loss)):
                    raise GatewayError(503, "training_failed")
                (loss / spec.adapter_config.gradient_accumulation).backward()
                total_loss += float(loss.detach()) / spec.adapter_config.gradient_accumulation
                tokens_seen += sum(len(x) for x, _ in selected)
            optimizer.step()
            losses.append(total_loss)
            if step % spec.checkpoint_every == 0 or step == spec.steps:
                save(step)
            boundary(step)
        report, files = reports[spec.steps]
        (directory / "adapter.safetensors").write_bytes(files["adapter.safetensors"])
        (directory / "adapter_config.json").write_bytes(encoded(spec.adapter_config.model_dump()))
        (directory / "training_report.json").write_bytes(
            encoded(
                {
                    key: report[key]
                    for key in (
                        "steps",
                        "examples",
                        "loss_curve",
                        "tokens_seen",
                        "base_manifest_digest",
                        "binding",
                        "adapter_digest",
                    )
                }
            )
        )
        merged = model.merge_and_unload()
        # Clone tied weights so the safetensors export has independent contiguous storage.
        (directory / "merged.safetensors").write_bytes(
            libs.tensors.save(
                {k: v.detach().cpu().contiguous().clone() for k, v in merged.state_dict().items()}
            )
        )
        return ResourceUsage(
            examples=len(rows),
            steps=spec.steps,
            peak_memory_bytes=rss_bytes(),
            artifact_bytes=sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()),
        )

    def _read_checkpoint(self, path: Path, binding: str) -> tuple[dict[str, Any], dict[str, bytes]]:
        try:
            names = {
                "adapter.safetensors",
                "optimizer.safetensors",
                "training_report.json",
                "checkpoint.json",
            }
            if (
                path.is_symlink()
                or {p.name for p in path.iterdir()} != names
                or any(p.is_symlink() for p in path.iterdir())
            ):
                raise ValueError
            files = {name: (path / name).read_bytes() for name in names}
            seal = json.loads(files["checkpoint.json"])
            hashes = {
                k: hashlib.sha256(v).hexdigest() for k, v in files.items() if k != "checkpoint.json"
            }
            if seal["files"] != hashes or not hmac.compare_digest(
                seal["mac"], self.keyring.artifact_mac(encoded(hashes).decode())
            ):
                raise ValueError
            report: dict[str, Any] = json.loads(files["training_report.json"])
            if report["binding"] != binding or len(report["loss_curve"]) != report["steps"]:
                raise ValueError
            return report, files
        except Exception:
            raise GatewayError(409, "checkpoint_integrity_failed") from None


class LoraGenerator:
    """In-memory verified adapter loading and greedy CPU generation for SpecialistProvider."""

    def __init__(self, manifest: ModelManifest, data_dir: Path, files: dict[str, bytes]) -> None:
        self.manifest = manifest
        self.libs = libraries()
        base = base_files(
            data_dir,
            manifest.base_model_id,
            manifest.base_model_revision,
            manifest.tokenizer_id,
            manifest.chat_template_version,
            manifest.base_model_licence,
        )
        try:
            if json.loads(files["training_report.json"])["base_manifest_digest"] != base.digest:
                raise GatewayError(409, "invalid_base_model")
            with cpu(self.libs, manifest.seed):
                model, self.tokenizer = load_base(
                    base, self.libs, manifest.adapter_config.precision
                )
                self.model = attach(model, manifest.adapter_config, self.libs)
                self.libs.peft.set_peft_model_state_dict(
                    self.model, self.libs.tensors.load(files["adapter.safetensors"])
                )
                self.model.eval()
            self.limit = min(base.context_limit, manifest.context_limit)
        except GatewayError:
            raise
        except Exception:
            raise GatewayError(409, "artifact_integrity_failed") from None

    def generate(self, request: ProviderRequest) -> ProviderResult:
        try:
            start = perf_counter()
            with cpu(self.libs, self.manifest.seed), self.libs.torch.inference_mode():
                messages = [{"role": m.role, "content": m.content} for m in request.messages]
                if request.context:
                    messages.append(
                        {
                            "role": "user",
                            "content": "\n".join(
                                f"<<source {c.document_id}/{c.chunk_id} {c.document_version}>>"
                                f"{c.content}<</source>>"
                                for c in request.context
                            ),
                        }
                    )
                tokens = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                tokens = tokens[-(self.limit - 1) :]
                budget = min(request.max_output_tokens, self.limit - len(tokens))
                ids = self.libs.torch.tensor([tokens])
                output = self.model.generate(
                    input_ids=ids,
                    attention_mask=self.libs.torch.ones_like(ids),
                    do_sample=False,
                    max_new_tokens=budget,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
                generated = output[0, len(tokens) :].tolist()
                content = self.tokenizer.decode(generated, skip_special_tokens=True)
            return ProviderResult(
                content=content,
                citations=tuple(
                    Citation(document_id=m[0], chunk_id=m[1])
                    for m in CITATION_PATTERN.findall(content)
                ),
                usage=Usage(
                    input_tokens=len(tokens),
                    output_tokens=len(generated),
                    source="provider_reported",
                    tokenizer=self.manifest.tokenizer_id,
                ),
                finish_reason="stop"
                if generated and generated[-1] == self.tokenizer.eos_token_id
                else "length",
                latency_ms=(perf_counter() - start) * 1000,
            )
        except Exception:
            raise GatewayError(503, "specialist_generation_failed") from None


def generate_tiny_base(directory: Path) -> None:
    """Fixed-seed, random-init two-layer Llama and byte tokenizer; no pretrained assets."""
    libs = libraries()
    with cpu(libs, 17):
        tokenizers = importlib.import_module("tokenizers")
        vocabulary = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
        vocabulary.update(
            {c: i + 3 for i, c in enumerate(sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet()))}
        )
        backend = tokenizers.Tokenizer(
            tokenizers.models.BPE(vocab=vocabulary, merges=[], unk_token="<unk>")
        )
        backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = tokenizers.decoders.ByteLevel()
        tokenizer = libs.transformers.PreTrainedTokenizerFast(
            tokenizer_object=backend,
            pad_token="<pad>",
            eos_token="<eos>",
            unk_token="<unk>",
            model_max_length=512,
        )
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['role'] + ': ' + message['content'] }}"
            "{% if message['role'] == 'assistant' %}{{ eos_token }}{% else %}{{ '\n' }}"
            "{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
        )
        config = libs.transformers.LlamaConfig(
            vocab_size=len(vocabulary),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            bos_token_id=None,
            eos_token_id=1,
            pad_token_id=0,
            tie_word_embeddings=False,
        )
        model = libs.transformers.LlamaForCausalLM(config)
        directory.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(directory, safe_serialization=True)
        tokenizer.save_pretrained(directory)
        (directory / "LICENSE.txt").write_text(
            "Synthetic random tensors and byte vocabulary generated locally for tests. CC0-1.0.\n"
        )
        manifest = {
            "model_id": "tiny",
            "revision": "seed-17",
            "licence": "CC0-1.0",
            "tokenizer_id": "tiny-byte-v1",
            "chat_template_version": "tiny-chat-v1",
            "context_limit": 512,
            "files": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(directory.iterdir())
                if p.name != "manifest.json"
            },
        }
        (directory / "manifest.json").write_bytes(encoded(manifest))
