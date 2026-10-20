"""KohakuMossTrainer adapter: MOSS-Image LyCORIS adapters and full finetune.

The clone is a Python package (``kohakumoss``) whose scripts are **kogine** scripts: every
setting is a module-level global, and a config file is a Python module of the same names.
A run therefore generates ``config.py`` and calls
``kogine run <script> --config config.py``.

Caches are a separate preparation: ``scripts/data/cache_text.py`` then
``scripts/data/cache_latents.py`` run before the trainer, like musubi-tuner's stages.
No engine imports here.
"""

import ast
import json
import os
import re
import subprocess
import tempfile
import tomllib
from pathlib import Path

from adapters.base import (
    Artifact,
    DetectionResult,
    EngineAdapter,
    InstallationPlan,
    RecoveryCapabilities,
    ValidationIssue,
)
from adapters.capabilities import check_value, specs
from adapters.kohaku_moss_trainer.progress import parse_progress, runtime_info as read_runtime_info
from adapters.kohaku_moss_trainer.params import ARCHITECTURES, PARAMS
from app.schemas.training import LaunchSpec
from app.services.jobs import JobFiles

TRAIN_SCRIPT = "scripts/train/moss.py"
CACHE_STAGES = ("scripts/data/cache_text.py", "scripts/data/cache_latents.py")
# The engine's own weight layout (src/kohakumoss/weights/layout.py): a ComfyUI models
# folder holds one file per component in these subfolders; an HF pipeline folder uses
# `transformer/`. LP must accept both instead of demanding the HF layout.
COMFY_DIRS = {"transformer": "diffusion_models", "text_encoder": "text_encoders", "vae": "vae"}
COMFY_DEFAULTS = {
    "transformer": ("moss_image_{variant}_int8_convrot", "moss_image_{variant}_bf16"),
    "text_encoder": ("qwen3.5_9b_int8_convrot", "qwen3.5_9b_bf16", "qwen3.5_9b_w4a8_gptq"),
    "vae": ("flux2-vae_small", "flux2-vae"),
}
FULL_FT_DIT = ("moss_image_{variant}_bf16", "moss_image_{variant}_int8_convrot")
WEIGHT_SUFFIX = ".safetensors"
# Tool-generated folders that must never become training images. The engine's `folder`
# source is recursive and has no exclude option, so LP checks and refuses instead of
# silently changing files or the engine's scanner.
TOOL_CACHE_DIRS = frozenset(("cache", ".local-captioner"))
IMAGE_SUFFIXES = frozenset((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"))
# Dict knobs the engine applies **whole** (a layer's value replaces the previous one, no
# deep merge). LP therefore merges the user's keys onto the value the generated config
# inherits instead of dropping every field the user never mentioned.
MERGED_KNOBS = frozenset((
    "SCHEDULE", "TEXT_ENCODE", "VAE_ENCODE", "MEMORY_OVERRIDES", "KERNELS",
    "ALGO_KWARGS", "DROPOUT", "TIMESTEP_KWARGS", "OPTIMIZER_KWARGS", "LOSS_KWARGS",
    "PLANNER_KWARGS", "FP16_FIX_KWARGS",
))
# Knobs the engine only reads with MEMORY_PROFILE=manual (memory/knob_map.py rejects a
# profile combined with them). Listed so a value the user *did* change is never dropped
# silently.
MANUAL_KNOBS = frozenset((
    "CHECKPOINT", "STREAM", "STREAM_RESIDENT", "PINNED", "ACT_OFFLOAD", "OPTIM_STATE",
    "GRAD_OFFLOAD", "FUSED_STEP", "MEMORY_OVERRIDES",
))


class KohakuMossTrainerAdapter(EngineAdapter):
    engine_id = "kohaku_moss_trainer"
    display_name = "KohakuMossTrainer"
    adapter_version = "0.1.0"
    installation_help = (
        "使用引擎自己的 Python 包：建独立 venv 后 pip install -e . 再安装 kogine 运行器；"
        "该引擎的脚本都是 kogine 脚本，缺少运行器无法启动。"
    )
    installation_source_help = "由引擎 pyproject 声明依赖；PyTorch/CUDA wheel 镜像通过 pip 索引注入。"
    # The engine declares these as optional-dependencies; LP installs one only when the
    # user picks it, so the base environment stays the engine's default set.
    optional_extras = (
        ("kernels", "CUDA 内核组件（triton-windows / tilelang；启用 torch.compile 时需要）"),
    )
    default_installation_source = "cu130"
    environment_strategy = "precreate_venv"
    environment_dir = ".venv"
    environment_candidates = (".venv", "venv")
    python_version = "3.12"
    torch_index_mode = "env"
    native_format = "python"
    submittable = True
    creates_output_dir = True
    supports_nested_images = True
    # Import names, measured in a working install: kogine is the CLI of kohaku-engine,
    # so the module to check is kohakuengine (there is no importable "kogine").
    diagnostic_modules = ("torch", "transformers", "safetensors", "lycoris", "kohakuengine")
    clone_url = "https://github.com/KohakuBlueleaf/KohakuMossTrainer.git"
    # Trains from WEIGHTS (a model directory), so the generic base-model file is hidden.
    base_model_optional_architectures = ("moss_image",)
    clone_branch = "main"
    training_notice = "MOSS-Image 专用：先缓存文本与潜空间，再训练；配置以 Python 文件生成。"

    # ---- detection / install ----

    def detect(self, source_path):
        source = Path(source_path)
        try:
            project = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8")).get("project", {})
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
            return DetectionResult(self.engine_id, False, "缺少或无法解析 pyproject.toml")
        matched = project.get("name") == "kohakumoss" and (source / TRAIN_SCRIPT).is_file()
        return DetectionResult(self.engine_id, matched, "kohakumoss 项目元数据与 scripts/train/moss.py")

    def architectures(self, source_path):
        source = Path(source_path)
        return tuple(a for a in ARCHITECTURES if (source / a.script).is_file())

    def installation_sources(self, source_path):
        """CUDA indexes this engine can install torch from (its own guide's mapping).

        Driver CUDA >= 13.0 -> cu130, >= 12.8 -> cu128, >= 12.6 -> cu126. LP ranks them
        against the driver; the chosen tag also seeds the wheel-mirror URLs.
        """
        return ("cu130", "cu128", "cu126")

    def installation_plan(self, source_path, python, torch_source, uv=None, torch_mirror_url=None):
        source = Path(source_path)
        if not self.detect(source).matched:
            raise ValueError("KohakuMossTrainer 源码不完整")
        # Mirrors the engine's own guide: torch first from the index that matches the
        # driver, then the package with that index as an *extra* index so the torch
        # build survives. kogine is the CLI of `kohaku-engine`, a normal dependency —
        # there is no distribution called "kogine".
        existing = torch_source in ("existing", "", None)
        tag = "" if existing else (torch_source if re.fullmatch(r"cu\d{3}", str(torch_source)) else "")
        if not existing and not tag:
            raise ValueError(f"不支持的 CUDA 源：{torch_source}；可用 cu130 / cu128 / cu126")
        index = torch_mirror_url.rstrip("/") if torch_mirror_url else (
            f"https://download.pytorch.org/whl/{tag}" if tag else "")
        stages = []
        if tag:
            stages.append(tuple([python, "-u", "-m", "pip", "install", "--require-virtualenv",
                                 "--no-input", "torch>=2.13,<2.15", "torchvision",
                                 "--index-url", index]))
        package = [python, "-u", "-m", "pip", "install", "--require-virtualenv", "--no-input",
                   "--no-build-isolation"]
        if index:
            # pip has no --index-strategy (that is a uv flag); the extra index is what
            # keeps the already installed cu… torch build.
            package += ["--extra-index-url", index]
        package += ["-e", str(source), "torch>=2.13,<2.15"]
        stages.append(tuple(package))
        return InstallationPlan("KohakuMossTrainer：torch + 可编辑安装（kogine 随 kohaku-engine）",
                                tuple(stages))

    def source_issues(self, source_path):
        source = Path(source_path)
        issues = []
        if not (source / TRAIN_SCRIPT).is_file():
            issues.append("缺少 scripts/train/moss.py 训练入口")
        return tuple(issues)

    # ---- parameters / validation ----

    def capabilities(self, installation):
        return dict(installation_id=installation.installation_id,
                    architectures=[a.id for a in self.architectures(installation.source_path)],
                    params=specs(PARAMS), submittable=self.submittable,
                    base_model_optional_architectures=list(self.base_model_optional_architectures))

    @staticmethod
    def _memory_and_sample_issues(values, explicit=None):
        """Constraints the engine enforces at run time, checked at submit time.

        The engine rejects a profile combined with manual plan knobs, wants one
        seed per sample prompt, requires positive sampling steps/shifts, and needs
        image sizes that are multiples of its 16-pixel latent grid.
        """
        issues = []
        profile = str(values.get("MEMORY_PROFILE") or "").strip()
        if profile and profile != "manual":
            # PINNED has a True default, so comparing values would warn on every profile
            # even when the user never touched the manual knobs. Only values that differ
            # from their defaults -- or that the user really did edit -- are meaningful.
            blocked = [key for key in MANUAL_KNOBS
                       if str(values.get(key) or "").strip() not in ("", "{}", "False", "0")]
            if explicit is not None:
                blocked += [key for key in sorted(MANUAL_KNOBS - set(blocked)) if key in explicit]
            if blocked:
                # The engine ignores these under a profile, so this is a hint, not a
                # reason to refuse submission (a warning is never blocking).
                issues.append(ValidationIssue(
                    "MEMORY_PROFILE",
                    f"显存档位为 {profile} 时这些手动项不会生效：{', '.join(blocked)}；"
                    "要使用它们请把档位改成 manual，否则可以留空",
                    "warning"))
        try:
            prompts = ast.literal_eval(str(values.get("SAMPLE_PROMPTS") or "[]"))
            seeds = ast.literal_eval(str(values.get("SAMPLE_SEEDS") or "(0,)"))
        except (ValueError, SyntaxError):
            prompts, seeds = None, None
        if isinstance(prompts, (list, tuple)) and isinstance(seeds, (list, tuple)) and prompts:
            if len(seeds) != len(prompts):
                issues.append(ValidationIssue(
                    "SAMPLE_SEEDS", f"引擎要求每个提示词一个种子：{len(prompts)} 个提示词，{len(seeds)} 个种子"))
        for key in ("SAMPLE_STEPS", "SAMPLE_SHIFT"):
            try:
                if float(values.get(key)) <= 0:
                    issues.append(ValidationIssue(key, "必须为正数"))
            except (TypeError, ValueError):
                pass
        # memory/profiles.py: "weight streaming needs stream_resident >= 2".
        if str(values.get("STREAM") or "").strip() in ("ring", "swap"):
            try:
                resident = int(values.get("STREAM_RESIDENT") or 0)
            except (TypeError, ValueError):
                resident = 0
            if resident < 2:
                issues.append(ValidationIssue(
                    "STREAM_RESIDENT", "流式模式下引擎要求设备槽位 >= 2（当前为空或 0）"))
        try:
            size = ast.literal_eval(str(values.get("SAMPLE_SIZE") or "(1024, 1024)"))
            if isinstance(size, (list, tuple)) and len(size) == 2:
                if any(int(v) <= 0 or int(v) % 16 for v in size):
                    issues.append(ValidationIssue("SAMPLE_SIZE", "尺寸必须是 16 的正整数倍"))
        except (ValueError, SyntaxError, TypeError):
            issues.append(ValidationIssue("SAMPLE_SIZE", "请填写 Python 元组，如 (1024, 1024)"))
        return issues

    @staticmethod
    def _weights_issue(weights: str) -> ValidationIssue | None:
        """The engine's WEIGHTS accepts a ComfyUI models folder, an HF pipeline folder,
        or one DiT safetensors file (weights/layout.py)."""
        root = Path(weights)
        if not root.exists():
            return ValidationIssue("WEIGHTS", f"模型目录不存在：{weights}")
        if root.is_file():
            if root.suffix.lower() != WEIGHT_SUFFIX:
                return ValidationIssue("WEIGHTS", "WEIGHTS 文件必须是 .safetensors")
            return ValidationIssue(
                "WEIGHTS",
                "已将 WEIGHTS 作为单个 DiT 文件；文本编码器与 VAE 需由引擎默认设置解析，"
                "如启动时找不到，请改用 ComfyUI 模型目录或显式填写 TEXT_ENCODER / VAE",
                "warning")
        if (root / "transformer").is_dir():
            return None
        if any(any((root / COMFY_DIRS[name]).glob(f"*{WEIGHT_SUFFIX}")) for name in COMFY_DIRS):
            missing = [COMFY_DIRS[name] for name in COMFY_DIRS
                       if not any((root / COMFY_DIRS[name]).glob(f"*{WEIGHT_SUFFIX}"))]
            if missing:
                return ValidationIssue(
                    "WEIGHTS",
                    f"ComfyUI 模型目录缺少组件文件：{', '.join(missing)} 下没有 "
                    f"*{WEIGHT_SUFFIX}（diffusion_models / text_encoders / vae 都要有）")
            return None
        return ValidationIssue(
            "WEIGHTS",
            "模型目录不是引擎支持的布局：需要 ComfyUI 的 diffusion_models/ + text_encoders/ + vae/，"
            "或 HF 的 transformer/ 目录，或单个 DiT .safetensors")

    @staticmethod
    def _tool_cache_images(dataset_path: str) -> list[str]:
        """Image files under tool cache folders inside a dataset (non-destructive check)."""
        root = Path(dataset_path)
        if not root.is_dir():
            return []
        found = []
        for directory, dirs, files in os.walk(root):
            relative = Path(directory).relative_to(root)
            inside = any(part.lower() in TOOL_CACHE_DIRS for part in relative.parts)
            if not inside:
                continue
            for name in files:
                if Path(name).suffix.lower() in IMAGE_SUFFIXES:
                    found.append(str(Path(directory) / name))
                    if len(found) >= 5:
                        return found
        return found

    def validate_dataset(self, installation, dataset, images):
        """Refuse datasets whose tool caches would be scanned as training images.

        The engine's `folder` source is recursive and `kohya` treats every first-level
        folder as a subset, so a `cache/` or `.local-captioner/` holding images would be
        trained on. The engine has no exclude option: report the problem and let the user
        move those files (LP never modifies the dataset).
        """
        found = self._tool_cache_images(dataset["path"])
        if not found:
            return ()
        shown = "；".join(found[:3])
        return (ValidationIssue(
            "dataset_id",
            f"数据集里的工具缓存目录包含图片，会被引擎当作训练图：{shown}"
            f"{' 等' if len(found) >= 5 else ''}。请先把 cache/ 或 .local-captioner/ 中的图片移出数据集"
            "（LP 不会自动删除或移动任何文件）"),)

    def validate(self, installation, config):
        issues = []
        architecture = config.get("architecture")
        if architecture not in {a.id for a in self.architectures(installation.source_path)}:
            issues.append(ValidationIssue("architecture", "当前源码没有对应的训练入口"))
        values = config.get("params", {})
        known = {p.key for p in PARAMS}
        for key in values.keys() - known:
            issues.append(ValidationIssue(key, "不支持的参数"))
        # The engine derives the text encoder / VAE from WEIGHTS, so an empty WEIGHTS is
        # unusable even though an empty string is a valid type: refuse it at submit time
        # instead of failing later inside cache_text.py.
        weights = str(values.get("WEIGHTS") or "").strip()
        if not weights:
            issues.append(ValidationIssue(
                "WEIGHTS",
                "该引擎需要 MOSS-Image 权重：ComfyUI 模型目录（diffusion_models/、text_encoders/、vae/）、"
                "HF 目录（含 transformer/），或单个 DiT safetensors"))
        else:
            issue = self._weights_issue(weights)
            if issue:
                issues.append(issue)
        issues.extend(self._memory_and_sample_issues(values, self.explicit_keys(config)))
        issues.extend(self._schedule_issues({p.key: values.get(p.key, p.default) for p in PARAMS}))
        # `kohakumoss-config new train --vram` takes a GiB budget or "auto"; an empty value
        # means the flag is not passed at all (the generator then picks the card's layer).
        vram = str(values.get("VRAM") or "").strip()
        if vram and vram.lower() != "auto":
            try:
                budget = float(vram)
            except ValueError:
                budget = 0.0
            if budget <= 0:
                issues.append(ValidationIssue(
                    "VRAM", "显存预算请填 auto、留空，或正数 GiB（如 12 / 11.5）；留空 = 官方按本机显存选层"))
        for p in PARAMS:
            if not p.native:
                continue
            value = values.get(p.key, p.default)
            if p.key in ("ALGO_KWARGS", "DROPOUT", "BETAS", "SCHEDULE", "TIMESTEP_KWARGS",
                         "CAPTION", "DEVICES"):
                try:
                    json.loads(value if isinstance(value, str) and value.strip() else "null")
                except ValueError:
                    issues.append(ValidationIssue(p.key, f"{p.label} 需要合法的 JSON"))
                continue
            message = check_value(p, value)
            if message:
                issues.append(ValidationIssue(p.key, message))
        return tuple(issues)

    # ---- native config (Python module) ----

    @staticmethod
    def _literal(value):
        """A Python literal for the generated config (JSON text becomes a real object)."""
        if isinstance(value, str):
            text = value.strip()
            if text[:1] in "{[(":
                # JSON first, then a Python literal: both spellings are accepted in the
                # form, and a container must never end up quoted as a string.
                for parse in (json.loads, ast.literal_eval):
                    try:
                        return repr(parse(text))
                    except (ValueError, SyntaxError):
                        continue
            return repr(value)
        return repr(value)

    @staticmethod
    def _dataset_entry(config):
        """One DATASETS entry of the user's chosen engine source.

        `folder` scans the whole tree recursively; `kohya` only first-level
        `<repeats>_<name>/` folders. Both take `txt_as`; `weight` is a dataset-level
        key the engine strips before constructing the source, so it is safe for both.
        """
        params = config.get("params", {})
        source = str(params.get("DATASET_SOURCE") or "folder").strip()
        if source not in ("folder", "kohya"):
            source = "folder"
        return {
            "source": source,
            "path": str(config["dataset_path"]),
            "weight": 1.0,
            # sources/base.py:37-47 -- `.txt` goes to tags or to caption_nl.
            "txt_as": str(params.get("TXT_AS") or "tags"),
        }

    @staticmethod
    def _parse_mapping(text):
        """A dict from JSON or a Python literal; empty dict on anything else."""
        for parse in (json.loads, ast.literal_eval):
            try:
                parsed = parse(str(text or "").strip() or "{}")
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, dict):
                return parsed
        return {}

    @staticmethod
    def _warmup(values):
        """The Warmup input as a non-negative int; None = keep SCHEDULE's own warmup."""
        raw = values.get("WARMUP")
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return None
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _with_warmup(schedule, warmup):
        """`schedule` with `warmup` written into its lr sub-schedule (a copy)."""
        merged = dict(schedule)
        if "mode" in merged:
            merged["warmup"] = warmup
        elif isinstance(merged.get("lr"), dict):
            merged["lr"] = {**merged["lr"], "warmup": warmup}
        return merged

    def _schedule(self, values):
        """SCHEDULE with the separate Warmup input written into its lr sub-schedule.

        The engine's SCHEDULE is either the lr schedule itself (`{"mode": ...}`) or a
        mapping of sub-schedules (`{"lr": {...}, "ema": {...}}`); `warmup` lives inside
        each one (training/optim/schedule.py), so only `lr` is touched here and every
        other field of SCHEDULE survives untouched.

        Used by the validation rules; rendering merges onto the generated baseline
        instead (see `_render_values`).
        """
        schedule = dict(self._parse_mapping(values.get("SCHEDULE")))
        warmup = self._warmup(values)
        if warmup is None:
            return schedule
        return self._with_warmup(schedule, warmup)

    def _schedule_issues(self, values):
        """Warmup must fit inside the lr schedule: the engine never leaves it otherwise."""
        schedule = self._parse_mapping(values.get("SCHEDULE"))
        raw = values.get("WARMUP")
        warmup = self._warmup(values)
        if warmup is None:
            # Empty is the normal state: SCHEDULE's own warmup applies.
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                return ()
            return (ValidationIssue("WARMUP", "Warmup 必须是非负整数"),)
        lr = schedule if "mode" in schedule else schedule.get("lr")
        if not isinstance(lr, dict):
            return ()
        raw_end = lr.get("end", -1)
        end = raw_end
        if raw_end == -1:
            try:
                end = int(values.get("STEPS"))
            except (TypeError, ValueError):
                return ()
        if isinstance(end, int) and warmup >= end:
            bound = "STEPS" if raw_end == -1 else "SCHEDULE['end']"
            return (ValidationIssue(
                "WARMUP",
                f"Warmup {warmup} >= 学习率调度结束步 {end}（{bound}）：学习率会全程停在 "
                f"warmup 段，请把 Warmup 调小（约步数的 1/10）或增大步数"),)
        return ()

    def _text_plan(self, values, explicit=None):
        """The cache pass's TEXT_ENCODER_KWARGS["plan"].

        The cache stage has its own baseline (`PRECOMPUTE_DEFAULTS`: memory auto,
        batch 32, max_tokens 16384), and `text_encoder_kwargs` only carries TEXT_ENCODE's
        *compute* fields into it. So only the fields the user changed are written here,
        plus LP's `pinned: False` policy: the engine's TextEncoderPlan.pinned defaults to
        True and copies the encoder into page-locked host memory, which failed with
        CUDA_ERROR_OUT_OF_MEMORY (cuMemHostAlloc) on this 32 GB host.
        """
        keys = set(explicit) if explicit is not None else {"TEXT_ENCODE", "TEXT_ENCODER_PLAN"}
        plan = {"pinned": False}
        if "TEXT_ENCODE" in keys:
            plan.update(self._parse_mapping(values.get("TEXT_ENCODE")))
        if "TEXT_ENCODER_PLAN" in keys:
            plan.update(self._parse_mapping(values.get("TEXT_ENCODER_PLAN")))
        return plan

    @classmethod
    def _layer_assignments(cls, path, out=None, seen=None):
        """Every literal module-level assignment of a generated config *and* its layers.

        `kohakuengine.use_config` layers the referenced files under the config's own
        values ("later calls win over earlier", "your own top-level variables win"), but
        applies them *after* the module body runs. A dict knob replaced by a later layer
        therefore loses the previous value whole, which LP cannot see from the file it
        appends to -- so the inherited values are read back here statically.
        """
        out = {} if out is None else out
        seen = set() if seen is None else seen
        try:
            resolved = Path(path).resolve()
            source = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return out
        if resolved in seen:
            return out
        seen.add(resolved)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return out
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                name = call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", "")
                if name != "use_config" or not call.args:
                    continue
                argument = call.args[0]
                if not (isinstance(argument, ast.Constant) and isinstance(argument.value, str)):
                    continue
                layer = Path(argument.value)
                cls._layer_assignments(layer if layer.is_absolute() else resolved.parent / layer,
                                       out, seen)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    try:
                        out[target.id] = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError, TypeError):
                        # An expression LP cannot read statically: never guess a baseline.
                        out.pop(target.id, None)
        return out

    @classmethod
    def inherited_values(cls, path):
        """The dict knobs the generated config inherits from its `use_config` layers."""
        if not path:
            return {}
        try:
            assignments = cls._layer_assignments(path)
        except (OSError, RecursionError):
            return {}
        return {key: value for key, value in assignments.items()
                if key in MERGED_KNOBS and isinstance(value, dict)}

    @staticmethod
    def explicit_keys(config):
        """Which knobs the user actually changed.

        Every form field starts at its LP default, so "present in params" says nothing
        about intent — only `explicit_params` does. Clients that never send it (the HTTP
        API called by scripts) submit exactly the keys they mean, so a missing list falls
        back to "every submitted key is an override".
        """
        raw = config.get("explicit_params")
        if isinstance(raw, (list, tuple, set)):
            return {str(key) for key in raw}
        return set((config.get("params") or {}).keys())

    def _render_values(self, config, baseline=None):
        """The user's overrides only; the official generated config is the baseline.

        `kohakumoss-config new train` already wrote every default into the job's config
        (GPU-selected VRAM layer, memory profile, precision, kernels, sampling). Repeating
        LP's own defaults here silently replaced those official values -- e.g.
        BASE_PRECISION 'compute' dequantized the INT8 ConvRot bases -- so only keys the
        user actually changed (`explicit_params`) are appended.

        `baseline` holds the dict knobs the generated config inherits from its
        `use_config` layers. Those knobs are replaced whole by the engine, so the user's
        keys are merged onto that value instead of dropping every field the user never
        saw (TEXT_ENCODE's placement, SCHEDULE's min_value, ...).
        """
        values = {p.key: config.get("params", {}).get(p.key, p.default) for p in PARAMS}
        explicit = self.explicit_keys(config)
        baseline = dict(baseline or {})
        output = Path(config["output_dir"])
        # The job's own inputs and paths must be written even when untouched: the
        # generated file carries the template's placeholder weights / dataset paths and
        # a run directory relative to the engine checkout.
        lines = [
            f"WEIGHTS = {self._literal(values.get('WEIGHTS') or '')}",
            "DATASETS = " + repr([self._dataset_entry(config)]),
            f"OUTPUT_DIR = {self._literal(str(output))}",
            f"RUN_NAME = {self._literal(config.get('output_name') or config.get('name') or 'moss')}",
        ]
        # The engine refuses a profile together with the manual plan knobs: with a
        # profile they must stay None. Cache-stage-only knobs belong to the data config.
        profile = str(values.get("MEMORY_PROFILE") or "").strip()
        skip = {"CAPTION", "SCHEDULE", "TEXT_ENCODE", "TEXT_ENCODER_PLAN",
                "TEXT_ENCODER_KWARGS", "IMAGE_ENCODER_KWARGS", "IMAGE_ENCODER",
                "ENCODE_BATCH"}
        if profile and profile != "manual":
            skip |= set(MANUAL_KNOBS)
        for p in PARAMS:
            key = p.key
            if not p.native or key not in explicit or key in skip:
                continue
            value = values.get(key)
            if value is None or value == "":
                continue
            if key == "DEVICES":
                text = str(value).strip().strip("[]()")
                indices = [part.strip() for part in text.split(",") if part.strip().isdigit()]
                lines.append("DEVICES = (" + ", ".join(indices or ["0"]) + ",)")
                continue
            if key == "DATA_RESCAN":
                # Source: kohakumoss_data/scan/runner.py -- (False, True, "never").
                text = str(value).strip().lower()
                lines.append("DATA_RESCAN = " + ("'never'" if text == "never"
                                                  else ("True" if text == "true" else "False")))
                continue
            if key in MERGED_KNOBS:
                inherited = baseline.get(key)
                merged = {**(inherited if isinstance(inherited, dict) else {}),
                          **self._parse_mapping(value)}
                lines.append(f"{key} = " + repr(merged))
                continue
            lines.append(f"{key} = {self._literal(value)}")
        # A sub-schedule map ({"lr": ..., "ema": ...}) replaces SCHEDULE instead of being
        # merged into a flat schedule: mixing the two shapes would leave both "mode" and
        # "lr" in one dict, which the engine reads as the flat form.
        if "SCHEDULE" in explicit or "WARMUP" in explicit:
            schedule = dict(baseline.get("SCHEDULE") or {})
            chosen = self._parse_mapping(values.get("SCHEDULE")) if "SCHEDULE" in explicit else {}
            if chosen and "mode" not in chosen:
                schedule = dict(chosen)
            else:
                schedule.update(chosen)
            warmup = self._warmup(values) if "WARMUP" in explicit else None
            if warmup is not None:
                schedule = self._with_warmup(schedule, warmup)
            if schedule:
                lines.append("SCHEDULE = " + repr(schedule))
        # LP policy: TextEncoderPlan.pinned defaults to True and copies the encoder into
        # page-locked host memory (cuMemHostAlloc -> CUDA OOM on this 32 GB host). Only
        # `pinned` is set here, on top of the official value, so memory / ahead / batch /
        # max_tokens stay exactly what the selected VRAM layer chose.
        inherited_plan = baseline.get("TEXT_ENCODE")
        if isinstance(inherited_plan, dict) or "TEXT_ENCODE" in explicit:
            plan = dict(inherited_plan) if isinstance(inherited_plan, dict) else {}
            if "TEXT_ENCODE" in explicit:
                plan.update(self._parse_mapping(values.get("TEXT_ENCODE")))
            plan.setdefault("pinned", False)
            if plan != (inherited_plan or {}):
                lines.append("TEXT_ENCODE = " + repr(plan))
        # CAPTION is whole-replace by design: the form holds one complete template, and
        # the engine reads the keys of the chosen template only.
        caption = values.get("CAPTION")
        if "CAPTION" in explicit and caption:
            lines.append(f"CAPTION = {self._literal(caption)}")
        extra = str(config.get("params", {}).get("extra_params") or "").strip()
        if extra:
            lines.append("# 自定义参数")
            lines.extend(extra.splitlines())
        return lines

    # ---- native config (Python module) ----

    def _render(self, config, baseline=None):
        lines = ['"""LP native overrides for a kogine config.',
                 '',
                 'Only the fields the user changed are written here. The baseline is the',
                 'official `kohakumoss-config new train` output: its GPU-selected VRAM',
                 'layer, memory profile, precision and kernel defaults stay in force for',
                 'every knob that is not listed below.',
                 '"""']
        lines += self._render_values(config, baseline)
        return "\n".join(lines) + "\n"

    def _render_cache_values(self, config):
        """Cache-stage overrides, appended to the official `_data.py` config.

        The generated data config already defines TEXT_ENCODER_KWARGS / IMAGE_ENCODER /
        the encoder paths, so the appended lines merge a single key into that dict instead
        of rewriting it: `path` and `device` stay exactly what the generator wrote, and
        the cache pass keeps its own baseline (`PRECOMPUTE_DEFAULTS`).
        """
        values = {p.key: config.get("params", {}).get(p.key, p.default) for p in PARAMS}
        explicit = self.explicit_keys(config)
        lines = []
        plan = self._text_plan(values, explicit)
        update = {"plan": plan} if plan else {}
        if "TEXT_DEVICE" in explicit:
            update["device"] = str(values.get("TEXT_DEVICE") or "").strip() or "cuda"
        if update:
            merged = ", ".join(f"{key!r}: {value!r}" for key, value in update.items())
            lines.append(f"TEXT_ENCODER_KWARGS = {{**TEXT_ENCODER_KWARGS, {merged}}}")
        if "IMAGE_ENCODER" in explicit:
            factory = str(values.get("IMAGE_ENCODER") or "").strip()
            if factory:
                lines.append(f"IMAGE_ENCODER = {self._literal(factory)}")
        if "ENCODE_BATCH" in explicit:
            lines.append("ENCODE_BATCH = " + repr(int(values.get("ENCODE_BATCH") or 512)))
        return lines

    def native_config_text(self, config, dataset_config):
        return self._render(config)

    def preview_native_config_text(self, installation, config, dataset_config):
        """The official config shown before launch, not just LP's standalone rendering.

        This method is allowed to execute the engine's own generator (unlike the pure
        `native_config_text` contract); it is optional, and validate falls back to
        `native_config_text` when it is absent or fails.
        """
        if self._config_tool(installation) is None:
            return self._render(config)
        with tempfile.TemporaryDirectory(prefix="moss-preview-") as tmp:
            target = Path(tmp) / "config.py"
            ok, reason = self._generate_official(installation, config, target)
            if not ok:
                return (f"# LP：官方配置生成失败（{reason}）；以下为 LP 渲染的等价配置\n"
                        + self._render(config))
            data_config = target.with_name(f"{target.stem}_data.py")
            baseline = self.inherited_values(target)
            text = target.read_text(encoding="utf-8")
            text += ("\n\n# ---- LP 覆盖：只列出用户改动过的字段；其余沿用上面的官方配置 ----\n"
                     "# 官方生成器：kohakumoss-config new train（本机显存自动选层 / 官方显存档位）\n"
                     + "\n".join(self._render_values(config, baseline)) + "\n")
            text += ("\n\n# ---- 数据准备配置（cache_text.py / cache_latents.py 使用） ----\n"
                     + data_config.read_text(encoding="utf-8"))
            cache = self._render_cache_values(config)
            if cache:
                text += ("\n# ---- LP 覆盖：缓存阶段（只改用户动过的项） ----\n"
                         + "\n".join(cache) + "\n")
            return text

    def export_parameters(self, config):
        if config.get("architecture") not in {a.id for a in ARCHITECTURES}:
            raise ValueError("当前底模类型不受 KohakuMossTrainer 适配器支持")
        notice = "此文件是 kogine 的 Python 配置；运行时还需同目录的缓存阶段与 kogine 运行器。"
        return "# " + notice + "\n" + self._render(config), (notice,)

    # ---- official config generation ----

    @staticmethod
    def _config_tool(installation):
        python = getattr(installation, "python_executable", None)
        if not python:
            return None
        name = "kohakumoss-config.exe" if os.name == "nt" else "kohakumoss-config"
        tool = Path(python).parent / name
        return tool if tool.is_file() else None

    def _generate_official(self, installation, config, target):
        """Write the engine's own train + data configs into `target`'s directory.

        `kohakumoss-config new train` detects the GPU, picks the official VRAM preset and
        renders both configs. LP then appends only the user's overrides, so the inherited
        official fields stay intact. Returns (ok, reason).
        """
        tool = self._config_tool(installation)
        if tool is None:
            return False, "未找到引擎的 kohakumoss-config"
        params = config.get("params", {})
        weights = str(params.get("WEIGHTS") or "").strip()
        dataset = str(config.get("dataset_path") or "").strip()
        if not weights or not dataset:
            return False, "缺少 WEIGHTS 或数据集路径"
        source = str(params.get("DATASET_SOURCE") or "folder").strip()
        if source not in ("folder", "kohya"):
            source = "folder"
        algo = str(params.get("ALGO") or "lora").strip() or "lora"
        try:
            devices = json.loads(str(params.get("DEVICES") or "[0]"))
            gpus = max(1, len(devices)) if isinstance(devices, list) else 1
        except ValueError:
            gpus = 1
        name = re.sub(r"[^A-Za-z0-9._-]+", "-",
                      str(config.get("output_name") or config.get("name") or "moss")).strip("-._")
        command = [str(tool), "new", "train", "--weights", weights, "--data", dataset,
                   "--source", source, "--algo", algo, "--gpus", str(gpus),
                   "--name", name or "moss"]
        # No `--vram` by default: the generator detects this machine's card and composes
        # the matching official layer (a 12 GB GPU gets vram/vram_12g.py). "auto" is the
        # dynamic memory plan, which needs more VRAM, so it is passed only when the user
        # asked for it; a number is a manual budget in GiB.
        vram = str(params.get("VRAM") or "").strip()
        if vram:
            command += ["--vram", vram]
        command += ["--out", str(target), "--force"]
        precision = str(params.get("PRECISION") or "").strip()
        if precision in ("bf16", "fp16"):
            command += ["--precision", precision]
        try:
            result = subprocess.run(command, cwd=str(installation.source_path),
                                    capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", timeout=300)
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"kohakumoss-config 执行失败：{exc}"
        data_config = target.with_name(f"{target.stem}_data.py")
        if result.returncode or not target.is_file() or not data_config.is_file():
            tail = " ".join((result.stderr or result.stdout or "").strip().splitlines()[-2:])
            return False, f"kohakumoss-config 返回 {result.returncode}：{tail[:300]}"
        return True, ""

    def write_native_config(self, installation, config, job_directory):
        from app.services.common import atomic_text
        job_directory = Path(job_directory)
        target = job_directory / "config.py"
        ok, reason = self._generate_official(installation, config, target)
        if ok:
            data_config = target.with_name(f"{target.stem}_data.py")
            # What the generated file inherits from its `use_config` layers, so a dict
            # knob (TEXT_ENCODE / SCHEDULE / VAE_ENCODE ...) can be merged instead of
            # replacing the official value whole.
            baseline = self.inherited_values(target)
            header = ("\n\n# ---- LP 覆盖：只列出用户改动过的字段；其余沿用上面的官方配置 ----\n"
                      "# 官方生成器：kohakumoss-config new train（本机显存自动选层 / 官方显存档位）\n")
            atomic_text(target, target.read_text(encoding="utf-8") + header
                        + "\n".join(self._render_values(config, baseline)) + "\n")
            cache = self._render_cache_values(config)
            if cache:
                atomic_text(data_config, data_config.read_text(encoding="utf-8")
                            + "\n\n# ---- LP 覆盖：缓存阶段编码器 ----\n"
                            + "\n".join(cache) + "\n")
            # The generator checks the inherited config, but the LP overrides are appended
            # afterwards. Run the engine's own checker on the final files so a bad override
            # fails submission instead of minutes into the run.
            tool = self._config_tool(installation)
            if tool is not None:
                try:
                    checked = subprocess.run(
                        [str(tool), "check", str(target), str(data_config)],
                        cwd=str(installation.source_path), capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=300)
                except (OSError, subprocess.SubprocessError) as exc:
                    raise RuntimeError(f"官方配置检查无法执行：{exc}") from exc
                if checked.returncode:
                    tail = " ".join((checked.stderr or checked.stdout or "").strip().splitlines()[-3:])
                    raise RuntimeError(f"官方配置检查失败：{tail[:500]}")
        else:
            notice = (f"# LP：官方 kohakumoss-config 不可用（{reason}），"
                      "以下为 LP 渲染的等价配置；提交日志中会保留该原因。\n")
            atomic_text(target, notice + self._render(config))
        stages = self.preview_commands(installation, config, target)
        atomic_text(job_directory / "stages.json",
                    json.dumps([list(stage) for stage in stages], ensure_ascii=False, indent=2))
        return target

    # ---- launch ----

    def _values(self, config):
        params = config.get("params", {})
        return {p.key: params.get(p.key, p.default) for p in PARAMS}

    def preview_commands(self, installation, config, native_config):
        python = installation.python_executable
        if not python:
            raise RuntimeError("该引擎尚未绑定 Python 解释器，无法启动 kogine")
        kogine = Path(python).parent / ("kogine.exe" if os.name == "nt" else "kogine")
        runner = str(kogine if kogine.is_file() else "kogine")
        source = Path(installation.source_path)
        config_path = str(native_config)
        # After submission the official generator also wrote `<stem>_data.py`; the cache
        # stages then use it (the engine's documented workflow). The preview path has no
        # files yet, so it falls back to --set on the train config.
        data_config = Path(native_config).with_name(f"{Path(native_config).stem}_data.py")
        values = {p.key: config.get("params", {}).get(p.key, p.default) for p in PARAMS}
        if data_config.is_file():
            stages = []
            for script in CACHE_STAGES:
                stages.append((runner, "run", str(source / script),
                               "--config", str(data_config)))
            stages.append((runner, "run", str(source / TRAIN_SCRIPT), "--config", config_path))
            return tuple(stages)
        # Cache-only knobs: the trainer's own check_script rejects them, so they are
        # passed with --set on the two preparation stages (cache_text.py:3-4 documents it).
        # The path is the WEIGHTS root: the engine resolves `text_encoder/` (HF) or the
        # file under `text_encoders/` (ComfyUI) itself.
        weights = str(values.get("WEIGHTS") or "").strip()
        text_plan = self._text_plan(values, self.explicit_keys(config))
        cache_sets = [
            "--set", "TEXT_ENCODER_KWARGS=" + repr({
                "path": weights, "device": str(values.get("TEXT_DEVICE") or "").strip() or "cuda",
                "plan": text_plan}),
            "--set", "IMAGE_ENCODER_KWARGS=" + repr({
                "path": weights, "device": "cuda"}),
            # IMAGE_ENCODER exists only in cache_latents.py, never in the trainer.
            # kogine takes the value as-is: a repr() here would arrive as a quoted string.
            "--set", "IMAGE_ENCODER=" + str(values.get("IMAGE_ENCODER")
                                            or "kohakumoss.encoders.vae_encoder"),
            "--set", "ENCODE_BATCH=" + repr(int(values.get("ENCODE_BATCH") or 512)),
        ]
        stages = []
        for script in CACHE_STAGES:
            stages.append((runner, "run", str(source / script), "--config", config_path, *cache_sets))
        stages.append((runner, "run", str(source / TRAIN_SCRIPT), "--config", config_path))
        return tuple(stages)

    def preview_argv(self, installation, config, native_config):
        return self.preview_commands(installation, config, native_config)[-1]

    def build_launch(self, installation, native_config):
        native_config = Path(native_config)
        stages_path = native_config.parent / "stages.json"
        if not stages_path.is_file():
            raise RuntimeError("缺少 KohakuMossTrainer 运行阶段定义 stages.json")
        commands = tuple(tuple(stage) for stage in json.loads(stages_path.read_text(encoding="utf-8")))
        if not commands:
            raise RuntimeError("KohakuMossTrainer 运行阶段为空")
        return LaunchSpec(commands[-1], Path(installation.source_path),
                          {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}, commands[:-1])

    # ---- logs / artifacts ----

    def parse_log(self, line):
        # Its callbacks print `step N [loss=…]` and `steps A-B median X s/step`.
        return parse_progress(line)

    @staticmethod
    def runtime_info(lines):
        """The knobs the engine resolved itself and printed, read from its own lines.

        Optional hook (`GET /tasks/{id}/runtime`): the `auto` VRAM profile's outcome, the
        encoder placements and the VRAM peaks exist nowhere else, so LP reports exactly
        what the log stated and nothing more.
        """
        return read_runtime_info(lines)

    def recovery_capabilities(self, installation):
        return RecoveryCapabilities(graceful_stop=True, resume_weights=True)

    @staticmethod
    def _export_formats(value):
        """The adapter-only export formats of a stored job value.

        jobs store the (str, int, bool) draft, so a tuple arrives as its literal text.
        """
        if isinstance(value, str):
            text = value.strip()
            for parse in (json.loads, ast.literal_eval):
                try:
                    parsed = parse(text)
                except (ValueError, SyntaxError):
                    continue
                value = parsed
                break
            else:
                value = (text,) if text else ()
        if isinstance(value, str):
            value = (value,)
        return tuple(str(item) for item in (value or ()) if str(item) in ("comfy", "lycoris"))

    def finalize_artifacts(self, job_directory, python, source_path):
        """Export every saved checkpoint as a usable LoRA.

        The engine exports only the final state (training/entry.py:335), so per-step
        usable files are produced here with the engine's own exporter
        (scripts/export/export.py, ADAPTER = a run checkpoint's trainable.safetensors).
        A failed export is a notice: the run itself stays successful.
        """
        config = JobFiles(job_directory).read_job()
        if config.get("architecture") != "moss_image":
            return ()
        values = {p.key: config.get("params", {}).get(p.key, p.default) for p in PARAMS}
        weights = str(values.get("WEIGHTS") or "").strip()
        root = Path(config["output_dir"])
        # The engine nests OUTPUT_DIR/<RUN_NAME>/checkpoints, so search recursively.
        checkpoints = sorted(root.rglob("checkpoints/step_*/trainable.safetensors"))
        if not weights or not checkpoints:
            return ()
        python = str(python or "")
        if not python:
            return ("未找到引擎解释器，跳过逐检查点导出",)
        kogine = Path(python).parent / ("kogine.exe" if os.name == "nt" else "kogine")
        runner = str(kogine if kogine.is_file() else "kogine")
        script = str(Path(str(source_path)) / "scripts" / "export" / "export.py")
        formats = self._export_formats(values.get("EXPORT_FORMATS"))
        if not formats:
            return ()
        notices = []
        export_dir = root / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        name = config.get("output_name") or config.get("name") or root.name
        for checkpoint in checkpoints:
            step = checkpoint.parent.name.replace("step_", "step")
            for fmt in formats:
                target = export_dir / f"{name}-{step}-{fmt}.safetensors"
                if target.is_file():
                    continue
                command = [runner, "run", script, "--set", f"WEIGHTS={weights}",
                           "--set", f"ADAPTER={checkpoint}", "--set", f"OUT={target}",
                           "--set", f"FORMAT={fmt}"]
                try:
                    result = subprocess.run(command, cwd=str(source_path), capture_output=True,
                                            timeout=3600, text=True, encoding="utf-8", errors="replace")
                except (OSError, subprocess.SubprocessError) as exc:
                    notices.append(f"{step} 导出 {fmt} 失败：{exc}")
                    continue
                if result.returncode or not target.is_file():
                    tail = (result.stderr or result.stdout or "").strip().splitlines()[-1:]
                    notices.append(f"{step} 导出 {fmt} 失败：{' '.join(tail)[:200]}")
                else:
                    notices.append(f"{step} 已导出可用 LoRA：{target.name}")
        return tuple(notices)

    def collect_artifacts(self, job_directory):
        config = json.loads((Path(job_directory) / "job.json").read_text(encoding="utf-8"))
        root = Path(config["output_dir"])
        if not root.is_dir():
            return ()
        found = []
        for path in root.rglob("*"):
            # `"cache" in parts` compared whole names, so `moss-cache` slipped through:
            # match the substring on every parent directory instead.
            parts = path.relative_to(root).parts[:-1]
            if not path.is_file() or any("cache" in part.lower() for part in parts):
                continue
            suffix = path.suffix.lower()
            if suffix == ".safetensors":
                found.append(Artifact(path, "checkpoint", True))
            elif suffix in (".png", ".jpg", ".jpeg", ".webp"):
                # The engine writes samples/step_00000100/000.png and a combined
                # samples/grid.png (evaluate.py:75-104). The step lives in the parent
                # directory, not the file name, so derive it here instead of letting the
                # supervisor's file-name heuristic miss it.
                step = None
                for part in reversed(parts):
                    match = re.fullmatch(r"step_(\d+)", part)
                    if match:
                        step = int(match[1])
                        break
                if path.name.lower() == "grid.png":
                    found.append(Artifact(path, "grid", True))
                else:
                    found.append(Artifact(path, "sample", True, step))
        return tuple(found)
