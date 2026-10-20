"""KohakuMossTrainer adapter: ComfyUI weights, free-form profiles, dataset source,
artifact steps and the Python native config. No engine import, no training."""
import ast
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

from adapters.kohaku_moss_trainer.adapter import KohakuMossTrainerAdapter
from adapters.kohaku_moss_trainer.params import PARAMS
from adapters.kohaku_moss_trainer.progress import parse_progress, runtime_info


def spec(key):
    return next(p for p in PARAMS if p.key == key)


class MossAdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = KohakuMossTrainerAdapter()
        self.tmp = Path(tempfile.mkdtemp(prefix="moss-tests-"))
        # A minimal engine tree instead of a machine-specific checkout: validate()
        # only needs the entry script to recognise the architecture.
        self.engine = self.tmp / "engine"
        (self.engine / "scripts" / "train").mkdir(parents=True)
        (self.engine / "scripts" / "train" / "moss.py").write_text("", encoding="utf-8")
        self.installation = SimpleNamespace(source_path=str(self.engine),
                                            python_executable="",
                                            installation_id="moss")

    def comfy_weights(self):
        root = self.tmp / "comfy"
        for folder, name in (("diffusion_models", "moss_image_base_int8_convrot"),
                             ("text_encoders", "qwen3.5_9b_int8_convrot"),
                             ("vae", "flux2-vae")):
            (root / folder).mkdir(parents=True, exist_ok=True)
            (root / folder / f"{name}.safetensors").write_bytes(b"x")
        return root

    def config(self, weights, **params):
        base = {"WEIGHTS": str(weights), "VARIANT": "base", "DATASET_SOURCE": "folder",
                "MEMORY_PROFILE": "ckpt-ring4-actoff-optstream-gradoff",
                "TEXT_ENCODE": json.dumps({"memory": "ring", "ahead": 4,
                                           "batch": 16, "max_tokens": 8192}),
                "SAMPLE_PROMPTS": '["a cat"]', "SAMPLE_SEEDS": "(0,)"}
        base.update(params)
        return {"architecture": "moss_image", "name": "t", "output_name": "t",
                "dataset_path": str(self.tmp / "data"),
                "output_dir": str(self.tmp / "out"), "params": base}

    def official_layers(self):
        """A generated config whose `use_config` layer supplies the dict knobs."""
        layer = self.tmp / "base_layer.py"
        layer.write_text(
            'TEXT_ENCODE = {"memory": "ring", "ahead": 4, "batch": 16, "max_tokens": 8192}\n'
            'VAE_ENCODE = {"memory": "offload", "ahead": 4}\n'
            'SCHEDULE = {"mode": "cosine", "warmup": 100, "end": -1, "min_value": 0.1}\n'
            'BASE_PRECISION = None\n',
            encoding="utf-8")
        config = self.tmp / "generated_config.py"
        # The engine writes forward slashes; a Windows path would be read as an escape.
        config.write_text('from kohakuengine import use_config\n\n'
                          f'use_config("{str(layer).replace(chr(92), "/")}")\n\n'
                          'WEIGHTS = "placeholder"\n', encoding="utf-8")
        return config

    def render_lines(self, config, baseline=True):
        inherited = self.adapter.inherited_values(self.official_layers()) if baseline else None
        return self.adapter._render_values(config, inherited)

    def test_render_without_overrides_writes_only_the_job_paths_and_the_pinned_policy(self):
        config = self.config(self.comfy_weights())
        config["explicit_params"] = []
        lines = self.render_lines(config)
        self.assertEqual(["WEIGHTS", "DATASETS", "OUTPUT_DIR", "RUN_NAME", "TEXT_ENCODE"],
                         [line.split(" = ")[0] for line in lines if " = " in line])
        # The one deliberate deviation: pinned host memory off, everything else inherited.
        self.assertEqual({"memory": "ring", "ahead": 4, "batch": 16, "max_tokens": 8192,
                          "pinned": False},
                         ast.literal_eval(lines[-1].split(" = ", 1)[1]))
        self.assertNotIn("BASE_PRECISION", "\n".join(lines))
        self.assertNotIn("MEMORY_PROFILE", "\n".join(lines))
        self.assertNotIn("STEPS", "\n".join(lines))

    def test_one_explicit_key_adds_exactly_that_key(self):
        config = self.config(self.comfy_weights(), LR=0.00005)
        config["explicit_params"] = ["LR"]
        lines = self.render_lines(config)
        self.assertEqual(["WEIGHTS", "DATASETS", "OUTPUT_DIR", "RUN_NAME", "LR", "TEXT_ENCODE"],
                         [line.split(" = ")[0] for line in lines if " = " in line])
        self.assertIn("LR = 5e-05", lines)

    def test_warmup_merges_the_inherited_schedule_instead_of_replacing_it(self):
        config = self.config(self.comfy_weights(), WARMUP=30)
        config["explicit_params"] = ["WARMUP"]
        lines = self.render_lines(config)
        schedule = next(line for line in lines if line.startswith("SCHEDULE = "))
        self.assertEqual({"mode": "cosine", "warmup": 30, "end": -1, "min_value": 0.1},
                         ast.literal_eval(schedule.split(" = ", 1)[1]))
        self.assertNotIn("WARMUP = ", "\n".join(lines))

    def test_base_precision_is_inherited_unless_chosen(self):
        config = self.config(self.comfy_weights())
        config["explicit_params"] = []
        self.assertNotIn("BASE_PRECISION", "\n".join(self.render_lines(config)))

        chosen = self.config(self.comfy_weights(), BASE_PRECISION="compute")
        chosen["explicit_params"] = ["BASE_PRECISION"]
        self.assertIn("BASE_PRECISION = 'compute'", self.render_lines(chosen))

    def test_explicit_auto_base_precision_keeps_the_weights_int8_bases(self):
        config = self.config(self.comfy_weights(), BASE_PRECISION="auto")
        config["explicit_params"] = ["BASE_PRECISION"]
        self.assertIn("BASE_PRECISION = 'auto'", self.render_lines(config))

    def fake_generator(self, captured):
        def run(command, **kwargs):
            captured.append(list(command))
            target = Path(command[command.index("--out") + 1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('WEIGHTS = "placeholder"\n', encoding="utf-8")
            target.with_name(f"{target.stem}_data.py").write_text(
                'TEXT_ENCODER_KWARGS = {"path": "placeholder"}\n', encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return run

    def tool_installation(self):
        scripts = self.tmp / "venv" / "Scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        (scripts / "kohakumoss-config.exe").write_bytes(b"")
        return SimpleNamespace(source_path=str(self.engine), python_executable=str(scripts / "python.exe"))

    def generate(self, params):
        captured = []
        installation = self.tool_installation()
        config = self.config(self.comfy_weights(), **params)
        target = self.tmp / "job" / "config.py"
        with unittest.mock.patch("adapters.kohaku_moss_trainer.adapter.subprocess.run",
                                 self.fake_generator(captured)):
            ok, reason = self.adapter._generate_official(installation, config, target)
        self.assertTrue(ok, reason)
        return captured[0]

    def test_generator_gets_no_vram_flag_by_default(self):
        # Nothing on the command line: the engine detects the card and picks vram_12g here.
        self.assertNotIn("--vram", self.generate({}))

    def test_generator_passes_vram_auto_only_when_chosen(self):
        command = self.generate({"VRAM": "auto"})
        self.assertEqual("auto", command[command.index("--vram") + 1])

    def test_generator_passes_a_manual_vram_budget(self):
        command = self.generate({"VRAM": "12"})
        self.assertEqual("12", command[command.index("--vram") + 1])

    def test_vram_accepts_auto_blank_or_gib_only(self):
        bad = self.adapter.validate(self.installation, self.config(self.comfy_weights(), VRAM="lots"))
        self.assertEqual(["error"], [p.severity for p in bad if p.field == "VRAM"])
        for value in ("", "auto", "12", "11.5"):
            issues = self.adapter.validate(self.installation, self.config(self.comfy_weights(), VRAM=value))
            self.assertEqual([], [p for p in issues if p.field == "VRAM"], value)

    def test_cache_plan_merges_into_the_generated_kwargs(self):
        values = self.config(self.comfy_weights())["params"]
        joined = "\n".join(self.adapter._render_cache_values({"params": values}))
        self.assertIn("TEXT_ENCODER_KWARGS = {**TEXT_ENCODER_KWARGS, 'plan':", joined)
        self.assertIn("'memory': 'ring'", joined)
        # The generated path/device are kept, not rewritten with LP's own copy.
        self.assertNotIn("'path'", joined)

    def test_cache_plan_without_overrides_is_only_the_pinned_policy(self):
        config = self.config(self.comfy_weights())
        config["explicit_params"] = []
        self.assertEqual(
            ["TEXT_ENCODER_KWARGS = {**TEXT_ENCODER_KWARGS, 'plan': {'pinned': False}}"],
            self.adapter._render_cache_values(config))

    def test_memory_profile_is_free_text_with_auto_default(self):
        profile = spec("MEMORY_PROFILE")
        self.assertEqual(profile.type, "string")
        self.assertEqual(profile.default, "auto")
        self.assertEqual(profile.enum_values, [])

    def test_comfyui_layout_passes_and_long_profile_name_is_accepted(self):
        problems = self.adapter.validate(self.installation, self.config(self.comfy_weights()))
        self.assertEqual([], [p for p in problems if p.severity == "error"], problems)

    def test_hf_dir_without_transformer_is_rejected_with_both_layouts_named(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        problems = self.adapter.validate(self.installation, self.config(empty))
        message = " ".join(p.message for p in problems if p.field == "WEIGHTS")
        self.assertIn("diffusion_models", message)
        self.assertIn("transformer", message)

    def test_single_safetensors_is_a_warning_not_an_error(self):
        single = self.tmp / "dit.safetensors"
        single.write_bytes(b"x")
        problems = self.adapter.validate(self.installation, self.config(single))
        weights = [p for p in problems if p.field == "WEIGHTS"]
        self.assertEqual(["warning"], [p.severity for p in weights])
        self.assertFalse([p for p in problems if p.severity == "error"])

    def test_render_keeps_chosen_source_and_never_guesses_a_vae_path(self):
        text = self.adapter.native_config_text(self.config(self.comfy_weights()), None)
        self.assertIn("'source': 'folder'", text)
        self.assertNotIn("\nVAE = ", text)
        self.assertIn("MEMORY_PROFILE = 'ckpt-ring4-actoff-optstream-gradoff'", text)
        kohya = self.adapter.native_config_text(
            self.config(self.comfy_weights(), DATASET_SOURCE="kohya"), None)
        self.assertIn("'source': 'kohya'", kohya)

    def test_artifacts_take_the_step_from_the_parent_and_keep_grid_separate(self):
        out = self.tmp / "out"
        step = out / "run" / "samples" / "step_00000100"
        step.mkdir(parents=True)
        (step / "000.png").write_bytes(b"x")
        (out / "run" / "samples" / "grid.png").write_bytes(b"x")
        job = self.tmp / "job"
        job.mkdir()
        (job / "job.json").write_text(json.dumps({"output_dir": str(out)}), encoding="utf-8")
        artifacts = {a.path.name: a for a in self.adapter.collect_artifacts(job)}
        self.assertEqual(("sample", 100), (artifacts["000.png"].kind, artifacts["000.png"].step))
        self.assertEqual("grid", artifacts["grid.png"].kind)

    def test_images_in_tool_cache_folders_block_the_dataset(self):
        data = self.tmp / "data"
        (data / "cache").mkdir(parents=True)
        (data / "cache" / "thumb.png").write_bytes(b"x")
        problems = self.adapter.validate_dataset(self.installation, {"path": str(data)}, [])
        self.assertEqual(1, len(problems))
        self.assertEqual("error", problems[0].severity)

    def test_missing_official_generator_falls_back_to_a_complete_config(self):
        self.installation.python_executable = sys.executable
        job = self.tmp / "job"
        job.mkdir()
        target = self.adapter.write_native_config(
            self.installation, self.config(self.comfy_weights()), job)
        text = target.read_text(encoding="utf-8")
        self.assertIn("官方 kohakumoss-config 不可用", text)
        self.assertIn("DATASETS = ", text)
        stages = json.loads((job / "stages.json").read_text(encoding="utf-8"))
        self.assertEqual(3, len(stages))
        self.assertEqual("moss.py", Path(stages[-1][2]).name)

    def test_missing_explicit_list_means_every_submitted_key_is_an_override(self):
        # Script clients post exactly the keys they mean and know nothing about the list.
        config = self.config(self.comfy_weights())
        self.assertNotIn("explicit_params", config)
        text = "\n".join(self.render_lines(config))
        self.assertIn("MEMORY_PROFILE = 'ckpt-ring4-actoff-optstream-gradoff'", text)
        self.assertIn("SAMPLE_PROMPTS = ['a cat']", text)
        # Even here a key the client never sent is not invented.
        self.assertNotIn("STEPS = ", text)
        self.assertNotIn("BASE_PRECISION", text)

    def test_text_encode_policy_turns_pinned_host_memory_off(self):
        # The engine's TextEncoderPlan.pinned defaults to True and copies every layer to
        # page-locked host memory (cuMemHostAlloc -> CUDA OOM on a 32GB host).
        self.assertEqual({}, json.loads(spec("TEXT_ENCODE").default))
        values = {p.key: p.default for p in PARAMS}
        self.assertEqual({"pinned": False}, self.adapter._text_plan(values, set()))
        # A chosen plan wins field by field, including an explicit pinned.
        self.assertEqual({"pinned": True},
                         self.adapter._text_plan({"TEXT_ENCODE": '{"pinned": true}'}, {"TEXT_ENCODE"}))

    def test_text_pinned_inherit_leaves_the_official_plan_untouched(self):
        # "inherit" is the fully native mode: no TEXT_ENCODE line at all when the user
        # changed nothing else, and no cache-stage plan either.
        spec_ = spec("TEXT_PINNED")
        self.assertEqual("off", spec_.default)
        self.assertEqual(["off", "inherit", "on"],
                         [option[0] for option in (spec_.options or [])])
        values = {p.key: p.default for p in PARAMS}
        values["TEXT_PINNED"] = "inherit"
        self.assertEqual({}, self.adapter._text_plan(values, set()))

        config = self.config(self.comfy_weights(), TEXT_PINNED="inherit")
        config["explicit_params"] = []
        lines = self.render_lines(config)
        self.assertEqual(["WEIGHTS", "DATASETS", "OUTPUT_DIR", "RUN_NAME"],
                         [line.split(" = ")[0] for line in lines if " = " in line])
        self.assertEqual([], self.adapter._render_cache_values(config))

    def test_text_pinned_can_be_forced_on(self):
        values = {p.key: p.default for p in PARAMS}
        values["TEXT_PINNED"] = "on"
        self.assertEqual({"pinned": True}, self.adapter._text_plan(values, set()))
        config = self.config(self.comfy_weights(), TEXT_PINNED="on")
        config["explicit_params"] = []
        lines = self.render_lines(config)
        plan = ast.literal_eval(next(line for line in lines if line.startswith("TEXT_ENCODE = "))
                                .split(" = ", 1)[1])
        self.assertIs(True, plan["pinned"])

    def test_progress_reads_lr_and_grad_norm_and_skips_the_return_value_echo(self):
        line = ("[19:48:45] [training.callbacks.monitor] [INFO] "
                "step 16 [loss=0.5118, grad_norm=0.02272, lr=1.17e-05]")
        payload = parse_progress(line)[0].payload
        self.assertEqual({"step": 16, "loss": 0.5118, "lr": 1.17e-05,
                          "grad_norm": 0.02272}, payload)
        # kogine echoes TrainState when the stage exits: same step, unusable precision.
        self.assertEqual((), parse_progress(
            "Return value: TrainState(step=20, loss=0.6821029782295227, lr=0.0)"))

    def test_sampler_step_counter_never_overwrites_the_training_progress(self):
        # `SAMPLE_EVERY` runs the sampler inside the run; its steplog says 1/8 while the
        # trainer is at step 10 of 20.
        self.assertEqual((), parse_progress(
            "[19:48:23] [inference.steplog] [INFO] step 1/8: 6.331 s, H2D 5.89 GB"))
        self.assertEqual(20, parse_progress(
            "[19:47:23] [training.entry] [INFO] training from step 0 to 20")[0].payload[
                "total_steps"])

    def test_warmup_writes_only_the_lr_sub_schedule(self):
        config = self.config(self.comfy_weights(), WARMUP=30,
                             SCHEDULE='{"mode": "cosine", "warmup": 100, "end": -1}')
        text = self.adapter.native_config_text(config, None)
        self.assertIn("SCHEDULE = {'mode': 'cosine', 'warmup': 30, 'end': -1}", text)
        self.assertNotIn("WARMUP = ", text)

        # Multi-key schedules keep ema and every other lr field as written.
        grouped = self.adapter._schedule({"SCHEDULE": json.dumps(
            {"lr": {"mode": "cosine", "warmup": 100, "end": 500},
             "ema": {"mode": "linear", "end": 500}}), "WARMUP": 7})
        self.assertEqual({"mode": "cosine", "warmup": 7, "end": 500}, grouped["lr"])
        self.assertEqual({"mode": "linear", "end": 500}, grouped["ema"])

        # Unset (the default) never overrides SCHEDULE's own warmup.
        untouched = self.adapter._schedule({"SCHEDULE": json.dumps(
            {"mode": "cosine", "warmup": 2, "end": -1})})
        self.assertEqual({"mode": "cosine", "warmup": 2, "end": -1}, untouched)
        self.assertEqual((), self.adapter._schedule_issues(
            {"SCHEDULE": '{"mode": "cosine", "warmup": 2, "end": -1}', "STEPS": 20}))

    def test_warmup_must_end_before_the_schedule(self):
        problems = self.adapter.validate(
            self.installation,
            self.config(self.comfy_weights(), WARMUP=100, STEPS=20,
                        SCHEDULE='{"mode": "cosine", "warmup": 100, "end": -1}'))
        warmup = [p for p in problems if p.field == "WARMUP"]
        self.assertEqual(["error"], [p.severity for p in warmup])
        self.assertIn("STEPS", warmup[0].message)

    def test_runtime_info_reports_only_what_the_engine_printed(self):
        info = runtime_info([
            "[19:45:48] [training.entry] [INFO] run demo: lora bf16 on cuda:0, 20 steps, output X",
            "[19:45:51] [training.build.model] [INFO] MEMORY_PROFILE='auto' -> balanced",
            "[19:46:59] [training.build.model] [INFO] VAE placement: offload (VAE_ENCODE['memory'])",
            "[19:47:09] [training.build.model] [INFO] text encoder placement: ring (TEXT_ENCODE['memory'])",
            "[19:48:24] [inference.pipeline] [INFO] peak VRAM 8.95 GiB allocated "
            "[peak_reserved_gib=10.02, device_used_gib=6.505, estimated_gib=6.88]",
        ])
        self.assertEqual({"requested": "auto", "resolved": "balanced"}, info["memory_profile"])
        self.assertEqual({"text_encoder": "ring", "vae": "offload"}, info["placement"])
        self.assertEqual("bf16", info["run"]["precision"])
        self.assertEqual(10.02, info["vram"]["peak_reserved_gib"])
        # Nothing printed -> nothing claimed (never a defaulted profile).
        self.assertEqual({}, runtime_info(["unrelated line"]))


if __name__ == "__main__":
    unittest.main()
