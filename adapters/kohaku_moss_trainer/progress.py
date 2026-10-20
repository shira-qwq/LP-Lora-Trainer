"""Progress lines of KohakuMossTrainer's log callbacks, read from their own wording.

    [23:15:21] [training.callbacks.monitor] [INFO] step 1 [loss=0.2752, grad_norm=0.02899, lr=1e-06]
    [23:15:41] [training.callbacks.kernels] [INFO] step 1 18.371 s/step, steps 2-10 median 1.806 s/step (10.2x)

No value is guessed: a payload is produced only from fields the line actually prints.
"""

import math
import re

from adapters.capabilities import TrainingEvent

_STEP = re.compile(r"\bstep (\d+)(?:/(\d+))?\b")
_RANGE = re.compile(r"training from step (\d+) to (\d+)")
_LOSS = re.compile(r"\bloss=([-+0-9.eE]+)")
_LR = re.compile(r"\blr=([-+0-9.eE]+)")
_GRAD = re.compile(r"\bgrad_norm=([-+0-9.eE]+)")
_MEDIAN = re.compile(r"median ([-+0-9.eE]+) s/step")
_SINGLE = re.compile(r"\bstep \d+ ([-+0-9.eE]+) s/step")
# kogine echoes the script's return value when the stage exits. That TrainState repr
# repeats the last step's loss/grad_norm/lr (with full float precision), so parsing it
# would append a second metric row for a step that already has one.
_RETURNED = re.compile(r"Return value: TrainState\(")
# The sampler logs its own step counter (`[inference.steplog] step 3/8: ...`) while the
# training callback keeps `step N [loss=…]`. Both match the step wording, so without this
# filter the task's step/total jumped to the sampler's 1/8 at every SAMPLE_EVERY tick.
_INFERENCE = re.compile(r"\[inference\.")

# ---- what the engine resolved at run time (training.build.model / callbacks) ----
_RUN = re.compile(
    r"run (?P<name>.+?): (?P<algo>\w+) (?P<precision>[\w.]+) on (?P<device>cuda:\d+), "
    r"(?P<steps>\d+) steps")
_PROFILE = re.compile(r"MEMORY_PROFILE='(?P<requested>[^']*)'\s*->\s*(?P<resolved>\S+)")
_TEXT_PLACE = re.compile(r"text encoder placement: (?P<value>\w+)")
_VAE_PLACE = re.compile(r"VAE placement: (?P<value>\w+)")
_TRAIN_PEAK = re.compile(
    r"VRAM peak: prewarm (?P<p_alloc>[-+0-9.]+) GiB allocated / (?P<p_res>[-+0-9.]+) "
    r"reserved; step (?P<step>\d+) (?P<alloc>[-+0-9.]+) / (?P<res>[-+0-9.]+)")
_PEAK = re.compile(
    r"peak VRAM (?P<alloc>[-+0-9.]+) GiB allocated \[peak_reserved_gib=(?P<res>[-+0-9.]+), "
    r"device_used_gib=(?P<device>[-+0-9.]+), estimated_gib=(?P<estimated>[-+0-9.]+)\]")
_ESTIMATE = re.compile(r"estimated peak (?P<gib>[-+0-9.]+) GiB")
_VRAM = re.compile(
    r"vram after (?P<phase>build|prewarm|step \d+) on (?P<device>cuda:\d+): "
    r"(?P<gib>[-+0-9.]+) GiB")


def parse_progress(line):
    """One training line -> a progress event, or () when it carries no measurement."""
    if _RETURNED.search(line) or _INFERENCE.search(line):
        return ()
    payload = {}
    # `training from step 0 to 1500` announces the run: it is the only total it prints.
    span = _RANGE.search(line)
    if span:
        payload["step"] = int(span[1])
        payload["total_steps"] = int(span[2])
    step = _STEP.search(line)
    if step:
        payload["step"] = int(step[1])
        if step[2]:
            payload["total_steps"] = int(step[2])
    loss = _LOSS.search(line)
    if loss:
        value = float(loss[1])
        if math.isfinite(value):
            payload["loss"] = value
    # The monitor callback prints these next to the loss: lr (scheduler) and grad_norm.
    lr = _LR.search(line)
    if lr:
        value = float(lr[1])
        if math.isfinite(value):
            payload["lr"] = value
    grad = _GRAD.search(line)
    if grad:
        value = float(grad[1])
        if math.isfinite(value):
            payload["grad_norm"] = value
    # The median of the last window is the honest rate: step 1 includes kernel tuning.
    speed = _MEDIAN.search(line) or _SINGLE.search(line)
    if speed:
        value = float(speed[1])
        if value > 0:
            payload["it_per_sec"] = 1 / value
    return (TrainingEvent("progress", payload),) if payload else ()


def runtime_info(lines):
    """What the engine resolved at run time, read from its own log wording.

    LP cannot know what `MEMORY_PROFILE='auto'` resolved to, which placement the engine
    picked, or how much VRAM the run really used; the engine prints all of it. Only
    printed values are returned: a missing key means the log never stated it.
    """
    info = {}
    placement = {}
    vram = {}
    for line in lines:
        run = _RUN.search(line)
        if run:
            info["run"] = {"name": run["name"].strip(), "algo": run["algo"],
                           "precision": run["precision"], "device": run["device"],
                           "steps": int(run["steps"])}
        profile = _PROFILE.search(line)
        if profile:
            info["memory_profile"] = {"requested": profile["requested"],
                                      "resolved": profile["resolved"]}
        text = _TEXT_PLACE.search(line)
        if text:
            placement["text_encoder"] = text["value"]
        vae = _VAE_PLACE.search(line)
        if vae:
            placement["vae"] = vae["value"]
        train_peak = _TRAIN_PEAK.search(line)
        if train_peak:
            vram.update(train_peak_allocated_gib=float(train_peak["alloc"]),
                        train_peak_reserved_gib=float(train_peak["res"]),
                        train_peak_step=int(train_peak["step"]),
                        prewarm_peak_allocated_gib=float(train_peak["p_alloc"]),
                        prewarm_peak_reserved_gib=float(train_peak["p_res"]))
        peak = _PEAK.search(line)
        if peak:
            # The sampler's own peak: usually the largest phase of the whole run.
            vram.update(peak_allocated_gib=float(peak["alloc"]),
                        peak_reserved_gib=float(peak["res"]),
                        peak_device_used_gib=float(peak["device"]),
                        estimated_gib=float(peak["estimated"]))
        estimate = _ESTIMATE.search(line)
        if estimate:
            vram["estimated_gib"] = float(estimate["gib"])
        after = _VRAM.search(line)
        if after:
            vram["last_device_gib"] = float(after["gib"])
            vram["last_phase"] = after["phase"]
    if placement:
        info["placement"] = placement
    if vram:
        info["vram"] = vram
    return info
