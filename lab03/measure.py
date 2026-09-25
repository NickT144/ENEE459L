from __future__ import annotations

import statistics
from typing import Any

from bench import Bench, measured, read_first, read_text, unknown

import json

from pathlib import Path

# A sample is still warm-up while it exceeds the settled rate by this fraction.
WARMUP_TOL = 0.5

# How many samples must sit strictly above a quantile before that quantile is an
# estimate rather than "the biggest number we saw, wearing a hat".
MIN_SAMPLES_ABOVE = 5

# Percentiles the record carries, in the order the schema lists them.
PERCENTILES = (50, 95, 99)

# The widest gap between neighbouring measurements, as a multiple of the typical
# gap, beyond which the sample is treated as coming from two populations.
MULTIMODAL_GAP_RATIO = 20.0

# Neither side of that gap is a mode unless it holds at least this fraction.
MIN_MODE_FRACTION = 0.10

# Below this many retained samples, modality is not a question worth answering.
MIN_SAMPLES_FOR_MODALITY = 20

# How far the last third of a run may drift from the first third, relative to
# the run's own median, before the run is not one population either.
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12

THERMAL_ZONES = "sys/devices/virtual/thermal"

POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)

GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
)

CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"


# ===========================================================================
# 1. The loop
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    timings = list()

    bench.workload.synchronize()
    for _ in range(repeats):
        start = bench.clock()

        bench.workload.run()
        bench.workload.synchronize()

        end = bench.clock()
        timings.append((end - start) / 1000000.0)

    return timings

def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    count = len(samples)

    if count < 4:
        return unknown("find warmup boundaries", "Not enough samples")

    later_median = statistics.median(samples[count // 2:])

    if later_median <= 0:
        return unknown("find warmup boundaries", "Negative second half median")

    threshold = later_median * (1 + WARMUP_TOL)

    discarded_count = 0
    for sample in samples:
        if sample > threshold:
            discarded_count += 1

    return measured(discarded_count, "leading prefix above (1 + 0.5) x median of the run's second half", status="ok", settled_rate_ms=round(later_median, 4), threshold_ms=round(threshold, 4), tolerance=WARMUP_TOL, retained=(count - discarded_count))

def summarize(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
            "p99": None
        }
    
    samples.sort()

    mean = statistics.fmean(samples)
    minimum = min(samples)
    maximum = max(samples)
    std = 0.0
    count = len(samples) 
    if count > 2:
        std = statistics.stdev(samples)

    plist = [50, 95, 99]
    presults = list()

    for p in plist:
        h = (count - 1) * (p / 100)
        i = int(h)
        value = samples[i] + (h - i) * (samples[i+1] - samples[i])
        presults.append(round(value, 4))

    return {
            "n": count,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(minimum, 4),
            "max": round(maximum, 4),
            "p50": presults[0],
            "p95": presults[1],
            "p99": presults[2]
        }

def is_multimodal(samples: list[float]) -> dict[str, Any]:
    count = len(samples)
    if count < 20 :
        return unknown("is multimodal", "Not enough samples")

    samples.sort()

    trim_index= int(0.05 * count)
    trimmed_samples = samples[trim_index:-trim_index]
    trimmed_count = len(trimmed_samples)

    gaps = list()
    widest_gap = 0
    gap_index = 0
    for i in range(trimmed_count - 1):
        gap = samples[i+1] - samples[i]
        gaps.append(gap)

        if gap > widest_gap:
            widest_gap = gap
            gap_index = i + 1

    gaps.sort()
    median_gap = statistics.median(gaps)
    if median_gap <= 0:
        return unknown("is multimodal","Timer resolution is too coarse")

    ratio = widest_gap / median_gap

    left_samples = samples[:gap_index]
    left_samples_length = len(left_samples)
    left_samples_median = statistics.median(left_samples)

    right_samples = samples[gap_index:]
    right_samples_length = len(right_samples)
    right_samples_median = statistics.median(right_samples)
    ten_threshold = count // 10

    modes = [
        {
            "n": left_samples_length,
            "share": left_samples_length / count,
            "median_ms": round(left_samples_median, 4)
        },
        {
            "n": right_samples_length,
            "share": right_samples_length / count,
            "median_ms": round(right_samples_median, 4)
        }
    ]

    value = ratio >= 20.0 and left_samples_length >= ten_threshold and right_samples_length >= ten_threshold

    return {
        "value": value,
        "source": "widest trimmed gap >= 20.0x the median gap, with >= 10% of samples on each side",
        "status": "ok",
        "gap_ratio": round(ratio, 4),
        "widest_gap_ms": round(widest_gap, 4),
        "typical_gap_ms": round(median_gap, 4),
        "modes": modes
    }

# ===========================================================================
# 7. The clock ceiling the run happened under
# ===========================================================================


def probe_power_state(bench: Bench) -> dict[str, Any]:
    src = ["nvpmodel","-q"]
    output = bench.runner(src)

    if output.returncode != 0:
        return unknown("probe power state", output.error)

    output = output.stdout.split("\n")
    mode_id = int(output[1])
    mode = output[0].replace("NV Power Mode: ", "")

    root = Path("/")
    scaling_min = read_text(root, CPUFREQ_MIN)
    scaling_max = read_text(root, CPUFREQ_MAX)

    jetson_clocks = None
    if scaling_max and scaling_min:
        jetson_clocks = False
        if scaling_max == scaling_min:
            jetson_clocks = True

    return {
        "value": mode,
        "source": "nvpmodel -q",
        "status": "ok",
        "mode_index": mode_id,
        "jetson_clocks": jetson_clocks,
        "jetson_clocks_source": {
            "value": f"scaling_min_freq={scaling_min}, scaling_max_freq={scaling_max}",
            "source": "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq vs sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq",
            "status": "ok"
        } 
    }

def probe_telemetry(bench: Bench) -> dict[str, Any]:
    root = Path('/')

    path = Path('/sys/class/thermal/')
    thermal_dirs = list(path.glob('thermal_zone*/'))
    max_temp = 0.0
    thermal_count = 0
    zone = ""

    # iterate through temp zones and parse out temps and types
    for thermal_dir in thermal_dirs: 
        try:
            for file in thermal_dir.iterdir():
                if "temp" == file.name:
                    temp = float(read_text(root, str(file))) / 1000
                    if temp <= -1000:
                        continue

                    thermal_count += 1

                    if temp > max_temp:
                        max_temp = temp
                        zone = read_text(root, str(file).replace("temp", "type"))
        except TypeError as error:
            continue

    temperature_c = {
        "value": max_temp,
        "source": "sys/devices/virtual/thermal/*/temp",
        "status": "ok",
        "zone": zone,
        "zones_read": thermal_count
    }

    power_mw = unknown("sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input | sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input | sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input", "none of the documented INA3221 rail paths could be read")
    output = read_first(root, POWER_RAIL_CANDIDATES)
    if output:
        _, power = output
        power = float(power) / 10.0

        power_mw = {
            "value": power,
            "source": "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input | sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input | sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
            "status": "ok",
        }
    
    gpu_utilization_percent = unknown("sys/devices/platform/gpu.0/load","no GPU load file is found")
    output = read_first(root, GPU_LOAD_CANDIDATES)
    if output:
        _, gpu_load = output
        gpu_load = float(gpu_load) / 10.0

        gpu_utilization_percent = {
            "value": gpu_load,
            "source": "sys/devices/platform/gpu.0/load",
            "status": "ok",
            "units": "per-mille / 10"
        }

    return {
        "temperature_c": temperature_c,
        "power_mw": power_mw,
        "gpu_utilization_percent": gpu_utilization_percent
    }

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
    # env = Bench.real()
    # out = find_warmup_boundary(samples)
    # print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Bench.real()

    # get your samples
    samples = run_timed_iterations(env, repeats=100)

    # testing measurments and probes
    report = {
        "warmup_boundary": find_warmup_boundary(samples),
        "summarize_setup": summarize(samples),
        "is_multimodal": is_multimodal(samples),
        "probe_power_state": probe_power_state(env),
        "probe_telemetry": probe_telemetry(env),
    }

    # save samples
    path = "samples_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)

    # save report
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)