
# File: `data/README_DATA.md`

## Data Overview

This folder provides the **minimal artifacts** required to reproduce the plots and metrics reported in the paper (without releasing the full training code). It contains:

* **Mobility** (`data/mobility/`): UAV trajectories and altitude plans across time slots.
* **Arrivals** (`data/arrivals/`): Task-arrival traces under *Poisson*, *bursty*, and *heavy-tailed* models.
* **Reliability** (`data/reliability/`): GBS resource evolution and failure/recovery traces.



## 1) Mobility (`data/mobility/`)

Example files:

* `uav_traj_seed{{SEED}}.csv`
* `altitudes.csv`

### `uav_traj_seed{{SEED}}.csv` schema

| column      | type  | unit | description                        |
| ----------- | ----- | ---- | ---------------------------------- |
| `time_slot` | int   | slot | Discrete time index `n`            |
| `uav_id`    | int   | -    | UAV index `m`                      |
| `x`         | float | m    | X coordinate                       |
| `y`         | float | m    | Y coordinate                       |
| `z`         | float | m    | Altitude                           |
| `v_xy_max`  | float | m/s  | Max horizontal speed (from config) |
| `v_z_max`   | float | m/s  | Max vertical speed (from config)   |

### `altitudes.csv` schema

| column       | type  | unit | description                         |
| ------------ | ----- | ---- | ----------------------------------- |
| `uav_id`     | int   | -    | UAV index                           |
| `altitude_m` | float | m    | Planned cruise altitude for the UAV |

**Notes**

* Trajectories cover initial and final waypoints from `configs/mobility_urban.yaml`.
* If you simulate different seeds/scenarios, store them as separate files, e.g., `uav_traj_seed42.csv`.

---



## Provenance & Reproducibility

To ensure full reproducibility, we commit to making the complete source code and the necessary data used to generate the results presented in this manuscript publicly available.

---

# File: `scripts/reproduce_fig6_delay.py`

```python
#!/usr/bin/env python3
"""Reproduce Fig. 6 (Average Delay) from prepared data & configs.
This script DOES NOT train models; it reads mobility/arrival/reliability data
and demonstrates how to compute & plot high-level delay metrics from a precomputed
per-slot/per-UAV results CSV if available. Otherwise, it illustrates placeholders
for queueing-based aggregation.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_configs(config_paths: List[str]) -> Dict:
    cfg = {}
    for p in config_paths:
        with open(p, 'r', encoding='utf-8') as f:
            part = yaml.safe_load(f) or {}
        # shallow merge (later overrides earlier)
        for k, v in part.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def compute_average_delay(delay_csv: Path) -> pd.DataFrame:
    """Expect a CSV with columns: time_slot, uav_id, delay_ms, scheme.
    If you don't have per-task delay, you can adapt to per-slot average delay.
    """
    df = pd.read_csv(delay_csv)
    # defensive checks
    needed = {"time_slot", "uav_id", "delay_ms", "scheme"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {delay_csv}: {missing}")

    # Average over time and UAVs per scheme
    avg = (
        df.groupby("scheme")["delay_ms"].mean().reset_index()
        .rename(columns={"delay_ms": "avg_delay_ms"})
    )
    return avg


def plot_bar(df: pd.DataFrame, out_png: Path, title: str):
    plt.figure()
    plt.bar(df["scheme"], df["avg_delay_ms"])  # color not specified by design
    plt.ylabel("Average Delay (ms)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", help="One or more YAML config files (ordered)")
    ap.add_argument("--delay_csv", type=str, default="outputs/intermediate/delay_timeseries.csv",
                    help="CSV containing per-slot per-UAV delay with a 'scheme' column")
    ap.add_argument("--out", type=str, default="outputs/fig6/",
                    help="Output directory for figure")
    args = ap.parse_args()

    cfg = load_configs(args.config) if args.config else {}
    out_dir = Path(args.out)
    ensure_outdir(out_dir)

    # Compute
    avg = compute_average_delay(Path(args.delay_csv))

    # Save table
    avg.to_csv(out_dir / "fig6_delay_table.csv", index=False)

    # Plot
    plot_bar(avg, out_dir / "fig6_delay.png", title=cfg.get("titles", {}).get("fig6", "Fig.6 Average Delay"))

    print(json.dumps({
        "out_table": str(out_dir / "fig6_delay_table.csv"),
        "out_fig": str(out_dir / "fig6_delay.png")
    }, indent=2))


if __name__ == "__main__":
    main()
```

---

# File: `scripts/reproduce_fig7_energy.py`

```python
#!/usr/bin/env python3
"""Reproduce Fig. 7 (Energy Consumption) from prepared data & configs.
Assumes an input CSV with columns: time_slot, uav_id, tx_power_w, cpu_power_w, scheme.
Computes average energy per slot and aggregates across time and UAVs.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_configs(paths: List[str]) -> Dict:
    cfg = {}
    for p in paths:
        with open(p, 'r', encoding='utf-8') as f:
            part = yaml.safe_load(f) or {}
        for k, v in part.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def compute_energy(telemetry_csv: Path, slot_duration_s: float = 0.02) -> pd.DataFrame:
    """Compute energy = power * time for TX and CPU components.
    Input columns: time_slot, uav_id, tx_power_w, cpu_power_w, scheme
    """
    df = pd.read_csv(telemetry_csv)
    need = {"time_slot", "uav_id", "tx_power_w", "cpu_power_w", "scheme"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {telemetry_csv}: {missing}")

    df["slot_energy_j"] = (df["tx_power_w"] + df["cpu_power_w"]) * slot_duration_s

    agg = (
        df.groupby("scheme")["slot_energy_j"].mean().reset_index()
        .rename(columns={"slot_energy_j": "avg_energy_per_slot_j"})
    )
    return agg


def plot_bar(df: pd.DataFrame, out_png: Path, title: str):
    plt.figure()
    plt.bar(df["scheme"], df["avg_energy_per_slot_j"])  # no explicit colors
    plt.ylabel("Avg Energy per Slot (J)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", help="One or more YAML config files (ordered)")
    ap.add_argument("--telemetry_csv", type=str, default="outputs/intermediate/power_timeseries.csv",
                    help="CSV with power telemetry and 'scheme'")
    ap.add_argument("--out", type=str, default="outputs/fig7/")
    args = ap.parse_args()

    cfg = load_configs(args.config) if args.config else {}
    out_dir = Path(args.out)
    ensure_outdir(out_dir)

    slot_dur = cfg.get("sim", {}).get("slot_duration_s", 0.02)
    agg = compute_energy(Path(args.telemetry_csv), slot_duration_s=slot_dur)
    agg.to_csv(out_dir / "fig7_energy_table.csv", index=False)

    plot_bar(agg, out_dir / "fig7_energy.png", title=cfg.get("titles", {}).get("fig7", "Fig.7 Energy"))

    print(json.dumps({
        "out_table": str(out_dir / "fig7_energy_table.csv"),
        "out_fig": str(out_dir / "fig7_energy.png")
    }, indent=2))


if __name__ == "__main__":
    main()
```

---

# File: `scripts/reproduce_fig8_reliability.py`

```python
#!/usr/bin/env python3
"""Reproduce Fig. 8 (Reliability–Utility Tradeoff) from data & configs.
Assumes two inputs:
  (1) reliability CSV with columns: time_slot, gbs_id, resource_level
  (2) performance CSV with columns: time_slot, uav_id, utility, scheme
Output: aggregated tradeoff curve/table per scheme.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_configs(paths: List[str]) -> Dict:
    cfg = {}
    for p in paths:
        with open(p, 'r', encoding='utf-8') as f:
            part = yaml.safe_load(f) or {}
        for k, v in part.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def aggregate_reliability(rel_csv: Path) -> float:
    df = pd.read_csv(rel_csv)
    need = {"time_slot", "gbs_id", "resource_level"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {rel_csv}: {missing}")
    # Example metric: time-averaged available resource level across GBSes
    return float(df["resource_level"].mean())


def aggregate_utility(perf_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(perf_csv)
    need = {"time_slot", "uav_id", "utility", "scheme"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {perf_csv}: {missing}")
    agg = df.groupby("scheme")["utility"].mean().reset_index().rename(columns={"utility": "avg_utility"})
    return agg


def plot_tradeoff(df: pd.DataFrame, avg_rel: float, out_png: Path, title: str):
    plt.figure()
    # plot each scheme's avg utility as a bar, annotate the shared reliability metric
    plt.bar(df["scheme"], df["avg_utility"])  # no explicit colors
    plt.ylabel("Average Utility (a.u.)")
    plt.title(title)
    # annotate reliability as text
    plt.figtext(0.99, 0.01, f"Avg GBS resource level = {avg_rel:.3f}", ha='right')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", help="One or more YAML config files (ordered)")
    ap.add_argument("--reliability_csv", type=str, default="data/reliability/gbs_profile_setA.csv")
    ap.add_argument("--performance_csv", type=str, default="outputs/intermediate/utility_timeseries.csv")
    ap.add_argument("--out", type=str, default="outputs/fig8/")
    args = ap.parse_args()

    cfg = load_configs(args.config) if args.config else {}
    out_dir = Path(args.out)
    ensure_outdir(out_dir)

    avg_rel = aggregate_reliability(Path(args.reliability_csv))
    util = aggregate_utility(Path(args.performance_csv))

    # Write table combining reliability with utility
    util["avg_gbs_resource_level"] = avg_rel
    util.to_csv(out_dir / "fig8_tradeoff_table.csv", index=False)

    plot_tradeoff(util, avg_rel, out_dir / "fig8_tradeoff.png", title=cfg.get("titles", {}).get("fig8", "Fig.8 Reliability–Utility"))

    print(json.dumps({
        "out_table": str(out_dir / "fig8_tradeoff_table.csv"),
        "out_fig": str(out_dir / "fig8_tradeoff.png")
    }, indent=2))


if __name__ == "__main__":
    main()
```

---

