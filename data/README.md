# data/ — Walch et al. (2019) Apple Watch sleep dataset

Data layout follows the cookiecutter-data-science convention:

| Subdir | Holds |
|--------|-------|
| `raw/` | the original, immutable recordings (`raw/walch2019/`) — local only |
| `interim/` | parse caches (`interim/walch2019_cache/`) — local only |
| `processed/` | the committed `featurematrix.npz` — the fixed feature set models train on |
| `external/` | third-party data, if any |

`raw/` and `interim/` stay local and are never committed (see the repo `.gitignore`);
`dataset.py` raises if `raw/walch2019/` is empty.

## Get the data

The dataset is "Sleep stage prediction with raw acceleration and photoplethysmography
heart rate data derived from a consumer wearable device" (Walch, Huang, Forger &
Goldstein, 2019) — 31 subjects, one night each.

- PhysioNet: https://physionet.org/content/sleep-accel/1.0.0/
- Source repo: https://github.com/ojwalch/sleep_classifiers

Place the per-subject text files **directly in `data/raw/walch2019/`** (flat, no
subdirectories):

```
data/raw/walch2019/
  46343_acceleration.txt
  46343_heartrate.txt
  46343_labeled_sleep.txt
  759667_acceleration.txt
  ...
```

`dataset.py` auto-discovers every `*_labeled_sleep.txt` and loads the matching
streams. Subject id = the filename prefix.

## Expected file formats

| File | Columns | Notes |
|------|---------|-------|
| `<id>_acceleration.txt` | `t(s)  x  y  z` | triaxial acceleration in g |
| `<id>_heartrate.txt`    | `t(s), bpm`     | comma-separated |
| `<id>_labeled_sleep.txt`| `t(s)  stage`   | 30 s epochs; PSG codes below |

PSG stage codes: `-1` unscored, `0` Wake, `1` N1, `2` N2, `3` N3, `4` (legacy stage 4
→ folded into N3), `5` REM. `dataset.py` maps these to the canonical `Wake/N1/N2/N3/REM`
set and treats REM as the positive class.

Once the files are present, confirm the loader sees them:

```bash
uv run python -m remdetect.dataset   # reports subjects, scored epochs, REM %
```
