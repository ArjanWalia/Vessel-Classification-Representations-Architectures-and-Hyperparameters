# Vessel Classification Pipeline (CCNC_notebook.ipynb)

End-to-end pipeline for underwater vessel classification: build leakage-free
train/validation/test splits from the VTUAD and a custom ONC dataset, convert
the audio into three representations (mel spectrogram, 3-channel MCG, MFCC),
sweep hyperparameter grids for seven model/representation combinations, and
evaluate the best checkpoints on the test set.

Everything runs locally on plain folders — no Google Drive, no tar archives,
no Google Sheets. All paths derive from a single **paths cell** at the top of
the notebook: set `BASE_DIR` once, run that cell first, and every other cell
finds its inputs and outputs automatically. Results grids are written as CSV
files; test results are printed to the terminal.

## Requirements

```bash
pip install pandas tqdm numpy librosa soundfile scikit-learn joblib \
            torch torchvision transformers
```

Classes: `background`, `cargo`, `passengership`, `tanker`, `tug`.

Notebook-on-macOS caveat: if a cell using `multiprocessing` or a PyTorch
`DataLoader` hangs or errors, set `WORKERS = 1` / `NUM_WORKERS = 0` in its
config block. Training uses CUDA if available, then Apple `mps`, then CPU.

## 0. Paths and inputs

Run the **PROJECT PATHS** cell at the top of the notebook first — it defines
the globals every other cell uses:

```python
BASE_DIR = Path(".../vessel_classification")   # CHANGE THIS — the only path you must set
INPUTS  = BASE_DIR / "Inputs"    # VTUAD range folders + ONC data live here
OUTPUTS = BASE_DIR / "Outputs"   # everything the pipeline produces
ONC_METADATA_ROOT = INPUTS / "ONC" / "07b_classified_wav_files" / "inclusion_2000_exclusion_4000"
```

Download the VTUAD dataset into `Inputs/` (one subfolder per
inclusion/exclusion range: `2000_4000`, `3000_5000`, `4000_6000`), and place
the custom ONC data under `Inputs/ONC/`. The paths cell must be re-run after
every kernel restart before running anything else.

---

## 1. VTUAD: split ship IDs per range

Groups every clip by (class, ship MMSI) so the same vessel never leaks across
splits. Reads the VTUAD `metadata_<split>.csv` files and copies each clip into
`<class>_ship_ids/ship_id_<mmsi>/` folders.

```python
RANGE = "2000_4000"                        # CHANGE for 3000_5000 / 4000_6000
ROOT = INPUTS / RANGE                      # VTUAD range folder
OUT = OUTPUTS / RANGE / f"{RANGE}_splits"  # ship-id folders written here
MOVE = False                               # copy (False) or move (True)
```

## 2. VTUAD: build train / validation / test

Three cells, run in order — **train, then validation, then test**. Each cell
randomly picks ship IDs (seeded), copies clips until the per-class target is
met, and renames what it consumed with a `_used` suffix so later splits can
never reuse the same ship. Background has a single ship ID, so its *files* are
partitioned (1/3, then 1/2 of the remainder, then the rest) instead of ships.

```python
RANGE = "2000_4000"                        # CHANGE for 3000_5000 / 4000_6000
OUT = OUTPUTS / RANGE / f"{RANGE}_splits"  # step 1 output
TRAIN_DIR = OUTPUTS / RANGE / "train"      # VAL_DIR / TEST_DIR in the other cells
SEED = 42
TARGETS = {"background": 688, "cargo": 688, ...}   # clips per class (72 val, 40 test)
```

Repeat steps 1–2 for `3000_5000` and `4000_6000` — only `RANGE` changes.

---

## 3. ONC: settings

The ONC dataset is produced by the ONC processing scripts using `config.py`
values (inclusion radius, metadata seconds, device codes). The settings cell
records the configuration used; it is reference, not something to run.

## 4. ONC: group clips by class / MMSI

Reads the classified-clip metadata CSV and places every WAV into
`<class>/<mmsi>/` folders (hardlink, symlink, or copy — or cut into fixed
1-second segments). Writes `grouped_manifest.csv` + `grouped_summary.csv`.

```python
METADATA_ROOT = str(ONC_METADATA_ROOT)             # from the paths cell
METADATA_FILE = "metadata_1s.csv"
OUTPUT_DIR = str(OUTPUTS / "ONC" / "ship_splits")
UNIT = "file"       # "file" = whole WAVs, "segment" = cut into SECONDS-long segments
MODE = "hardlink"   # how whole files are placed
```

## 5. ONC: build train / validation / test

One cell, run **three times** with `SPLIT = "train"`, then `"validation"`,
then `"test"`. Draws per-class segment quotas from a partition of the MMSI
units (train reserves 2/3, validation 1/2 of the rest, test takes what
remains), cuts the segments, and marks used units `_used` in the grouped
folders. Writes `<split>_manifest.csv` and `<split>_report.csv`.

```python
GROUPED_DIR = str(OUTPUTS / "ONC" / "ship_splits")        # step 4 OUTPUT_DIR
METADATA_CSV = str(ONC_METADATA_ROOT / "metadata_1s.csv")
SPLITS_DIR = str(OUTPUTS / "ONC" / "dataset_splits")
SPLIT = "train"                             # then "validation", then "test"
```

---

## 6. Cumulative dataset

Merges the three VTUAD ranges and the ONC dataset into one
`<split>/<class>/*.wav` tree, renaming files `<dataset>_<split>_<class>_<i>_0.wav`
so nothing collides. Prints per-dataset and per-class counts.

```python
SOURCE_DATASETS = ['2000_4000', '3000_5000', '4000_6000', 'ONC']
root_source = str(OUTPUTS)     # folder containing the four datasets
# -> writes to Outputs/cumulative_dataset
```

---

## 7. Representations

All three read `INPUT_DIR/<split>/<class>/*.wav` and share the same STFT
settings (`sr 16000, n_fft 1024, hop 128`) on 1-second clips.

**Mel spectrograms** — `(128, 126)` arrays, log-scaled + min-max to [0, 1]:

```python
INPUT_DIR = OUTPUTS / "cumulative_dataset"
OUTPUT_DIR = OUTPUTS / "mel_dataset"      # mirrored <split>/<class>/*.npy
```

**MCG (3-channel mel / CQT / gammatone)** — `(3, 128, 126)` CHW arrays, each
channel dB + min-max normalized:

```python
INPUT_DIR = OUTPUTS / "cumulative_dataset"
OUTPUT_DIR = OUTPUTS / "mcg_dataset"
```

**MFCC features** — 20 MFCCs aggregated as mean+std over time (40 features per
clip), standardized with a scaler fit on train only. Writes `X_<split>.npy`,
`y_<split>.npy`, the scaler arrays, `classes.json`, and `metadata.json`:

```python
INPUT_DIR = OUTPUTS / "cumulative_dataset"
OUTPUT_DIR = OUTPUTS / "mfcc_dataset"
```

**WAV (for AST)** — the AST model consumes the raw waveforms, preprocessed by
its own feature extractor. This cell precomputes those `(128, 128)` input
features once for every clip in every split into a folder cache, so the AST
training and testing cells skip extraction and start immediately
(skip-existing, so it resumes cleanly):

```python
WAV_DIR = OUTPUTS / "cumulative_dataset"
FEATURE_CACHE = OUTPUTS / "ast_features"   # mirrored <split>/<class>/*.npy
```

---

## 8. Training (grid searches, validation accuracy)

The seven deep-learning cells share one shape: a fresh model per grid cell over
`GRID_LRS = [0.05 ... 1e-6]` × `GRID_EPOCHS = [10, 20, 30, 40, 50]`, AdamW,
batch 64, no augmentation. After every cell the validation-accuracy grid is
rewritten to `OUTPUT_DIR/results_grid.csv` (rows = LR, columns = epochs) and
the best checkpoint is saved. `START_LR`/`START_EPOCHS` set the resume point;
`PRIOR_RESULTS = {(lr, epochs): acc}` seeds the best from earlier sessions.

```python
DATA_DIR = OUTPUTS / "mcg_dataset"      # or "mel_dataset"
OUTPUT_DIR = OUTPUTS / "resnet50_mcg"   # results_grid.csv + best .pt
START_LR, START_EPOCHS = 0.05, 10       # first grid cell to run
```

| Cell | Representation | Model | Best checkpoint |
|---|---|---|---|
| MCG + RESNET50 | `mcg_dataset` | `microsoft/resnet-50` | `Outputs/resnet50_mcg/resnet50_mcg_best.pt` |
| MCG + VGGNET19 | `mcg_dataset` | torchvision VGG19 | `Outputs/vgg19_mcg/vgg19_mcg_best.pt` |
| MCG + VIT | `mcg_dataset` | `google/vit-base-patch16-224` | `Outputs/vit_mcg/vit_mcg_best.pt` |
| mel + resnet50 | `mel_dataset` | `microsoft/resnet-50` | `Outputs/resnet50_mel/resnet50_mel_best.pt` |
| MEL + VGGNET19 | `mel_dataset` | torchvision VGG19 | `Outputs/vgg19_mel/vgg19_mel_best.pt` |
| Mel + ViT | `mel_dataset` | `google/vit-base-patch16-224` | `Outputs/vit_mel/vit_mel_best.pt` |
| WAV + AST | `ast_features` (from raw wavs) | `MIT/ast-finetuned-speech-commands-v2` | `Outputs/ast_wav/ast_wav_best.pt` |

The three MFCC cells sweep classical models on the standardized features and
save the best model with joblib plus a JSON log:

```python
DATA_DIR = OUTPUTS / "mfcc_dataset"
OUTPUT_DIR = OUTPUTS / "knn_mfcc"      # results CSV + best .joblib
```

| Cell | Sweep | Results |
|---|---|---|
| MFCC + KNN | k = 1..14 | `Outputs/knn_mfcc/results.csv` |
| MFCC + RANDOM FOREST | n_estimators × max_depth | `Outputs/rf_mfcc/results_grid.csv` |
| MFCC + LOGISTIC REGRESSION | C × max_iter | `Outputs/logreg_mfcc/results_grid.csv` |

---

## 9. Testing (test accuracy, printed to terminal)

One `(testing)` cell per trained combination. Each loads the best model from
its training cell's `OUTPUT_DIR`, evaluates the `test` split, and prints — no
files are written — the classification report, a text confusion matrix, the
validation → test gap, and a final line:

```
TEST ACCURACY: XX.XX%
```

```python
DATA_DIR = OUTPUTS / "mcg_dataset"                           # matching dataset
CKPT_PATH = OUTPUTS / "resnet50_mcg" / "resnet50_mcg_best.pt"  # matching checkpoint
TEST_SPLIT = "test"
```

The MFCC testing cells point at the `.joblib` model (KNN, RF) or the saved
coefficient/intercept arrays (logistic regression, evaluated with plain numpy)
plus the training grid log for the validation comparison. The WAV + AST
testing cell reads the wav tree, extracts AST features for the test split once
into a folder cache at `Outputs/ast_features` (reused on later runs), and
loads the checkpoint from `Outputs/ast_wav/ast_wav_best.pt`.

## Run order

```
[0] paths cell (after every kernel restart)
Inputs -> [1-2] VTUAD splits (x3 ranges) -> [3-5] ONC splits
       -> [6] cumulative dataset
       -> [7] mel / mcg / mfcc / wav (AST) datasets
       -> [8] training grids  (results_grid.csv, best model)
       -> [9] testing         (TEST ACCURACY printed)
```
