**Deep Vessel Classification: Representations, Architectures, and Hyperparameters**

For an in-depth break-down of the pipeline, go to the paper linked below:

{insert}


This is a complete pipeline that goes from raw WAV files to the training results of KNN, Logistic Regression, K-Nearest Neighbors, ResNet50, VGGNet19, MIT AST, and the ViT. This repo includes dataset leakage prevention, pre-processing of datasets, and then model traning and results. Below is a detailed list of the things included:

1) Processing of RAW wav file datasets (VTUAD, Deepship, custom ONC dataset) to prevent file leakage, split into 1 second clips, and create the class datasets (background, tanker, tug, passengership, cargo)
2) building cumulative dataset of all 5 separate datasets
3) Converting wav dataset into Mel-spectrogram dataset, 3-channel (Mel, CQT, Gammatone) spectrogram, and MFCCs
4) Training all Deep learning architectures and Machine learning algorithms on the datasets and various hyperparameters
5) Evaluation and metrics


**1. Processing of RAW wav file datasets (folder: dataset_processing)**

- For **VTUAD processing (folder: VTUAD_processing)**, I applied the same processing to inclusion_2000_exclusion_4000, inclusion_3000_exclusion_5000, inclusion_4000_exclusion_6000 datasets, since the structure and format are the same.

**Run order:**

**1. split_ship_ids.py**

**2.  distribute_ship_ids_train.py**

**3.  distribute_ship_ids_validation.py**

**4.  distribute_ship_ids_test.py**


a) I first used the metadata.csv files to identify the class, MMSI, and the source wav file

```
for row in reader:
                total_rows += 1
                cls = row["label"].strip()
                mmsi = mmsi_to_folder(row["MMSI"])
                fidx = row["file_index"].strip()
                src = audio_dir / cls / f"{fidx}.wav"
                if not src.is_file():
                    missing.append(str(src))
                    continue
                try:
                    order_key = (SPLIT_ORDER[split], int(fidx))
                except ValueError:
                    order_key = (SPLIT_ORDER[split], fidx)
                groups[(cls, mmsi)].append((order_key, src))
```
b) Then, I created new class folders containing each MMSI as independent folders with the corresponding WAV files within them

```
for (cls, mmsi), items in sorted(groups.items()):
        items.sort(key=lambda x: x[0])                 
        dest_dir = OUT / f"{cls}_ship_ids" / f"ship_id_{mmsi}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        for new_idx, (_, src) in enumerate(items):
            dest = dest_dir / f"{mmsi}_{new_idx}.wav"
```

- This is to prevent dataset leakage, so no two clips from the same ship ID are leaked into train, validation, and test datasets

c) To create the train dataset, I first partioned the background class's singular MMSI identifier's WAV files into 3 sections (one for each split), and then chose the required number of WAV files from 1/3 of the pool

```
all_wavs = []
    for s in ships:
        all_wavs.extend(wavs_in(s))          
    if not all_wavs:
        print(f"[background] no unused wavs left in {class_dir} -> skip", flush=True)
        return 0

    pool_size = len(all_wavs) // DIVISOR      
    if pool_size < 1:
        pool_size = 1
    pool = rng.sample(all_wavs, pool_size)
    rng.shuffle(pool)
```

d) Then, for the other classes, I iterate through the ship IDS for that class and partition the ship IDS by 3. Then, I define a quota of files that should be taken from each ship ID to evenly distribute the target among the ship IDs.

```
for cls, target in TARGETS.items():
        class_dir = OUT / f"{cls}_ship_ids"
        dest_dir = TEST_DIR / cls
        dest_dir.mkdir(parents=True, exist_ok=True)

        # background: single ship id
        if cls == BACKGROUND_CLASS:
            grand_total += build_background(class_dir, dest_dir, target, rng)
            continue

        ships = list_ship_dirs(class_dir)     
        if not ships:
            print(f"[{cls}] no available ship folders in {class_dir} -> skip", flush=True)
            continue

        # how many ships to use for test
        n_ships = len(ships) // DIVISOR 
        if n_ships < 1:
            n_ships = 1
        n_ships = min(n_ships, len(ships))

        chosen = rng.sample(ships, n_ships)            # random ship selection using random seed
        quota = target // n_ships
```
e) I take the quota from each ship ID in the partition:
```
for s in chosen:
            avail = pool[s]
            take = min(quota, len(avail), target - copied)
            for p in avail[taken[s]: taken[s] + take]:
                shutil.copy2(str(p), str(dest_dir / p.name))
            taken[s] += take
            copied += take
            if copied >= target:
                break
```
f) Finally, I mark that ship ID as used so it can't be accessed again by another split
```
for s in chosen:
            used_name = s.parent / f"{s.name}_used"
            if not used_name.exists():
                s.rename(used_name)
```

- This same process is followed for both train, validation, and test, with DIVISORs of 3, 2, and 1.


- Next, for **ONC processing (folder: ONC_processing)**, I followed the same process used for VTUAD, since VTUAD is derived from the ONC generation pipeline. This means the custom ONC dataset shares the same format as the VTUAD datasets, so the same process can be applied.

**Run order:**
**1.  group_by_mmsi.py**

**2.  build_split.py**


- Finally, for **Deepship (folder: DeepShip_processing)**, I first, using the metafile, listed the specific ship names and file indices corresponding to those ship names in a dictionary with a list value

**Run order:**
**1.  categorize_ship_names.py**

**2.  split_into_1_second.py**

**3.  train_split.py**

**4. validation_split.py**

**5.  test_split.py**



```
ships = defaultdict(list)
        for index, name in parse_metafile(metafile):
            ships[name].append(index)
```
- Then, I create the ship name folder within the class folder and append the WAV files
```
for index in indices:
                src = os.path.join(class_dir, f"{index}.wav")
                if not os.path.isfile(src):
                    # Only part of the dataset ships with the repo.
                    missing.append(f"{class_name}/{index}.wav")
                    continue
                dst = os.path.join(ship_dir, f"{index}.wav")
                if args.dry_run or place_wav(src, dst, args.mode):
                    placed += 1
```
- Then, we split all WAVs into 1 second clips depending on their current duration.
```
def clip_recording(audio):
    for i in range(count_clips(len(audio) / TARGET_SR)):
        yield audio[i * CLIP_SAMPLES:(i + 1) * CLIP_SAMPLES]


def count_clips(duration_seconds):
    return int(duration_seconds / CLIP_SECONDS)
```
- Then, we output the clips to their respective directory inside the ship name folders

```
for clip in clip_recording(audio):
                    dst = os.path.join(
                        ship_dir, f"{class_name}_{ship_name}_{clip_index}.wav")
                    clip_index += 1

```

- Note that all clips are downsampled to 16 kHz

- Then, to create the train, validation, and test folders, the same process is followed as the VTUAD and ONC datasets: partition by 3, and distribute evenly among ship IDs within the partition.

- Finally, we combine all class folders from each dataset into one dataset, forming the cumulative dataset.




- **Now, we transform the cumulative dataset into the three representations: (folder: preprocessing)**
  - Mel spectrogram
  - MFCCs
  - 3-channel representation (Mel, CQT, Gamma)


**Run Order:**

**1.  create_mel.py**

**2.  create_3_channel.py**

**3.  create_MFCC.py**
 
- For the mel spectrogram representation, we first define our Short Time Fourier Transform parameters:
```
SR = 16000
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 128           
N_MELS = 128
```

- Then, we use librosa to convert wav to mel_spectrogram:
```
S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window="hann", center=True, power=2.0,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )

    if LOG_SCALE:
        S = librosa.power_to_db(S, ref=np.max)

```
- We simply iterate through all wav file, convert them to mel_spectrogram .npy files, and input then into the new dataset. Then, we compress into .tar file:
```
with tarfile.open(LOCAL_TAR, "w") as tar:               
        tar.add(LOCAL_OUT, arcname=LOCAL_OUT.name)
    print(f"  {LOCAL_TAR.stat().st_size / 1e6:.0f} MB in {time.time() - t0:.0f}s", flush=True)

    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"\ncopying tar -> {DRIVE_TAR} ...", flush=True)
    t0 = time.time()
    sh(f'cp "{LOCAL_TAR}" "{DRIVE_TAR}"')
    print(f"  copied in {time.time() - t0:.0f}s", flush=True)
```
- Now for the 3-channel representation, we first set the specific parameters for our three channels:

```
N_MELS = 128
MEL_FMIN, MEL_FMAX = 0, SR // 2

CQT_N_BINS = 128
CQT_BINS_PER_OCTAVE = 16
CQT_FMIN = 30.0

N_GAMMA = 128
GAMMA_FMIN, GAMMA_FMAX = 30.0, SR // 2
GAMMA_ORDER = 4
```

- Then, we use librosa to compute the mel spectrogram of the image:
```

mel = librosa.feature.melspectrogram(S=power, sr=SR, n_mels=N_MELS,
                                         fmin=MEL_FMIN, fmax=MEL_FMAX)

```
- Then we load Gammatone parameters and weights:

```

def gammatone_weights():
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    cf = _erb_space(GAMMA_FMIN, GAMMA_FMAX, N_GAMMA)
    bw = 1.019 * _erb(cf)
    w = (1.0 + ((freqs[None, :] - cf[:, None]) / bw[:, None]) ** 2) ** (-GAMMA_ORDER / 2.0)
    w /= np.maximum(w.max(axis=1, keepdims=True), 1e-10)   
    return w
```

- And finally, the CQT from librosa:
```
cqt = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP_LENGTH, fmin=CQT_FMIN,
                             n_bins=CQT_N_BINS,
                             bins_per_octave=CQT_BINS_PER_OCTAVE)) ** 2

```
- Now, we have to compress these representations into 3-channels:
```
t = min(mel.shape[1], cqt.shape[1], gam.shape[1])
    mel, cqt, gam = mel[:, :t], cqt[:, :t], gam[:, :t]

    chans = [to_unit_db(mel), to_unit_db(cqt), to_unit_db(gam)]
    return np.stack(chans, axis=0).astype(DTYPE)  
```
- Note that this array has 3 channels, and 128 vertical values representing frequency bands.
- The output is a 128 x 126 image, which is then scaled up to 224 x 224 in resolution.

- Finally, for the MFCC dataset, we first use librosa to extract the MFCCs from each wav:
```
m = librosa.feature.mfcc(
        y=y, sr=SR, n_mfcc=N_MFCC, dct_type=DCT_TYPE, norm=DCT_NORM, lifter=LIFTER,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window="hann", center=True, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )                                                  

```
- Then, we compress this into a 1-d array that expresses the clip index, the MFCC coefficient, and its magnitude

```
return np.concatenate([m.mean(axis=1), m.std(axis=1)]).astype(np.float32)
```

- Finally, we stack all of the 1d arrays, representing the x axis as clip index, y axis as coefficient, and the value as magnitude at that specific clip and coefficient:

```
raw[split] = np.stack(feats).astype(np.float32)
        labels[split] = np.asarray(labs, dtype=np.int64)
        names[split] = fnames
```

- Now we have all three of our representations



**Now, we can review how our Deep and Machine learning models are trained on these datasets: (folder: Deep_learning_training)**

**Run order:**

**- No specific run order is needed, each python file corresponds to the combination that is trained**

- For all Deep learning architectures, we record results in a google sheet of different learning rates and epochs:

```
SHEET_LRS = [0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001, 0.000001]  
SHEET_EPOCHS = [10, 20, 30, 40, 50]                                  

START_LR = 0.05
START_EPOCHS = 10
```
- Then, for microsoft's resnet50 and torchvision's vggnet19, since they are pretrained on the ImageNet dataset, we load the channel normalization weights in:

```
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

```
- We apply these weights 
```
arr = np.load(path).astype(np.float32)

if arr.ndim == 3 and arr.shape[2] == 3 and arr.shape[0] != 3:
    arr = np.transpose(arr, (2, 0, 1))         # HWC -> CHW
t = torch.from_numpy(np.ascontiguousarray(arr))

if t.ndim == 2:
    t = t.unsqueeze(0).repeat(3, 1, 1)
elif t.shape[0] == 1:
    t = t.repeat(3, 1, 1)

return self.normalize(self.resize(t)), label
```



- Then, we load the model in with its optimizer and loss function:
-  (ResNet50)
```
def build_model(num_classes):
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes, ignore_mismatched_sizes=True)
    return model.to(DEVICE)
```
```
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()                # no label smoothing
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

```

- Then, we train

```
for x, y in train_loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
        loss = criterion(forward(model, x), y)
    if not torch.isfinite(loss):
        continue
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```
- Then, we compute validation accuracy
```
@torch.no_grad()
def validate_accuracy(model, loader):
    model.eval()
    correct += (logits.argmax(1) == y).sum().item()
    return correct / max(total, 1)
  
```
- We then save the weights of the best model among its different LR and epoch combinations
```
if state is not None and acc > best["acc"]:
    best = {"acc": acc, "lr": lr, "epochs": epochs, "state": state}
    path = save_best(state, classes, lr, epochs, acc)

```
- This process is applied to all Deep learning training pipelines


- **Now, for all Machine Learning combinations, we apply the following pipeline (folder: Machine_learning_training)**
- 
**- Run order:**

**- No specific run order is needed, the python file names correspond to the combination trained**




- First, we load the feature matrices

```
def load_split(root, split):
    X = np.load(root / f"X_{split}.npy").astype(np.float32)
    y = np.load(root / f"y_{split}.npy")
    return X, y
```
- Then, we load the algorithm (random forest)
```
model = RandomForestClassifier(
    n_estimators=n_est, max_depth=max_depth, criterion=CRITERION,
    max_features=MAX_FEATURES, random_state=RANDOM_STATE, n_jobs=N_JOBS)
model.fit(Xtr, ytr)
```

- Finally, we evaluate

```
acc = float(model.score(Xva, yva))
```
- We then save the weights of the best machine learning instance (random forest)
```
joblib.dump(model, OUTPUT_DIR / "rf_mfcc_best.joblib")
np.save(OUTPUT_DIR / "rf_mfcc_best_importances.npy", model.feature_importances_)
```

**Now, for model evaluation (folder: model_evaluation):**

**- Run Order**

**- No run order is necessary, as each python file corresponds to the specific combination tested**

Deep learning evaluation follows the pipeline:

1) Load the test dataset (ResNet50 + 3-channel dataset)

```
root = stage_dataset()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    ds = McgDataset(root / TEST_SPLIT, class_to_idx, mean, std)
    if len(ds) == 0:
        raise SystemExit(f"no .npy files found in {root / TEST_SPLIT}")
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
    probe, _ = ds[0]
```
2) Load model (ResNet50)
```
model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=len(classes), ignore_mismatched_sizes=True)
    model.load_state_dict(state)
    model = model.to(DEVICE)
```

3) Save results in outputs (confusion matrix, etc.)

```
(OUTPUT_DIR / "test_classification_report.txt").write_text(header + "\n\n" + text + "\n")
    (OUTPUT_DIR / "test_classification_report.tex").write_text(
        format_latex(per_class, summary) + "\n")
    save_confusion_matrix(cm, classes, OUTPUT_DIR / "test_confusion_matrix.png",
                          "ResNet50 + 3-channel (Mel, CQT, Gamma) \u2014 test")
    (OUTPUT_DIR / "test_results.json").write_text(json.dumps({
        "model": ckpt.get("model"), "model_name": ckpt.get("model_name"),
        "representation": ckpt.get("representation"), "classes": classes,
        "hyperparameters": hp, "normalization": {"mean": mean, "std": std},
        "val_accuracy": ckpt.get("val_accuracy"),
        "test_accuracy": summary["accuracy"],
        "per_class": [{k: (float(v) if k != "support" and k != "name" else v)
                       for k, v in r.items()} for r in per_class],
        "macro_avg": {k: float(v) for k, v in summary["macro"].items()},
        "weighted_avg": {k: float(v) for k, v in summary["weighted"].items()},
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(ds)),
    }, indent=2))
```

For Machine Learning algorithms, we do the following:

1) Load test dataset (x and y)

```
coef = np.load(COEF_PATH).astype(np.float64)
    intercept = np.load(INTERCEPT_PATH).astype(np.float64)
    print(f"loaded coef {coef.shape}  intercept {intercept.shape}", flush=True)

    root = stage_dataset()
    Xte = np.load(root / f"X_{TEST_SPLIT}.npy").astype(np.float64)
    yte = np.load(root / f"y_{TEST_SPLIT}.npy")
```

2) Save outputs
```
(OUTPUT_DIR / "test_classification_report.txt").write_text(header + "\n\n" + text + "\n")
    (OUTPUT_DIR / "test_classification_report.tex").write_text(
        format_latex(per_class, summary) + "\n")
    save_confusion_matrix(cm, classes, OUTPUT_DIR / "test_confusion_matrix.png",
                          "Logistic Regression + MFCC \u2014 test")
    (OUTPUT_DIR / "test_results.json").write_text(json.dumps({
        "model": "logistic_regression", "representation": "mfcc",
        "classes": classes, "n_features": int(Xte.shape[1]),
        "coef_shape": list(coef.shape), "intercept_shape": list(intercept.shape),
        "val_accuracy": val_acc, "test_accuracy": summary["accuracy"],
        "per_class": [{k: (float(v) if k not in ("support", "name") else v)
                       for k, v in r.items()} for r in per_class],
        "macro_avg": {k: float(v) for k, v in summary["macro"].items()},
        "weighted_avg": {k: float(v) for k, v in summary["weighted"].items()},
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(yte)),
    }, indent=2))

```

- This is the end of the entire pipeline.


