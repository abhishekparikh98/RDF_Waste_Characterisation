# Release Checklist — `v1.0-models`

This is the **only manual step** left to make the repo fully clone-and-run.
The committed `app.py` code will auto-pull these files on first request from
the GitHub Release tagged **`v1.0-models`**.

Without this Release published, the auto-download fails and the user sees the
fallback message from `app.py` telling them to either publish the Release,
drop the weights manually, or retrain.

---

## Files to attach

| File | Source path on your machine | Approx. size |
|---|---|---|
| `yolo_best.pt` | `D:\University\Msc Project\models\yolo_best.pt` | ~6.2 MB |
| `cnn_baseline_best.h5` | `D:\University\Msc Project\models\cnn_baseline_best.h5` | ~309 MB |

---

## Steps

### 1. Confirm the files are present

```powershell
Test-Path "D:\University\Msc Project\models\yolo_best.pt"          # True
Test-Path "D:\University\Msc Project\models\cnn_baseline_best.h5"   # True
```

If either is missing, recover it first:

- `yolo_best.pt` — from your last training run output, or rerun
  `scripts/train_yolo.py` (multi-hour, needs `data/yolo/` which itself comes
  from `scripts/convert_taco_to_yolo.py`).
- `cnn_baseline_best.h5` — from your last CNN training, or rerun
  `scripts/train_cnn.py` after running `scripts/preprocess_dataset.py`.

### 2. Verify both files actually load (catches corruption early)

```powershell
cd "D:\University\Msc Project"
.\.venv\Scripts\Activate.ps1     # activate the venv you used to train
python -c "from ultralytics import YOLO; YOLO(r'models\yolo_best.pt')"
python -c "import tensorflow as tf; tf.keras.models.load_model(r'models\cnn_baseline_best.h5').summary()"
```

Both commands should print model info with no traceback.

### 3. Open the Release creation page

Go to:

```
https://github.com/abhishekparikh98/RDF_Waste_Characterisation/releases/new
```

### 4. Fill in the form

| Field | Value |
|---|---|
| **Choose a tag** | type `v1.0-models` and select "**Create new tag: v1.0-models on publish**" |
| **Target** | `master` |
| **Release title** | `v1.0-models — Trained Waste Detection Weights` |
| **Description** | copy from the block below |

**Description block** (copy-paste as-is):

```markdown
## Trained model weights for `app.py`

These two files back the Flask inference app on a fresh clone. `app.py` downloads
them automatically on first request from
`https://github.com/abhishekparikh98/RDF_Waste_Characterisation/releases/download/v1.0-models/<file>`.

### Files

| File | Used by | Notes |
|---|---|---|
| `yolo_best.pt` | YOLOv8 multi-object detector (primary, runs at confidence >= 0.45) | YOLOv8n fine-tuned on TACO + TrashNet, 6 classes |
| `cnn_baseline_best.h5` | Legacy CNN (parallel, supplies Grad-CAM, YOLO fallback) | ResNet50-style baseline, single-object |

### How the download works

- Triggered the first time the upload form is submitted on a fresh clone.
- Cached under `models/` — only one download per machine.
- If this Release is not yet published, `app.py` raises a clear error pointing
  back here. See `docs/RELEASE_CHECKLIST_v1.0-models.md` in the repo.

### Integrity

- SHA-256 checksums are listed below for verification before you trust them.
- The repo's `.gitignore` keeps both files out of source control on purpose.

<!-- Run Get-FileHash -Algorithm SHA256 on each file before publishing, paste
     the hashes here. -->
```

### 5. Attach both files

In the "Attach binaries" section at the bottom of the Release page:

1. Drag-and-drop **`yolo_best.pt`** (or click "attach binaries" and select it).
2. Drag-and-drop **`cnn_baseline_best.h5`**.

GitHub's per-upload limit is **2 GB**, so both files fit comfortably.

### 6. Compute SHA-256 (optional but useful)

Before publishing, run on your local machine:

```powershell
Get-FileHash "D:\University\Msc Project\models\yolo_best.pt" -Algorithm SHA256
Get-FileHash "D:\University\Msc Project\models\cnn_baseline_best.h5" -Algorithm SHA256
```

Paste the hashes into the description so reviewers can verify file integrity.

### 7. Publish

Click **"Publish release"** (NOT "Save draft" — drafts aren't downloadable).

### 8. Verify the download URLs work

In a browser, both of these should return a 200 + a file download:

```
https://github.com/abhishekparikh98/RDF_Waste_Characterisation/releases/download/v1.0-models/yolo_best.pt
https://github.com/abhishekparikh98/RDF_Waste_Characterisation/releases/download/v1.0-models/cnn_baseline_best.h5
```

### 9. Smoke-test the auto-download

Easiest way is a temporary machine (or a `git clone` into a new folder). The
minimum-flow check:

```bash
git clone https://github.com/abhishekparikh98/RDF_Waste_Characterisation.git test-clone
cd test-clone
pip install -r requirements.txt
python app.py
```

Then upload any image in the browser and confirm:

- No `FileNotFoundError` is raised.
- Inference completes (you get a class label + confidence).
- `models/yolo_best.pt` and `models/cnn_baseline_best.h5` now exist locally.

If you don't have access to a second machine, delete one of the two files on
your own machine before running `python app.py` and confirm it gets
re-downloaded.

---

## After publishing

- The Release is **immutable** — if you need to fix a weight file, publish a
  new Release (`v1.1-models` etc.) and either bump `RDF_WEIGHTS_RELEASE_TAG`
  in `.env.example` and `WEIGHTS_RELEASE_TAG` in `app.py`, or live with users
  re-downloading the corrected file manually.
- Add a short note to your dissertation appendix or viva talking-point sheet:
  *"Trained weights are released under tag `v1.0-models`. Source of truth:
  this GitHub Release, not the git tree."*

---

## Rollback (if a Release needs to be unpublished)

GitHub does not allow editing a published Release's binary assets in place.
The clean rollback is:

1. Re-train or restore a known-good copy of the weight.
2. Publish a new Release (e.g. `v1.0.1-models`) with the corrected file.
3. Tell users in the Release description: "Use `v1.0.1-models`."
4. After a grace period, delete the old Release via the "Delete release"
   button in the Release sidebar.

If a weight file is uploaded corrupt or wrong, click **"Delete release"** on
the bad Release's page and republish under a new tag.
