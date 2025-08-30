## Hybrid Face Recognition Spec (SVC ↔ CNN)

Purpose: Define a consistent, production-ready pipeline that adapts between HOG+SVC for small datasets and CNN for larger datasets, with identical detection/alignment and photometric normalization to ensure robustness across lighting and sessions.

### Scope
- Same detection and alignment for both classifiers
- Unified preprocessing producing a canonical face crop
- Method-specific finalization steps (HOG vs. RGB normalization)
- Data thresholds and automatic model switching
- Training/inference parity and operational guidelines

---

### 1) Detection & Alignment (Common)
- Detector: Haar Cascade for faces; eye detection for alignment.
- Steps:
  1. Detect face bounding box; add 5–10% margin; crop.
  2. Detect eyes; rotate so the eye line is horizontal.
  3. Resize canonical crop:
     - SVC path: 128×128
     - CNN path: 224×224
- Quality gates (reject frame if any fails):
  - Min face size ≥ 100 px height/width
  - Blur check: variance of Laplacian ≥ threshold
  - Exposure check: mean luminance in [low, high] bounds

Rationale: Keeping detection and alignment identical ensures both SVC and CNN see comparable inputs across training and inference.

---

### 2) Photometric Normalization (Common)
Apply to the canonical face crop before branching:
- CLAHE on luminance only: convert to YCrCb or HSV; enhance Y/V; merge back.
- Adaptive gamma (≈0.8–1.4) based on brightness to stabilize exposure.
- Optional light denoise (bilateral/NLM) and subtle sharpen to restore edges.

Notes:
- Use identical settings for training and inference.
- Prefer luminance-space operations to avoid color shifts.

---

### 3) Branch Finalization
- SVC (HOG):
  - Convert to grayscale after normalization.
  - Extract HOG (fixed cell/block/orientation parameters).
  - L2-normalize HOG vector.
  - Feature standardization: store training mean/std; apply at inference.

- CNN:
  - Keep RGB; scale pixels to [0, 1].
  - Optional per-channel mean/std normalization using training stats.

---

### 4) Dataset Strategy & Storage
- Capture raw images per user: `static/faces_raw/{name$id$section}/`.
- Offline preprocessing → canonical crops: `static/faces_proc/{name$id$section}/`.
- Do not persist augmented images. Perform augmentations in-memory during training.
- Keep both raw and processed datasets; the processed set feeds both SVC and CNN.

---

### 5) Training Protocols
- SVC (small data ≤ ~300 images or ≤ 3 users):
  - Train on HOG features extracted from `faces_proc`.
  - Cache HOG features and labels to disk for fast retrains.
  - Class balance: similar samples per user; drop low-quality frames.

- CNN (≥ ~4 users or > 300 images total):
  - Train on RGB canonical crops from `faces_proc`.
  - Use light augmentations only during training: ±20–30% brightness/contrast, slight rotations/zoom, small noise/blur, mild color jitter.
  - Validation split by capture session/lighting, not by random image.

Switching guideline: If user count or total image count exceeds the thresholds, prefer CNN. Optionally train both during the transition and compare validation accuracy/F1 before switching serving model.

---

### 6) Inference Protocols
- Apply the identical detection, alignment, and photometric normalization.
- SVC path: grayscale → HOG → standardize → SVC.predict_proba.
- CNN path: RGB → scale/normalize → model.predict.
- Unknown handling:
  - SVC: threshold on max probability or margin between top-1 and top-2.
  - CNN: confidence threshold and top-1 vs top-2 margin.
- Temporal smoothing: require N consecutive frames before confirming identity; decay lock if faces are lost for M frames.

---

### 7) Lighting Robustness Checklist
- Enroll users under at least 3 lighting conditions (front-lit, dim, side-lit).
- Use CLAHE + adaptive gamma in preprocessing.
- Keep augmentations realistic; avoid heavy synthetic shadows.
- Monitor performance; add a few new samples if environment lighting changes.

---

### 8) Data & Thresholds (defaults; tune per deployment)
- Min face size: 100 px
- Blur threshold (variance of Laplacian): tune from sample frames
- Exposure range (mean luminance): tune from cameras/rooms
- Temporal smoothing: NEED_CONSEC ≈ 5 frames; lock grace ≈ 10 frames
- SVC probability/score threshold: tune to meet FAR/FRR goals
- CNN confidence threshold ≈ 0.7; margin ≈ 0.15 (adjust with validation)

---

### 9) Switching Logic (SVC ↔ CNN)
- Heuristic thresholds:
  - Use SVC if users ≤ 3 or total images ≤ 300
  - Use CNN if users ≥ 4 or total images > 300
- Optional parallel evaluation during transition (A/B) to verify no regression.
- Persist the active model tag and preprocessing config with versioning.

---

### 9.1) Single-User Mode (Verification)
- When only 1 user is registered, do not train a multiclass classifier.
- Enrollment:
  - After preprocessing (detect → align → normalize), compute features per image
    (HOG vector or deep embedding) and store a template (e.g., mean vector).
- Inference:
  - Preprocess the live face identically; compute feature vector; compare to template
    (cosine similarity or Euclidean distance).
  - If similarity ≥ threshold across N consecutive frames, accept; else “Unknown.”
- Attendance behavior:
  - On acceptance, mark attendance once per day for that user (idempotent).
  - Keep unknown path and multi-frame confirmation to avoid false positives.
- Transition:
  - When a second user is added, retrain to multiclass identification (SVC first,
    then CNN when data grows). Maintain the same preprocessing.

---

### 10) Operational Guidance
- Consistency: the exact same preprocessing at train and inference.
- Caching: store HOG features and training stats (means/stds) for SVC; store CNN preprocessing config alongside the model.
- Quality: reject low-quality frames during preprocessing to avoid training on junk.
- Governance: record dataset stats (num users, images/user, lighting coverage) and model metrics (accuracy/F1 per class). Version datasets and models.
- Safety: always include an “Unknown” path with thresholds and multi-frame confirmation.

---

### 11) Deliverables for Any Implementation
- Preprocessing module: detection, alignment, normalization (shared).
- Feature module (SVC): HOG extraction + standardization utilities.
- Model trainers: SVC trainer (with caching) and CNN trainer (with augmentations).
- Inference service: unified pipeline with runtime switching based on active model.
- Config: thresholds, image sizes, augmentation ranges, and switching rules.
- Documentation: this spec plus environment-specific parameter values.

---

### 12) Acceptance Criteria
- Reproducible preprocessing across train and inference.
- Meets target accuracy/F1 under varied lighting on a held-out session-based validation set.
- Stable unknown rejection at chosen thresholds.
- Clean switch between SVC and CNN without regressions (validated by A/B or backtests).


