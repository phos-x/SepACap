This documentation is designed to serve as a **Source of Truth** for the **SepReformer/SepACap** project. It distills the technical hurdles we've solved into a clear set of standards and architectural definitions to ensure stable training and inference.

---

## 1. Data Integrity Standard (Pre-Flight Requirements)

To avoid the crashes related to `KeyErrors`, `Dimension Mismatches`, and `ValueError`, your data must adhere to the "Parallel Triplet" rule.

### The "Parallel Triplet" Rule

For every mixture file, there **must** be a corresponding clean source file for each speaker.

* **File Count Alignment:** Folder `Mix`, Folder `S1`, and Folder `S2` must have the exact same number of files.
* **Alphabetical Order:** When sorted alphabetically, the  file in `Mix` must be the combination of the  files in `S1` and `S2`.

### Signal Technical Specifications

| Requirement | Value | Purpose |
| --- | --- | --- |
| **Sampling Rate** | 8000 Hz | Ensures STFT kernels match frequency resolution. |
| **Bit Depth** | 16-bit PCM | Standardizes dynamic range. |
| **Channels** | Mono | The encoder only accepts single-channel audio. |
| **Minimum Length** | 1024 Samples | Prevents "Kernel size > input size" convolution crashes. |
| **Stride Rule** | Multiple of 4 | Matches the downsampling factor of the Transformer stages. |

- Mandatory Resampling: Raw jaCappella data is provided at 48kHz. Before manifest generation, all files must be downsampled to 8kHz. Failure to do so results in an SR Mismatch error during validation and invalid spectral loss gradients during training.

- Kernel Alignment: The STFT module in your criterions.py uses fixed window sizes (512, 1024, 2048). At 8kHz, these windows represent roughly 64ms to 256ms of audio, which is the "sweet spot" for capturing vocal transients and vibrato.

- Power Set Consistency: When the dataset.py sums multiple stems for the Power Set augmentation, all stems must have the same sampling rate to avoid alignment drift.

- Memory Optimization: Reducing the sample rate from 48kHz to 8kHz reduces your memory footprint by 6x, which is vital for maintaining the batch_size: 2 on standard Kaggle GPUs.

---

## 2. Dataset Preparation & Manifesting

We use `.scp` files to index the data. To avoid the "Manifest Mismatch" errors, follow this protocol:

1. **Normalization:** Use **Absolute Paths** in `.scp` files (e.g., `/kaggle/working/data/sample.wav`). This prevents "File Not Found" errors when the script moves between directories.
2. **Synchronization:** Use a **Force Alignment** script. Never assume the folders match; always use a script to intersect the filenames and create manifests that are perfect "twins."
3. **The Index-Matching Logic:** Our updated `dataset.py` ignores filename prefixes (like `050_` vs `051_`) and links files purely by their position in the sorted list.

---

## 3. Architecture Breakdown

The SepReformer is a "U-Net style" Transformer designed specifically for audio separation.

### A. Audio Encoder (The Translator)

* **What it does:** Converts raw time-domain waves (1D) into high-dimensional feature maps (2D).
* **Contribution:** It breaks the complex audio into "bits" that the Transformer can understand, similar to how a human ear breaks sound into different frequencies.

### B. The Separator (The Brain)

This is the core Transformer engine. It consists of:

* **Global Blocks:** Look at the "big picture" (the whole song) to identify which singer is which.
* **Local Blocks:** Look at the "small details" (short notes and textures) to clean up the edges of the sound.
* **Speaker Attention:** Dynamically focuses on one voice while "muting" the other based on learned patterns.

### C. Audio Decoder (The Reconstructor)

* **What it does:** Converts the processed high-dimensional features back into raw 1D audio.
* **Contribution:** It takes the "cleaned" bits from the separator and weaves them back into a sound file you can play.

---

## 4. Error Prevention Logic (The "Safety Shields")

We implemented three specialized "Shields" in the code to handle the errors we encountered:

1. **The "V" Key Check (Security Shield):** In `util_engine.py`, we set `weights_only=False` to allow loading older model checkpoints that contain complex Python metadata.
2. **The Shape-Sync Utility (Math Shield):** In `criterions.py`, the `sync_tensors` function center-crops signals if they differ by a few samples. This prevents the `RuntimeError: size mismatch` that happens when STFT math "shaves off" the end of a file.
3. **The Length Floor (Stability Shield):** In `dataset.py`, we pad any audio file shorter than 1024 samples. This prevents the `Conv1d` crash where the "kernel" tries to scan audio that is physically too short for it.

---

## 5. Summary Checklist for Success

* [ ] **Validate SR:** Are all files strictly 8000Hz?
* [ ] **Check Counts:** Do all data folders have the same number of files?
* [ ] **Absolute Paths:** Does your `.scp` file start with `/kaggle/working/...`?
* [ ] **GPU Memory:** If you hit "Out of Memory," reduce `batch_size` in `configs.yaml` from `2` to `1`.
* [ ] **Retries:** Use the provided Bash Loop to handle temporary system hiccups or Google Colab timeouts.

**Would you like me to generate a "Quick Start" bash script that performs all these data checks and manifest syncs in one single command?**