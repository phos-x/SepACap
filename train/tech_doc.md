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

- Verification of Manifests

Ensure your .scp files look like this (strictly ID SPACE PATH):

    alto_song1 /kaggle/working/jaCappella/ballad/song1/alto.wav

    bass_song1 /kaggle/working/jaCappella/ballad/song1/bass.wav

If your song IDs or paths have spaces, the maxsplit=1 ensures the ID is the first word and the rest of the line is treated as one continuous path.
- Technical Documentation Context: Manifest Robustness

Empty Line Defensive Logic: Manifest parsers must explicitly skip \n characters at the end of files to prevent RuntimeError.

Token Guard: Use maxsplit=1 when parsing .scp files to ensure compatibility with file systems that may contain spaces in song titles or absolute paths.
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

Here is the updated, comprehensive technical documentation for the **SepACap** architecture. This document reflects all the structural upgrades, memory optimizations, and dimensional alignments we have implemented.

---

# Technical Architecture 

## 1. Overview

**SepACap** is an advanced adaptation of the SepReformer architecture, specifically re-engineered for **multi-singer acapella source separation** (using the jaCappella dataset). Unlike standard speech separation models that isolate 2 speakers, SepACap isolates **6 distinct singing stems** (Alto, Bass, Lead Vocal, Soprano, Tenor, Vocal Percussion) simultaneously from a single monaural mixture.

Key architectural upgrades include:

* **Direct Waveform Modeling:** End-to-end processing in the time domain.
* **Harmonic Extrapolation:** Utilization of **SNAKE** periodic activation functions to model musical pitch and harmonics.
* **Two-Stage Detached PIT:** A highly optimized Permutation Invariant Training pipeline that prevents GPU Out-of-Memory (OOM) errors.
* **Composite Loss:** A weighted combination of Waveform L1, Multi-Resolution Spectral L1, and Psychoacoustic Mel-scale Loss.

---

## 2. The Data Flow: Step-by-Step

How a raw audio mixture transforms into 6 isolated stems.

### Step 1: Input ingestion

* **Input Shape:** `[Batch, 1, 24000]`
* Raw audio is loaded at 8000 Hz. With a `max_len` of 24,000, the network processes exactly 3 seconds of audio per forward pass.

### Step 2: Audio Encoder (`AudioEncoder`)

* **Operation:** A 1D Convolution (`kernel_size=16`, `stride=4`) acts as a learned STFT/patcher.
* **Activation:** GELU is applied to the extracted features.
* **Shape Transition:** `[Batch, 1, 24000]`  `[Batch, 256, 6000]`

### Step 3: Feature Projection (`FeatureProjector`)

* **Operation:** Group Normalization followed by a  Convolution.
* **Purpose:** Compresses the 256 channels down to 128 to save memory before entering the heavy Transformer blocks.
* **Shape Transition:** `[Batch, 256, 6000]`  `[Batch, 128, 6000]`

### Step 4: The Separator Engine (`Separator`)

This is the core of the network, consisting of 4 Encoder Stages, a Bottleneck, and 4 Decoder Stages.

* **Internal Transposition:** The network rigorously transposes tensors to satisfy different operational requirements:
* **Global Blocks (Transformers):** Require sequence-first format `[Batch, Time, Channels]`. Captures long-range dependencies (e.g., song structure).
* **Local Blocks (CNNs):** Require channel-first format `[Batch, Channels, Time]`. Captures local acoustic textures (e.g., breath sounds, consonants).


* **Downsampling/Upsampling:** At each stage, the sequence length is halved (DownConv) and channels remain constant, building a multi-scale hierarchical representation.

### Step 5: Speaker Splitting (`SpkSplitStage`)

* **Operation:** A GLU-based gating network expands the separated latent features into 6 distinct latent spaces.
* **Shape Transition:** `[Batch, 128, Time]`  `[Batch * 6, 128, Time]`

### Step 6: Output Layer & Masking (`OutputLayer`)

* **Operation:** The 6 separated latent spaces are mapped to estimated masks, which are multiplied back against the original Encoder features. This ensures only the vocal features belonging to a specific stem are passed through.

### Step 7: Audio Decoder (`AudioDecoder`)

* **Operation:** A 1D Transposed Convolution (Deconvolution) using the exact inverse parameters of the Encoder (`kernel=16`, `stride=4`).
* **Shape Transition:** `[Batch * 6, 256, 6000]`  `[Batch, 6, 24000]`
* **Output:** The final separated audio stems.

---

## 3. Loss Function: Detached PIT & Composite Loss

Because calculating gradients for 6 stems yields 720 possible permutations, SepACap uses a **Two-Stage Detached Assignment** to prevent GPU memory crashes:

1. **Stage 1 (No Gradients):** Calculates the pairwise distances between the 6 estimates and 6 targets (36 operations). It tests all 720 combinations purely in memory to find the mathematically optimal `best_perm`.
2. **Stage 2 (With Gradients):** Builds the massive computational graph only for the 6 target-estimate pairs defined by the `best_perm`.

**The Components of the Composite Loss:**

* **Waveform L1 (Weight: 1.0):** Direct absolute error between the output audio and target audio.
* **Mel Loss (Weight: 0.7):** Converts audio to an 80-bin Mel-spectrogram (mimicking human hearing) and calculates the error.
* **Multi-Res Spectral (Weight: 0.3):** Uses 3 STFT window sizes (512, 1024, 2048) to calculate both Magnitude and Log-Magnitude L1 losses. This forces the model to learn both sharp transients (percussion) and sustained harmonics (vocals).

---

## 4. Configuration Breakdown (`configs.yaml`)

### A. Dataset & Dataloader

* `max_len` (24000): The maximum length of audio samples per chunk. Set to 3 seconds to optimize attention memory.
* `sampling_rate` (8000): The Hz rate for the jaCappella dataset.
* `scp_dir`: Path to the definition files linking audio paths to the training engine.
* `dynamic_mixing` (true): Augments data by randomly mixing stems from different songs during training to prevent overfitting.
* `batch_size` (1): The number of mixtures processed at once. Locked to 1 to fit the 6-stem architecture inside a 16GB GPU.
* `num_workers` (12): How many CPU threads are dedicated to loading audio files from disk.

### B. Model Dimensions

* `num_stages` (4): The depth of the U-Net style Separator hierarchy.
* `num_spks` (6): The number of target stems.
* `activation` ("SNAKE"): The periodic activation function used inside the Separator blocks.
* `module_audio_enc` / `module_audio_dec`:
* `out_channels` (256): Number of latent feature maps.
* `kernel_size` (16) & `stride` (4): Controls the time-resolution of the latent space.
* `bias` (false): Excluded to prevent shifting the zero-mean audio signals.



### C. Separator Modules

* `relative_positional_encoding`: Injects timing information into the Transformer so it understands the sequence of audio chunks (`maxlen: 2000`).
* `enc_stage` / `dec_stage`:
* `num_mha_heads` (8): Number of attention heads in the Global Transformer blocks.
* `kernel_size` (65): Receptive field of the Local CNN blocks. Large kernel captures wider acoustic context.
* `dropout_rate` (0.05): Randomly zeroes out 5% of neurons during training to prevent memorization.



### D. Criterion (Loss Setup)

* `name`: `["SepACapCompositeLoss", "SepACapCompositeLoss", "PIT_SISNRi", "PIT_SDRi"]`. Padded array to satisfy the Engine's 4-metric unpacking expectation.
* `weights`: Controls the priority of the loss functions (`waveform: 1.0`, `mel: 0.7`, `spectral: 0.3`).
* `window_sizes`: `[512, 1024, 2048]`. The resolutions for the Multi-Res Spectral Loss.

### E. Optimizer & Engine Configuration

* `AdamW`: The optimizer used to update weights, configured with a learning rate (`lr: 1.0e-3`) and `weight_decay: 1.0e-2` for regularization.
* `ReduceLROnPlateau`: Reduces the learning rate automatically if the validation loss stops improving for 2 epochs (`patience: 2`).
* `WarmupConstantSchedule`: Slowly increases the learning rate for the first `1000` steps to prevent early divergence.
* `max_epoch` (200): Total full passes over the dataset.
* `clip_norm` (5): Prevents exploding gradients by capping their maximum value.

---
