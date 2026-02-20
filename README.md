# 🎙️ SepACap: 6-Stem Acapella Source Separation

*An advanced adaptation of the NeurIPS 2024 **SepReformer** architecture, engineered specifically for multi-singer acapella extraction.*

SepACap takes a single monaural audio mixture and successfully isolates it into **6 distinct singing stems** (Alto, Bass, Lead Vocal, Soprano, Tenor, and Vocal Percussion) utilizing the jaCappella dataset.

Unlike standard speech separation models that isolate 2 speakers, SepACap incorporates musical harmonic modeling via SNAKE periodic activations and a Two-Stage Detached Permutation Invariant Training (PIT) loss to prevent GPU memory limits during complex 6-factorial separation.

---

## ☁️ The GitOps Workflow (Kaggle + VSCode)

This project is optimized for cloud execution rather than local hardware. It utilizes a modern **GitOps approach**, bridging local development in VSCode with heavy-duty training on **Kaggle Notebook GPUs**.

**How it works:**

1. Code is edited and managed locally via **VSCode**.
2. Updates are pushed to a Git repository.
3. The **Kaggle Notebook** acts as the execution engine, pulling the latest repository changes directly into the GPU's storage environment.
4. *Note: All primary execution steps, environment setups, and Git pulling commands are documented and run directly inside the provided Kaggle Notebook.*
5. *You can check out steps to connect external editors like Colab or VSCode to the same Jupyter Server that powers your Kaggle notebook on kaggle official docs.*

---

## 📂 Repository Structure

* **`📁 /data` (The Fuel):** Handles dataset preparation. Contains scripts to mix isolated singing stems into training mixtures and generates the lightweight `.scp` (Script) files that point the dataloader to the audio without crashing the RAM.
* **`📁 /models` (The Brains):** The core neural network. Contains the specific model versions (e.g., `SepReformer_Base_WSJ0` which we adapted for SepACap), the PyTorch network `modules/`, the `configs.yaml` master control panel, and the `log/scratch_weights/` where the model saves its learned checkpoints.
* **`📁 /sample_wav` (The Testing Ground):** The input/output tray for human evaluation. Drop a mixed song in here, and the network will spit out the 6 isolated `.wav` stems for you to listen to.
* **`📁 /utils` (The Toolbox):** Contains the infrastructure scripts. Includes the `util_engine.py` (the training loop foreman), `util_implement.py` (the dynamic PyTorch object factory), and `criterions.py` (the custom Loss Functions).

---

## 🚀 Getting Started

### 1. Git LFS Requirement

This repository uses **Git LFS (Large File Storage)** to manage the massive pretrained model weight files (`.pth`). If Git LFS is not installed in your Kaggle environment before cloning, the weights will not download properly.

```bash
# Inside your Kaggle Notebook terminal/cell
sudo apt update
sudo apt install git-lfs
git lfs install

```

### 2. Execution Commands

While the full pipeline is handled inside the Kaggle Notebook, here are the core commands used to trigger the engine via `run.py`:

**To Train the Network:**

> *Make sure your `.scp` file paths are correctly set in `models/SepACap_Base/configs.yaml` before running.*

```bash
python run.py --model SepACap_Base --engine-mode train

```

**To Run Inference on a Single Song:**

> *This will separate the audio and save the 6 output `.wav` files to your directory.*

```bash
python run.py --model SepACap_Base--engine-mode infer_sample --sample-file "filename"

```

**To Evaluate on the Test Dataset:**

> *Runs validation metrics without saving the heavy audio files.*

```bash
python run.py --model SepACap_Base --engine-mode test

```

---

## 🧠 Technical Highlights

SepACap heavily modifies the base SepReformer with the following upgrades:

* **SNAKE Activations:** Replaced standard ReLU/GELU in the separator blocks with periodic Snake activations to better extrapolate musical pitch and harmonics.
* **Composite Loss:** A highly tuned loss function blending Waveform L1 (1.0), Psychoacoustic Mel-scale (0.7), and Multi-Res Spectral L1 (0.3).
* **Two-Stage Detached PIT:** Optimizes the  (720) permutation calculations by detaching the gradient graph during the pairing phase, dropping VRAM usage from ~14GB to ~50MB during loss calculation.

---

## 📜 Acknowledgments & Citation

SepACap is built upon the foundational research of **SepReformer**. If you find the core asymmetric encoder-decoder architecture helpful, please cite the original authors' NeurIPS 2024 paper:

```bibtex
@inproceedings{
shin2024separate,
title={Separate and Reconstruct: Asymmetric Encoder-Decoder for Speech Separation},
author={Ui-Hyeop Shin and Sangyoun Lee and Taehan Kim and Hyung-Min Park},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=99y2EfLe3B}
}

```
