# 🎙️ SepACap: 6-Stem Acapella Source Separation

*An advanced adaptation of the NeurIPS 2024 **SepReformer** architecture, engineered specifically for multi-singer acapella extraction and deployed via an Enterprise AWS MLOps pipeline.*

SepACap takes a single monaural audio mixture and successfully isolates it into **6 distinct singing stems** (Alto, Bass, Lead Vocal, Soprano, Tenor, and Vocal Percussion) utilizing the jaCappella dataset.

Unlike standard speech separation models that isolate 2 speakers, SepACap incorporates musical harmonic modeling via SNAKE periodic activations and a Two-Stage Detached Permutation Invariant Training (PIT) loss to prevent GPU memory limits during complex 6-factorial separation.

---

## ☁️ The Enterprise MLOps Workflow

This project has evolved from a local research script into a fully automated, cloud-native CI/CD ecosystem. It utilizes **GitHub Actions** to orchestrate heavy-duty distributed training on **AWS SageMaker**, automated hyperparameter tuning via **Optuna**, and highly optimized C++ production inference using **ONNX Runtime**.

**The Pipeline:**

1. **Research (`dev` branch):** Code pushes trigger GitHub Actions to securely spin up AWS SageMaker GPU instances, run the PyTorch training loop, and log metrics/audio directly to **Weights & Biases (W&B)**.
2. **Promotion (PR to `prod`):** Merging a model triggers an automated CPU runner to compile the heavy `.pth` PyTorch weights into a lightning-fast static C++ `.onnx` graph.
3. **Production (`prod` branch):** The `.onnx` model is packaged into a custom **FastAPI Docker Container**, scanned for vulnerabilities, and pushed to Amazon ECR to serve live API traffic with zero disk I/O bottlenecks.

---

## 📂 Repository Structure

To separate heavy research dependencies from lightweight production code, the repository is split into two distinct environments:

* **`📁 .github/workflows/` (The Automation):** Contains the CI/CD pipelines for automated SageMaker training, ONNX conversion, ECR Docker builds, and DevSecOps scanning.
* **`📁 training/` (The Lab):** * `trigger_job.py`: The master switch to launch AWS SageMaker jobs remotely.
* `run.py` & `tune.py`: The decoupled PyTorch execution scripts (Standard vs. Optuna Sweeps).
* `models/`: The core neural network architectures and `configs.yaml` control panel.
* `utils/`: The infrastructure toolkit (`util_engine.py`, dynamic factories, and custom criterions).


* **`📁 inference/` (The Factory):** * `Dockerfile`: The lightweight production image (No PyTorch required).
* `app/serve.py`: The custom FastAPI web server designed for in-memory audio zipping.
* `weights/`: Where the compiled `.onnx` models live.



---

## 🚀 Execution & Automation Commands

This repository uses a "Master Switch" approach. You do not run the training loop directly; you use `trigger_job.py` to command AWS SageMaker to spin up the required GPU infrastructure.

### 1. Standard Training (AWS SageMaker)

Launch a full 200-epoch training run on an AWS `ml.g4dn.xlarge` instance. The script fires asynchronously (fire-and-forget) to save CI/CD costs.

```bash
python training/trigger_job.py \
    --job-name sepacap-standard-run \
    --config training/models/SepReformer_Base_WSJ0/configs.yaml \
    --mode train

```

### 2. Automated Hyperparameter Tuning (Optuna)

Launch a 30-trial Optuna sweep. SageMaker will dynamically invent learning rates and batch sizes, utilizing a `MedianPruner` to instantly kill underperforming runs and save GPU costs.

```bash
python training/trigger_job.py \
    --job-name sepacap-optuna-sweep \
    --config training/models/SepReformer_Base_WSJ0/configs.yaml \
    --mode tune

```

### 3. ONNX Model Promotion

Convert a trained `.pth` PyTorch research model into a production-ready C++ `.onnx` graph. (This runs automatically during Pull Requests to `prod`).

```bash
python training/promote_to_prod.py \
    --config training/models/SepReformer_Base_WSJ0/configs.yaml \
    --weights training/weights/epoch.best.pth \
    --output inference/weights/sepacap_model.onnx

```

---

## 🛡️ DevSecOps & Security

This repository enforces strict security standards to protect cloud infrastructure and model integrity:

* **Bandit (SAST):** Automatically scans Python code for ML-specific vulnerabilities (e.g., enforcing `weights_only=True` to prevent arbitrary code execution from poisoned `.pth` pickles).
* **GitLeaks:** Blocks commits containing hardcoded AWS Credentials or Weights & Biases API keys.
* **Trivy Container Scanning:** Scans the FastAPI production Docker image for OS-level CVEs before pushing to Amazon ECR.

---

## 🧠 Technical Highlights & Upgrades

SepACap heavily modifies the base SepReformer with the following deep learning and architectural upgrades:

* **SNAKE Activations:** Replaced standard ReLU/GELU in the separator blocks with periodic Snake activations to better extrapolate musical pitch and harmonics.
* **Two-Stage Detached PIT:** Optimizes the 6! (720) permutation calculations by detaching the gradient graph during the pairing phase, dropping VRAM usage drastically during loss calculation.
* **In-Memory RAM Processing:** The production FastAPI server utilizes `io.BytesIO` to load user `.wav` files, run ONNX inference, and compress the 6 output stems into a `.zip` archive entirely in RAM, preventing IOPS crashes under high web traffic.
* **Cloud Experiment Tracking:** Native integration with **Weights & Biases**, allowing for real-time loss tracking and in-browser audio playback (`wandb.Audio`) of the separated stems at different training epochs.

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