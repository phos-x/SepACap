# 📁 `/utils`

**The Engineering Toolbox**

### Purpose

This folder contains the "helper" scripts. These files don't define the neural network itself, but they provide the essential infrastructure needed to train it, measure its success, and prevent it from crashing.

### Contents

* **`util_engine.py`**: The "foreman" of the factory. It dictates the actual training loop (Epoch 1, Epoch 2...), calculates the hardware memory usage (MACs/Params), and handles the logic for safely saving and loading checkpoints without causing size-mismatch crashes.
* **`util_implement.py`**: A dynamic "factory" script. Instead of hard-coding which optimizer or loss function to use, this script reads your `configs.yaml` and dynamically builds the right PyTorch tools on the fly. It includes "Type Shields" to prevent code crashes when switching between 2-speaker and 6-speaker setups.
* **`implements/criterions.py`**: The "grader." This contains the Loss Functions. It includes the math (like the *Two-Stage Detached PIT* and *Multi-Resolution Spectral Loss*) that compares the model's separated audio against the true audio, scores it, and tells the network how to improve.
* **`decorators.py`**: Contains Python "wrappers" (like `@logger_wraps`). These attach themselves to other functions to automatically print out beautiful, color-coded logs to your terminal so you can track exactly what the code is doing in real-time.