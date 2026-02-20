# 📁 `/models`

**The Brains of the Operation**

### Purpose

This is the core of the repository. It houses the actual neural network architectures, the training instructions, and the saved "memories" (weights) of the models after they have learned how to separate audio.

### Contents

* **`SepReformer_Base_WSJ0/`** *(and other variations)*: Each subfolder represents a specific version or size of the model. Keeping them separate allows you to experiment with different architectures without breaking your main code.
* **`modules/`**: This is where the actual PyTorch code for the neural network lives.
* `module.py`: Contains the major building blocks of the network (e.g., the `AudioEncoder` that reads the audio, the `Separator` that splits the frequencies, and the `AudioDecoder` that turns it back into sound).
* `network.py`: Contains the lower-level math operations, like the multi-head attention blocks, Convolutional Neural Network (CNN) layers, and the SNAKE activation functions.


* **`log/scratch_weights/`**: When the model trains, it periodically saves its progress as `.pth` (checkpoint) files here. If the power goes out, or if you want to use the model to separate a new song, it loads the "brain" from this folder.
* **`configs.yaml`**: The master control panel. This simple text file dictates the batch size, learning rate, audio length, and structural sizes (like 6 speakers instead of 2) for that specific model.
* **`engine.py` & `main.py**`: The drivers that read the configuration, build the model, and trigger the training or testing processes.

