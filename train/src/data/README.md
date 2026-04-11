# 📁 `/data`

**The Fuel for the Network**

### Purpose

This folder is the starting point of the pipeline. It handles all the audio processing, dataset preparation, and data management required before the neural network even sees the audio. Since audio files (especially uncompressed `.wav` files) are massive, this folder organizes how the network will efficiently load and mix them during training.

### Contents

* **`create_mixture_data/`**: Contains scripts to artificially mix isolated singing stems (Alto, Bass, Soprano, etc.) together. By adding different stems together at different volumes, it creates the "Acapella Mixtures" that the model will later try to unmix.
* **`create_scp_script`**: SCP (Script) files are essentially lightweight text files that act as pointers. Instead of loading thousands of heavy audio files into the computer's RAM at once, these scripts generate lists of file paths. The dataloader reads the SCP file and fetches the exact audio chunk it needs at that exact millisecond, saving massive amounts of memory.