# Audio Sound Classifier (YAMNet)

This project is a simple audio classification tool built using Google's pretrained YAMNet model from TensorFlow Hub.

It analyzes WAV audio files and prints the top predicted sound class with its confidence score.

No training or dataset collection is required.

---

## Features

- Uses pretrained YAMNet model (trained on 500+ sound classes)
- Automatically detects the first `.wav` file inside a folder
- Removes silence before analysis
- Prints only the top prediction and confidence score
- No manual training required

---

## Project Structure

```
AudioClassifier/
│
├── main.py
├── README.md
└── input_audio/
      └── your_audio_file.wav
```

---

## Requirements

- Python 3.10 or 3.11
- TensorFlow
- TensorFlow Hub
- Librosa
- NumPy

Example output:

```
Processing file: dog.wav
Analyzing audio...

Top Prediction:
Dog (0.83)
