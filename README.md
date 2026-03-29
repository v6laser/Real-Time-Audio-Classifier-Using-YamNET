# Real-Time Audio Classification on Raspberry Pi (YAMNet)

This project implements real-time audio classification using a USB microphone, TensorFlow Lite (YAMNet), and a Raspberry Pi 4.

## Features

- Real-time audio capture
- TensorFlow Lite inference
- Pretrained YAMNet model (521 classes)
- Rolling average smoothing
- Optimized for Raspberry Pi
- Multithreaded processing

## Project Structure


AudioClassifier/
├── main.py
├── yamnet.tflite
├── yamnet_class_map.csv
└── README.md


## Requirements

### Hardware
- Raspberry Pi 4
- USB microphone

### Software
- Python 3.10+

Install dependencies:


pip install numpy sounddevice tflite-runtime


## Usage

Clone the repository:


git clone https://github.com/your-username/audio-classifier.git

cd audio-classifier


Run:


python3 main.py


Example output:


Listening...
Speech (0.85)
Dog bark (0.67)


## How It Works

1. Captures audio from the microphone  
2. Queues audio data for processing  
3. Normalizes and resamples to 16 kHz  
4. Runs inference using YAMNet  
5. Applies rolling average smoothing  
6. Prints the top prediction  

## Model

- YAMNet (TensorFlow Lite)
- Input: ~15600 samples (~0.975 s at 16 kHz)
- Output: 521 audio classes

## Notes

- Uses lightweight resampling (`numpy.interp`)
- Drops frames if processing lags to avoid overflow
- Uses threading to separate audio capture and inference

## Troubleshooting

**Input overflow**
- Increase `CHUNK_DURATION`
- Reduce system load

**Invalid sample rate**
- Your microphone may not support 16 kHz
- The code uses the device’s default sample rate

**Tensor dtype errors**
- Ensure input is `float32` (handled in code)

## License

MIT
