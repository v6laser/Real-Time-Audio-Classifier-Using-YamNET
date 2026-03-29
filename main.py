import numpy as np
import sounddevice as sd
import tflite_runtime.interpreter as tflite
import librosa
import csv
from collections import deque

# -----------------------------
# File paths
# -----------------------------
MODEL_PATH = "yamnet.tflite"
CLASS_MAP_PATH = "yamnet_class_map.csv"

# Load class labels
class_names = []
with open(CLASS_MAP_PATH, newline='') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        class_names.append(row[2])

# Load TFLite model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Real-time parameters
TARGET_SR = 16000       # YAMNet expects 16 kHz
CHUNK_DURATION = 1.0    # seconds per chunk
ROLLING_SECONDS = 3     # rolling window for smoothing
CONF_THRESHOLD = 0.3    # minimum probability to print

# USB mic info
mic_device_id = 2  # From 'sd.query_devices()'
device_info = sd.query_devices(mic_device_id)
ACTUAL_SR = int(device_info['default_samplerate'])
CHUNK_SIZE = int(ACTUAL_SR * CHUNK_DURATION)

print(f"Using USB mic: {device_info['name']} at {ACTUAL_SR} Hz")

# Rolling buffer for scores
score_buffer = deque(maxlen=int(ROLLING_SECONDS / CHUNK_DURATION))

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
        return

    waveform = indata[:, 0] if indata.ndim > 1 else indata
    waveform = librosa.util.normalize(waveform).astype(np.float32)

    # Resample to 16 kHz for YAMNet
    if ACTUAL_SR != TARGET_SR:
        waveform = librosa.resample(waveform, orig_sr=ACTUAL_SR, target_sr=TARGET_SR)

    interpreter.resize_tensor_input(input_details[0]['index'], [len(waveform)])
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], waveform)
    interpreter.invoke()
    scores = interpreter.get_tensor(output_details[0]['index'])
    mean_scores = np.mean(scores, axis=0)

    # Add to rolling buffer
    score_buffer.append(mean_scores)
    smoothed_scores = np.mean(score_buffer, axis=0)

    top_index = np.argmax(smoothed_scores)
    top_score = smoothed_scores[top_index]

    if top_score >= CONF_THRESHOLD:
        print(f"Top Prediction: {class_names[top_index]} ({top_score:.2f})")

with sd.InputStream(device=mic_device_id, channels=1, samplerate=ACTUAL_SR,
                    blocksize=CHUNK_SIZE, callback=audio_callback):
    print("Listening...")
    while True:
        sd.sleep(1000)
