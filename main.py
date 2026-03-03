import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
import warnings

warnings.filterwarnings("ignore")

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import csv


tf.get_logger().setLevel(logging.ERROR)


model = hub.load("https://tfhub.dev/google/yamnet/1")


class_map_path = model.class_map_path().numpy()

class_names = []
with open(class_map_path, newline='') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        class_names.append(row[2])


input_folder = "input_audio"
audio_files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]

if not audio_files:
    print("No WAV files found.")
    exit()

audio_file = os.path.join(input_folder, audio_files[0])


waveform, sr = librosa.load(audio_file, sr=16000)
waveform, _ = librosa.effects.trim(waveform)

scores, embeddings, spectrogram = model(waveform)

mean_scores = tf.reduce_mean(scores, axis=0).numpy()
top_index = np.argmax(mean_scores)

print(f"{class_names[top_index]} ({mean_scores[top_index]:.2f})")
