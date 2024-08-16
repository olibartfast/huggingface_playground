from datasets import load_dataset, load_from_disk
import soundfile as sf
import numpy as np
import os
from transformers import pipeline

# Prepare the dataset of audio recordings
# This dataset is a collection of different sounds of 5 seconds
dataset = load_dataset("ashraq/esc50", split="train[0:10]")
# dataset = load_from_disk("./models/ashraq/esc50/train")

audio_sample = dataset[0]

print("audio_sample: ", audio_sample)

# Save the audio sample as a .wav file for playback if desired
audio_file_path = "audio_sample.wav"
sf.write(audio_file_path, audio_sample["audio"]["array"], audio_sample["audio"]["sampling_rate"])

print(f"Audio sample saved to: {audio_file_path}")
print(f"Audio sample information: {audio_sample['audio']}")

# You can play the audio file using an external player or Python libraries like pydub, playsound, etc.

# Build the audio classification pipeline using 🤗 Transformers Library
zero_shot_classifier = pipeline(
    task="zero-shot-audio-classification",
    model="laion/clap-htsat-unfused"
)

# Set the correct sampling rate for the input and the model
from datasets import Audio
dataset = dataset.cast_column("audio", Audio(sampling_rate=48_000))
audio_sample = dataset[0]

# Define candidate labels
candidate_labels = ["Sound of a dog", "Sound of vacuum cleaner"]

# Perform zero-shot classification
result = zero_shot_classifier(audio_sample["audio"]["array"], candidate_labels=candidate_labels)
print("Classification result for first set of labels:", result)

# Additional set of candidate labels
candidate_labels = ["Sound of a child crying", "Sound of vacuum cleaner", "Sound of a bird singing", "Sound of an airplane"]

# Perform zero-shot classification again
result = zero_shot_classifier(audio_sample["audio"]["array"], candidate_labels=candidate_labels)
print("Classification result for second set of labels:", result)
