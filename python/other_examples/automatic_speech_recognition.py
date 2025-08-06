from transformers.utils import logging
logging.set_verbosity_error()

from datasets import load_dataset

# Load the dataset
dataset = load_dataset(
    "librispeech_asr",
    split="train.clean.100",
    streaming=True,
    trust_remote_code=True
)

# Get an example from the dataset
example = next(iter(dataset))

# Get the first 5 samples from the dataset
dataset_head = dataset.take(5)
dataset_head_list = list(dataset_head)

# Access the third sample
third_sample = dataset_head_list[2]

# Print the example details
print("Example: ", example)

# sudo apt-get install portaudio19-dev
# pip install pydub pyaudio
# Play the audio using pydub and pyaudio
from pydub import AudioSegment
from pydub.playback import play

# Convert the audio array to an AudioSegment
audio_segment = AudioSegment(
    data=example["audio"]["array"].tobytes(),
    sample_width=2,  # assuming 16-bit audio
    frame_rate=example["audio"]["sampling_rate"],
    channels=1  # assuming mono audio
)

# Play the audio
play(audio_segment)


