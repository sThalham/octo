import sounddevice as sd
import numpy as np
import queue
import threading
import time
import speech_recognition as sr

# Initialize a queue to store audio data
audio_queue = queue.Queue()

# Audio settings
sample_rate = 16000  # Sampling rate
duration_after_trigger = 5  # Duration to record after detecting the word (in seconds)
trigger_word = "Mycobot"  # Word to listen for

# Function to capture audio continuously
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio error: {status}")
    audio_queue.put(indata.copy())

# Function to recognize the trigger word
def recognize_trigger():
    recognizer = sr.Recognizer()
    while True:
        try:
            # Get audio from the queue
            audio_data = audio_queue.get()
            # Convert to audio data for SpeechRecognition
            audio = sr.AudioData(audio_data.tobytes(), sample_rate, 2)
            # Perform speech recognition
            text = recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            if trigger_word in text.lower():
                print(f"Trigger word '{trigger_word}' detected!")
                return True
        except sr.UnknownValueError:
            continue  # Skip unrecognized audio
        except Exception as e:
            print(f"Recognition error: {e}")

# Function to record audio after the trigger word
def record_after_trigger():
    global recorded_data
    recorded_data = []  # To store the captured audio
    print("Recording...")
    start_time = time.time()
    while time.time() - start_time < duration_after_trigger:
        recorded_data.append(audio_queue.get())
    print("Recording finished.")
    # Combine recorded data into a single NumPy array
    recorded_data = np.concatenate(recorded_data, axis=0)

#mic = sr.Microphone()
#sr.Microphone.list_microphone_names()

audio = sd.rec(int(5 * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()
print("audio: ", audio)

# Start audio stream
stream = sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback)
with stream:
    print("Listening for the trigger word...")
    if recognize_trigger():
        record_after_trigger()

# The recorded audio data is now in `recorded_data`
print(f"Captured {len(recorded_data)} samples.")