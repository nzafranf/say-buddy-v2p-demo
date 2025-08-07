# Install dependencies
# pip install transformers torchaudio pyaudio numpy torch scipy python-Levenshtein
import torch
import pyaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import time
from Levenshtein import distance as levenshtein_distance
import librosa

# Check GPU/CPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() and input("Use GPU? (y/n): ").lower() == "y" else "cpu")
print(f"Using device: {device}")

# Load pre-trained phoneme model
processor = Wav2Vec2Processor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
model = Wav2Vec2ForCTC.from_pretrained("vitouphy/wav2vec2-xls-r-300m-timit-phoneme").to(device)

# Audio parameters
CHUNK_DURATION = 1.2  # 1 second chunks
RATE = 16000
CHUNK_SAMPLES = int(CHUNK_DURATION * RATE)
FORMAT = pyaudio.paFloat32
CHANNELS = 1

# Vowel tolerance mapping (based on articulatory similarity)
vowel_tolerance = {
    "a": {"a", "ɑ", "ə", "æ"},  # /a/ can be transcribed as /ɑ/, /ə/, or /æ/
    "e": {"e", "ɛ", "ə"},       # /e/ can be /ɛ/ or /ə/
    "i": {"i", "ɪ"},            # /i/ can be /ɪ/
    "o": {"o", "ɔ"},            # /o/ can be /ɔ/
    "u": {"u", "ʊ"}             # /u/ can be /ʊ/
}

#  m ə ʃ ɑ ɾ ə  k ɑ

# Combined inference and evaluation function with tolerance
def process_audio_chunk(chunk, start_time, target_word="masyarakat"):
    if len(chunk) == 0:
        return None
    # Preprocess and infer phonemes
    inputs = processor(chunk, sampling_rate=RATE, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    phonemes = processor.batch_decode(predicted_ids)[0].split()

    # Simplified pitch-based label
    pitch_mean = np.mean(librosa.piptrack(y=chunk, sr=RATE)[0][librosa.piptrack(y=chunk, sr=RATE)[0] > 0])
    label = "child" if pitch_mean > 300 else "parent" if pitch_mean > 0 and pitch_mean < 250 else None

    if label == "child":
        # Target IPA for "masyarakat" (approximate Indonesian phonology)
        # target_phonemes = "m a ʃ a ɾ a k a t".split()
        target_phonemes = "ʃ u k u ɾ".split()
        child_phonemes = phonemes

        # Apply tolerance mapping to child phonemes
        tolerant_child = []
        for ph in child_phonemes:
            if any(ph in vowel_tolerance.get(v, set()) for v in vowel_tolerance):
                # Find the closest matching vowel from tolerance sets
                tolerant_ph = ph
                for v, variants in vowel_tolerance.items():
                    if ph in variants:
                        tolerant_ph = v  # Map to the base vowel
                        break
                tolerant_child.append(tolerant_ph)
            else:
                tolerant_child.append(ph)

        # Calculate Levenshtein distance with tolerance
        print(f"Target Phonemes: {"".join(target_phonemes)}, Child Phonemes: {list("".join(tolerant_child))}")
        edit_distance = levenshtein_distance("".join(target_phonemes), list("".join(tolerant_child)))
        max_distance = max(len(target_phonemes), len("".join(tolerant_child)))
        score = ((max_distance - edit_distance) / max_distance) * 100 if max_distance > 0 else 0

        # Identify mismatches (only significant deviations)
        mismatches = []
        tolerant_child = list("".join(tolerant_child))
        for i, (target, child) in enumerate(zip(target_phonemes, tolerant_child + [""] * (len(target_phonemes) - len(tolerant_child)))):
            if i >= len(tolerant_child) or (target != child and not (target in vowel_tolerance and child in vowel_tolerance.get(target, set()))):
                mismatches.append(f"\nPosition {i+1} ({target} → {child if i < len(tolerant_child) else 'missing'})")

        return {
            "start": start_time, "end": start_time + CHUNK_DURATION,
            "phonemes": " ".join(phonemes), "label": label,
            "score": score, "mismatches": mismatches
        }
    return {"start": start_time, "end": start_time + CHUNK_DURATION, "phonemes": " ".join(phonemes), "label": label}

# Real-time audio capture and processing
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SAMPLES)

print("Recording... Say 'masyarakat'. Press Ctrl+C to stop.")
start_time = 0
transcription = []

try:
    while True:
        data = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
        audio_chunk = np.frombuffer(data, dtype=np.float32)
        
        # Process chunk
        result = process_audio_chunk(audio_chunk, start_time)
        if result and result["label"] == "child" and result.get("score") is not None:
            transcription_line = f"{result['label']} ({result['start']:.2f}-{result['end']:.2f}s): {result['phonemes']} | Score: {result['score']:.1f}%"
            if result["mismatches"]:
                transcription_line += f" | Issues: {', '.join(result['mismatches'])}"
            transcription.append(transcription_line)
            print(transcription_line)
            time.sleep(0.1)  # Simulate processing delay
        
        start_time += CHUNK_DURATION

except KeyboardInterrupt:
    print("Stopped recording.")
    final_transcription = "\n".join(transcription) if transcription else "No clear child speech detected."
    print("Final Labeled Transcription:\n", final_transcription)

    # Prepare data for Supabase
    supabase_data = {
        "age_group": "3-4 Tahun",  # Adjust based on user input
        "transcription": final_transcription,
        "timestamp": "2025-08-04 10:38"
    }
    print(f"Data untuk Supabase: {supabase_data}")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
