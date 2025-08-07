import os, time, io
from io import BytesIO

import streamlit as st
from audio_recorder_streamlit import audio_recorder

import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
import soundfile as sf
from Levenshtein import distance as levenshtein_distance

# --- PHONEME MODEL SETUP ---
@st.cache_resource
def load_phoneme_model():
    processor = Wav2Vec2Processor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
    model = Wav2Vec2ForCTC.from_pretrained("vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
    return processor, model

vowel_tolerance = {
    "a": {"a", "ɑ", "ə", "æ"},
    "e": {"e", "ɛ", "ə"},
    "i": {"i", "ɪ"},
    "o": {"o", "ɔ"},
    "u": {"u", "ʊ"}
}

def phoneme_score(audio_bytes, target_word="masyarakat"):
    processor, model = load_phoneme_model()
    # Load audio (wav) from bytes
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    # Model expects float32 numpy array
    input_values = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    phonemes = processor.batch_decode(predicted_ids)[0].split()

    # Target IPA for "masyarakat" (approximate)
    target_phonemes = "m a ʃ a ɾ a k a t".split()
    child_phonemes = phonemes

    # Apply tolerance mapping
    tolerant_child = []
    for ph in child_phonemes:
        if any(ph in vowel_tolerance.get(v, set()) for v in vowel_tolerance):
            for v, variants in vowel_tolerance.items():
                if ph in variants:
                    tolerant_child.append(v)
                    break
        else:
            tolerant_child.append(ph)

    # Levenshtein distance
    edit_distance = levenshtein_distance("".join(target_phonemes), "".join(tolerant_child))
    max_distance = max(len(target_phonemes), len(tolerant_child))
    score = ((max_distance - edit_distance) / max_distance) * 100 if max_distance > 0 else 0

    # Mismatches
    mismatches = []
    for i, (target, child) in enumerate(zip(target_phonemes, tolerant_child + [""] * (len(target_phonemes) - len(tolerant_child)))):
        if i >= len(tolerant_child) or (target != child and not (target in vowel_tolerance and child in vowel_tolerance.get(target, set()))):
            mismatches.append(f"Pos {i+1}: {target} → {child if i < len(tolerant_child) else 'missing'}")

    return {
        "phonemes": " ".join(phonemes),
        "score": score,
        "mismatches": mismatches,
        "target_phonemes": " ".join(target_phonemes),
        "tolerant_child": " ".join(tolerant_child)
    }

def split_audio_chunks(y, sr, chunk_duration=1.2):
    chunk_samples = int(chunk_duration * sr)
    total_samples = len(y)
    chunks = []
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunks.append(y[start:end])
    return chunks

# --- MAIN STREAMLIT APP ---
def main():
    st.title("🗣 ⇢ TalkSee ⇢ 👀")
    st.header("Phoneme Pronunciation Checker (Realtime Chunks)")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Select Input Mode")
        input_type = st.radio(
            'Input Mode',
            ('Mic', 'File'),
            horizontal=True
        )

    audio_data = None
    with col2:
        if input_type == 'Mic':
            st.write("Click below to start/stop mic recording:")
            audio_bytes = audio_recorder(text='', icon_size="2x")
            if audio_bytes is not None and len(audio_bytes) > 0:
                audio_data = BytesIO(audio_bytes)
        else:
            uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
            if uploaded_file is not None:
                audio_data = uploaded_file

    # Process as soon as audio is available (no button needed)
    if audio_data is not None:
        st.info("Note: Due to Streamlit and browser limitations, real-time chunk feedback is shown after recording/upload. Latest chunk appears at the top.")
        # Read bytes from audio_data
        if hasattr(audio_data, "read"):
            audio_bytes = audio_data.read()
            audio_data.seek(0)
        else:
            audio_bytes = audio_data.getvalue()
        # Load audio and split into chunks
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        chunks = split_audio_chunks(y, sr, chunk_duration=1.2)
        results = []
        for idx, chunk in enumerate(chunks):
            if len(chunk) == 0:
                continue
            # Convert chunk to bytes for phoneme_score using soundfile
            chunk_bytes = BytesIO()
            sf.write(chunk_bytes, chunk, sr, format='WAV')
            chunk_bytes.seek(0)
            chunk_bytes_data = chunk_bytes.read()
            result = phoneme_score(chunk_bytes_data)
            results.append((idx, result))
        # Display latest chunk at the top
        for idx, result in reversed(results):
            st.subheader(f"Chunk {idx+1} ({idx*1.2:.1f}-{(idx+1)*1.2:.1f}s)")
            st.markdown(f"**Target Phonemes:** `{result['target_phonemes']}`")
            st.markdown(f"**Detected Phonemes:** `{result['tolerant_child']}`")
            st.markdown(f"**Raw Model Output:** `{result['phonemes']}`")
            st.markdown(f"**Score:** `{result['score']:.1f}%`")
            if result["mismatches"]:
                st.error("Missed/Incorrect Phonemes:\n" + "\n".join(result["mismatches"]))
            else:
                st.success("All phonemes correct! 🎉")

# Run
if __name__ == "__main__":
    main()