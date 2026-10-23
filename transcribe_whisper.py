"""Whisper-small transcription for the spoken grammar scoring pipeline.

Run once on a GPU machine (e.g. a Kaggle notebook with GPU + Internet on):
    pip install openai-whisper soundfile pandas numpy
    python transcribe_whisper.py

Reads train/test audio from ROOT and writes transcripts.jsonl (one JSON record per clip).
The competition data and the generated transcripts are private and are not part of this repo.
"""
import json
import numpy as np
import pandas as pd
import soundfile as sf
import whisper

# Folder that contains train.csv, test.csv and the train/ and test/ audio folders
ROOT = "/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final"
OUT = "transcripts.jsonl"

# Disfluent prompt: keeps fillers, repeats and false starts, which carry grammar signal
PROMPT = "Umm, let me think, like, hmm... okay, I, I mean, so..."

train = pd.read_csv(f"{ROOT}/train.csv")
test = pd.read_csv(f"{ROOT}/test.csv")
asr = whisper.load_model("small", device="cuda")

# File names repeat across train and test, so records are keyed on (split, filename)
jobs = [("train", f) for f in train.filename] + [("test", f) for f in test.filename]

with open(OUT, "w") as fh:
    for split, name in jobs:
        path = f"{ROOT}/{split}/{name}"
        result = asr.transcribe(path, language="en", initial_prompt=PROMPT, fp16=True)
        segs = result["segments"]
        record = dict(
            filename=name,
            split=split,
            text=result["text"].strip(),
            duration=sf.info(path).duration,
            logprob=float(np.mean([s["avg_logprob"] for s in segs])) if segs else -5.0,
            no_speech=float(np.mean([s["no_speech_prob"] for s in segs])) if segs else 1.0,
        )
        fh.write(json.dumps(record) + "\n")
        fh.flush()

print(f"Wrote {len(jobs)} records to {OUT}")
