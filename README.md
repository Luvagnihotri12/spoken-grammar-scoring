# Spoken Grammar Scoring Engine

Predicts a **0-5 grammar score** from 45-60 second speech clips.
Built for the SHL Research Intern Kaggle challenge (private competition, so the data is **not** included in this repo).

**Result:** public leaderboard RMSE **0.3972** (computed on ~60% of the test set; final standing uses the remaining 40%).
5-fold CV on the training set: RMSE ~0.49 and Pearson ~0.87 (excluding the 37 label-0 files, see below).

## Approach

```
audio ──► Whisper-small ASR ──► text features ─────────────┐
  │                                                         ├─► Ridge + SVR per feature set ─► non-negative stacker ─► clip [0, 5]
  └────► frozen speech encoders ─► pooled audio features ───┘
```

1. **ASR.** Whisper `small` transcribes every clip. A disfluent initial prompt keeps fillers, repeats and false starts, which carry grammar signal.
2. **Text features** (`text_blend`):
   - hand-made features: word count, words per second, sentence count and length, type-token ratio, repetition rate, filler rate, Whisper log-prob, no-speech probability
   - GPT-2 loss as a fluency proxy
   - `all-mpnet-base-v2` sentence embeddings
   - models: Ridge and SVR, averaged
3. **Audio features.** Hidden states from frozen encoders, pooled over time, for several layers:
   - HuBERT-base
   - WavLM-large (mean pooling, and mean+std pooling over 7 layers)
   - Whisper-medium encoder (mean pooling, and mean+std pooling over 5 layers, windowed in 30 s chunks)
4. **Per-feature-set models.** Each embedding is standardised and concatenated with the hand and GPT-2 features. A Ridge (alpha chosen by CV) and an SVR (C=3, epsilon=0.1) are averaged.
5. **Stacking.** A non-negative Ridge combines the out-of-fold predictions of all feature sets plus the text blend. The final prediction is clipped to [0, 5].

## Validation

- 5-fold CV with out-of-fold predictions throughout. The stacker's CV estimate is averaged over 3 fold seeds.
- Metrics are reported **with and without the 37 label-0 files.** Those are garbled or silent recordings that appear only in the training set (the test set has none), so excluding them is closer to the test conditions.
- **Train RMSE** is printed in the notebook. It is much lower than CV RMSE because the stack fits the training data tightly. **CV RMSE is the honest estimate**, and it agrees with the leaderboard.

## Findings

- **Audio beats text.** Whisper partly corrects grammar mistakes while transcribing, so those errors are invisible in the transcript but still present in the audio. Text-only CV Pearson was ~0.64 (excluding zeros) vs. ~0.86 for the best audio model.
- **Score progression (public RMSE):** text-only Ridge 0.5911, plus embeddings and GPT-2 0.5144, plus WavLM-base audio 0.4228, plus more encoders 0.4172, plus WavLM-large, Whisper-medium and stacking 0.3983, plus mean+std pooling 0.3972.
- **Tried, no gain (<0.003 RMSE):** Whisper-large-v3 encoder, SVR epsilon tuning, pause and speech-rate features, down-weighting the label-0 files.

## Limitations

- Alpha, SVR settings and stacker weights were tuned on the same CV folds as the reported CV, so CV is mildly optimistic.
- Only 769 training clips, so heavier models (fine-tuning) are likely to overfit.
- The Whisper ASR output is itself a lossy view of the speech.

## Reproduce

1. Run on Kaggle with a GPU and Internet enabled. Attach the competition data.
2. Run `transcribe_whisper.py` once to produce `transcripts.jsonl` (Whisper-small; about an hour for 985 clips).
3. Run the notebook `grammar_scoring.ipynb` top to bottom. The audio-embedding cells need a GPU (WavLM-large and Whisper-medium take the longest, roughly 15-25 minutes each).
4. The notebook writes `submission.csv` with columns `filename,label`.

## Repo contents

| File | Purpose |
|---|---|
| `grammar_scoring.ipynb` | full pipeline, metrics, train RMSE and plots |
| `transcribe_whisper.py` | Whisper-small transcription script |
| `README.md` | this file |

The audio, CSVs, transcripts and saved embeddings are **not** included because the competition data is private.

## Author

Luv Agnihotri
