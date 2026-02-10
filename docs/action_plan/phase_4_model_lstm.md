# What is LSTM?

LSTM = Long Short-Term MemoryA type of neural network designed for sequences (where order matters).

ANALOGY: The Memory DetectiveLinear Regression (what we just did):

Detective with a checklist:
"Yesterday's price? Check. Moving average? Check."
Makes decision based on fixed list of clues.

LSTM:

Detective with actual memory:
"I remember the pattern from 60 days ago...
The stock rose for 10 days, then crashed.
Wait, I'm seeing that same pattern now!
I'll remember this for future cases."Key difference: LSTM can remember patterns from the past, not just fixed features.

------------------------------------------------------------------------------------
### Output

python3 src/models/02_lstm_model.py
2026-02-10 08:46:40.687538: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-02-10 08:46:40.760372: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2026-02-10 08:46:42.210880: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
============================================================
LSTM MODEL - AAPL
============================================================
Loaded: (1194, 40)

============================================================
PREPARING DATA FOR LSTM
============================================================
Using Close price only
Shape: (1194, 1)

Original price range: $119.89 to $286.19
Scaled range: 0.0000 to 1.0000

Sequence length: 60 days
Created 1134 sequences
X shape: (1134, 60, 1)  (samples, timesteps, features)
y shape: (1134, 1)  (samples, 1)

Train: 907 sequences
Test:  227 sequences

============================================================
BUILDING LSTM MODEL
============================================================
2026-02-10 08:46:42.594513: E external/local_xla/xla/stream_executor/cuda/cuda_platform.cc:51] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
/home/sebastian/ai-projects/sentiment-analyzer/venv/lib/python3.12/site-packages/keras/src/layers/rnn/rnn.py:199: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)

Model Architecture:
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ lstm (LSTM)                          │ (None, 60, 50)              │          10,400 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 60, 50)              │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_1 (LSTM)                        │ (None, 50)                  │          20,200 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 50)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 25)                  │           1,275 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 1)                   │              26 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 31,901 (124.61 KB)
 Trainable params: 31,901 (124.61 KB)
 Non-trainable params: 0 (0.00 B)

============================================================
TRAINING LSTM
============================================================
Training... (this may take a few minutes)
Epochs: 50 (or until early stopping)
Epoch 1/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 5s 78ms/step - loss: 0.0198 - val_loss: 0.0041
Epoch 2/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 2s 62ms/step - loss: 0.0043 - val_loss: 0.0043
Epoch 3/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 2s 66ms/step - loss: 0.0035 - val_loss: 0.0052
Epoch 4/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 2s 63ms/step - loss: 0.0028 - val_loss: 0.0051
Epoch 5/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 2s 65ms/step - loss: 0.0028 - val_loss: 0.0034
Epoch 6/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 2s 63ms/step - loss: 0.0024 - val_loss: 0.0030
Epoch 7/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 2s 63ms/step - loss: 0.0023 - val_loss: 0.0028
Epoch 8/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 2s 66ms/step - loss: 0.0022 - val_loss: 0.0027
Epoch 9/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 2s 64ms/step - loss: 0.0021 - val_loss: 0.0035
Epoch 10/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 2s 64ms/step - loss: 0.0020 - val_loss: 0.0026
Epoch 11/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 2s 63ms/step - loss: 0.0019 - val_loss: 0.0025
Epoch 12/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 2s 63ms/step - loss: 0.0017 - val_loss: 0.0029
Epoch 13/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 2s 66ms/step - loss: 0.0018 - val_loss: 0.0029
Epoch 14/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 2s 55ms/step - loss: 0.0015 - val_loss: 0.0024
Epoch 15/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 28ms/step - loss: 0.0016 - val_loss: 0.0024
Epoch 16/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 29ms/step - loss: 0.0016 - val_loss: 0.0037
Epoch 17/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 32ms/step - loss: 0.0017 - val_loss: 0.0026
Epoch 18/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 26ms/step - loss: 0.0015 - val_loss: 0.0023
Epoch 19/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 26ms/step - loss: 0.0015 - val_loss: 0.0023
Epoch 20/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 27ms/step - loss: 0.0014 - val_loss: 0.0033
Epoch 21/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 27ms/step - loss: 0.0014 - val_loss: 0.0024
Epoch 22/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 30ms/step - loss: 0.0013 - val_loss: 0.0028
Epoch 23/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 35ms/step - loss: 0.0014 - val_loss: 0.0021
Epoch 24/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 31ms/step - loss: 0.0012 - val_loss: 0.0021
Epoch 25/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 33ms/step - loss: 0.0013 - val_loss: 0.0026
Epoch 26/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 29ms/step - loss: 0.0013 - val_loss: 0.0020
Epoch 27/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 32ms/step - loss: 0.0011 - val_loss: 0.0021
Epoch 28/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 32ms/step - loss: 0.0013 - val_loss: 0.0020
Epoch 29/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 29ms/step - loss: 0.0011 - val_loss: 0.0021
Epoch 30/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 33ms/step - loss: 0.0011 - val_loss: 0.0020
Epoch 31/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 32ms/step - loss: 0.0011 - val_loss: 0.0019
Epoch 32/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 29ms/step - loss: 0.0011 - val_loss: 0.0022
Epoch 33/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 31ms/step - loss: 0.0011 - val_loss: 0.0018
Epoch 34/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 30ms/step - loss: 0.0012 - val_loss: 0.0018
Epoch 35/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 33ms/step - loss: 0.0010 - val_loss: 0.0019
Epoch 36/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 32ms/step - loss: 0.0011 - val_loss: 0.0018
Epoch 37/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 31ms/step - loss: 0.0010 - val_loss: 0.0019
Epoch 38/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 29ms/step - loss: 9.3726e-04 - val_loss: 0.0018
Epoch 39/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 33ms/step - loss: 9.2149e-04 - val_loss: 0.0018
Epoch 40/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 29ms/step - loss: 9.3694e-04 - val_loss: 0.0021
Epoch 41/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 31ms/step - loss: 0.0010 - val_loss: 0.0021
Epoch 42/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 32ms/step - loss: 9.7396e-04 - val_loss: 0.0018
Epoch 43/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 35ms/step - loss: 9.0204e-04 - val_loss: 0.0019
Epoch 44/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 32ms/step - loss: 9.0885e-04 - val_loss: 0.0018
Epoch 45/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 34ms/step - loss: 9.5285e-04 - val_loss: 0.0022
Epoch 46/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 27ms/step - loss: 9.9408e-04 - val_loss: 0.0016
Epoch 47/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 33ms/step - loss: 8.6431e-04 - val_loss: 0.0016
Epoch 48/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 33ms/step - loss: 9.7599e-04 - val_loss: 0.0015
Epoch 49/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 31ms/step - loss: 8.7256e-04 - val_loss: 0.0015
Epoch 50/50
29/29 ━━━━━━━━━━━━━━━━━━━━ 1s 31ms/step - loss: 8.2225e-04 - val_loss: 0.0016

✓ Training complete!

============================================================
MAKING PREDICTIONS
============================================================
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
Predictions shape: (227, 1)

============================================================
EVALUATION
============================================================

LSTM Results:
  MAE:  $4.35
  RMSE: $6.36
  MAPE: 1.96%
  R²:   0.9524

Comparison to Naive Baseline:
  Naive RMSE:  $4.31
  LSTM RMSE:   $6.36
  ⚠️  Naive still wins
  This is expected - LSTM needs MUCH more data to shine

============================================================
CREATING VISUALIZATIONS
============================================================
✓ Saved: results/plots/lstm_model.png

============================================================
SAVING MODEL
============================================================
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
✓ Saved: models/lstm_stock_predictor.h5
✓ Saved: models/lstm_scaler.pkl

✓ Phase 4 complete! LSTM model trained.

------------------------------------------------------------------------------------------

# Analysis
Results Summary
ModelMAERMSEMAPER²Naive (Tomorrow=Today)$2.81$4.311.25%0.9774Linear Regression$2.84$4.381.26%0.9765LSTM$4.35$6.361.96%0.9524
LSTM is actually WORSE than both baselines!

Why LSTM Failed
1. Not Enough Data
Training sequences: 907
LSTM parameters: 31,901

That's 35 parameters PER training sample!
Rule of thumb: Need 10-100x more data than parameters

We have: 907 samples
We need: 100,000+ samples for this model

ANALOGY:
Teaching someone chess by showing them 900 games when they need to memorize 31,901 different patterns. They'll memorize the training games but fail on new ones (overfitting).
2. Overfitting
Training loss kept dropping (good on training data):
Epoch 1:  loss: 0.0198
Epoch 50: loss: 0.0008  ← Got way better!
But validation loss plateaued:
Epoch 1:  val_loss: 0.0041
Epoch 50: val_loss: 0.0016  ← Barely improved after epoch 10
This means: Model learned training data TOO well, doesn't generalize.
3. Stock Prices Are Too Predictable
The problem:

99% of tomorrow's price is explained by today's price
LSTM tries to learn the remaining 1% (random noise)
Can't learn noise!

ANALOGY:
Naive: "Temperature tomorrow ≈ today" (99% accurate)
LSTM: "Let me analyze 60 days of patterns to predict tomorrow"
       (Gets distracted by noise, performs worse)

When LSTM Actually Works
LSTM shines when:
✅ Long-term dependencies matter

Language: "The cat [20 words later] jumped" (need to remember "cat")
Weather: Seasonal patterns spanning months

✅ Complex patterns exist

Speech recognition
Music generation
Video prediction

✅ LOTS of data available

Millions of sequences
Can actually learn patterns vs memorizing

❌ Stock prices fail all three:

Tomorrow mostly = today (no long-term dependency needed)
Mostly random walk (no learnable patterns)
Only 1,194 days of data (way too little)


What We Learned
About Stock Prediction:
✅ Simple baselines are incredibly hard to beat
✅ More complex ≠ better (LSTM was worse!)
✅ Stock prices have too much noise vs signal
✅ "Tomorrow = today" captures 97%+ of information
About LSTM:
✅ How to prepare sequences for LSTM
✅ Scaling is critical for neural networks
✅ Architecture: stacked LSTM layers
✅ Early stopping prevents overfitting
✅ When LSTM works (and when it doesn't)
About Time Series:
✅ Proper train/test split (chronological, no shuffle)
✅ Naive baseline is the first thing to try
✅ High R² doesn't mean good predictions
✅ Domain knowledge matters (understand what you're predicting)

Final Model Comparison
ModelComplexityRMSETraining TimeWinner?NaiveSimplest$4.310 seconds✓Linear RegMedium$4.381 secondLSTMMost Complex$6.3660 seconds
The simplest model won!

The Honest Conclusion
For stock price prediction:

Don't use LSTM (waste of time)
Simple baseline is best
Markets are mostly efficient
Can't predict the unpredictable

But we still learned valuable skills:

Time series preprocessing
LSTM architecture
Proper evaluation
When NOT to use deep learning


🎓 Project Complete!
What You Built:

✅ Downloaded 5 years of stock data
✅ Engineered 40 features (lags, MAs, RSI, MACD, etc.)
✅ Trained 3 models (Naive, Linear Regression, LSTM)
✅ Learned why stock prediction is hard
✅ Proper time series methodology
