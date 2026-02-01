# Reading vs Natural Speech Classifier

This project is a small end-to-end audio classification system that distinguishes
**reading-style speech** from **natural conversational speech** using short audio clips.

The goal is not to build a production-grade model, but to demonstrate:
- Audio feature extraction (MFCCs)
- Basic neural network implementation from scratch
- Model training and evaluation
- A simple web interface with Flask
- Clean separation between data, features, model, and interface

This project was developed as part of a technical portfolio.

---

## Overview

The pipeline is:

1. Load short mono WAV files (16kHz, 16-bit)
2. Extract MFCC features (13 coefficients per clip)
3. Normalize features using training statistics
4. Train a small neural network (1 hidden layer, ReLU + Sigmoid)
5. Evaluate classification accuracy
6. Save model weights and feature scaler
7. Serve the model via a Flask web interface for online predictions

---

## Dataset

The dataset consists of **23 short audio clips**:

- 12 reading-style recordings
- 11 natural speech recordings

> **Note:** Due to the small dataset size, overfitting is expected.  
> Accuracy may vary across runs, but the focus is on demonstrating a complete ML workflow.

---

## Project Structure

reading-vs-natural-online/
├── app.py # Flask application
├── requirements.txt # Python dependencies
├── README.md
├── src/
│ ├── init.py
│ ├── features.py # MFCC extraction
│ ├── model.py # Tiny neural network implementation
│ ├── train.py # Training script
│ └── predict.py # Inference helper
├── data/
│ ├── reading/ # Reading-style audio clips
│ └── natural/ # Natural speech clips
├── uploads/ # Uploaded audio files via Flask
└── models/
├── model.npz # Saved NN weights
└── scaler.npz # Saved StandardScaler statistics




---

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/reading-vs-natural-online.git
   cd reading-vs-natural-online
   ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # Mac/Linux
    .venv\Scripts\activate     # Windows
    ```
3. Install dependencies:
    ```bash
    python3 -m pip install -r requirements.txt
    ```

## Training the Model

1. Place your audio clips in data/reading and data/natural.

2. Run the training script:
    ```bash
    python3 -m src.train
    ```

3. This will:
    * Extract MFCC features
    * Train the TinyNN model
    * Save model weights (models/model.npz)
    * Save scaler statistics (models/scaler.npz)

## Using the Web Interface
1. Start the Flask app:
    ```bash
    python3 app.py
    ```
2. Open your browser at http://127.0.0.1:5000
3. Upload an audio file (mono WAV, 16kHz recommended)
4. Click Upload and Predict to see the classification
5. The uploaded file name and prediction will remain displayed until a new file is uploaded.
   > Tip: Only short clips were used during training. Very long files may yield unreliable predictions.

## Notes on Audio Files
* Sampling rate: 16 kHz
* Bit depth: 16-bit PCM
* Mono audio

For best results, match these settings when recording or preprocessing your audio.  
I used these settings to keep this prototype at a manageable size.

## Development
* `src/features.py`: MFCC extraction logic
* `src/model.py`: Tiny neural network with forward, backward, training, and predict functions
* `src/train.py`: Training pipeline including model/scaler saving
* `src/predict.py`: Loading saved model and scaler, making predictions
* `app.py`: Flask app exposing a web interface