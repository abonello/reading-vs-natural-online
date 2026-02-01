import os
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
# import librosa

# --- Using librosa for MFCC extraction ---
# def extract_mfcc(path, n_mfcc=13, sr=16000, duration=3):
#     """
#     Load an audio file and return its MFCC feature vector.

#     Parameters
#     ----------
#     path : str
#         Path to the audio file
#     n_mfcc : int
#         Number of MFCC coefficients to extract
#     sr : int
#         Sampling rate
#     duration : float
#         Maximum duration to load (seconds)

#     Returns
#     -------
#     np.ndarray
#         MFCC feature vector (shape: n_mfcc,)
#     """
#     audio, _ = librosa.load(path, sr=sr, duration=duration)
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
#     return mfcc.mean(axis=1)

# --- Using python_speech_features for MFCC extraction ---
def extract_mfcc(path, num_coeffs=13, duration=None):
    """
    Extract mean MFCCs from an audio file using python_speech_features.
    Returns a (num_coeffs,) vector.
    """
    # Read WAV file
    sr, audio = wav.read(path)  # audio is int16

    # Convert to float32 in [-1, 1]
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / 32768.0

    # Trim or pad to desired duration (in seconds)
    if duration is not None:
        max_len = int(sr * duration)
        if len(audio) > max_len:
            audio = audio[:max_len]
        elif len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)))

    # Compute MFCCs
    mfcc_feat = mfcc(audio, samplerate=sr, numcep=num_coeffs)
    
    # Take mean across time frames to get a fixed-size vector
    return np.mean(mfcc_feat, axis=0)

def load_dataset(base_dir="data", classes=None):
    """
    Load all audio files in the data folder and extract MFCCs.

    Parameters
    ----------
    base_dir : str
        Path to the data folder
    classes : list[str]
        Names of subfolders/classes (default: ["reading", "natural"])

    Returns
    -------
    X : np.ndarray
        Feature matrix (n_samples, n_mfcc)
    y : np.ndarray
        Labels (0,1,...)
    audio_paths : list[str]
        List of all file paths
    """
    if classes is None:
        classes = ["reading", "natural"]

    audio_paths = []
    labels = []

    for label, cls in enumerate(classes):
        folder = os.path.join(base_dir, cls)
        for fname in os.listdir(folder):
            if fname.endswith(".wav"):
                audio_paths.append(os.path.join(folder, fname))
                labels.append(label)

    features = [extract_mfcc(p) for p in audio_paths]

    X = np.array(features)
    y = np.array(labels)
    return X, y, audio_paths
