import os
import numpy as np
import librosa

def extract_mfcc(path, n_mfcc=13, sr=16000, duration=3):
    """
    Load an audio file and return its MFCC feature vector.

    Parameters
    ----------
    path : str
        Path to the audio file
    n_mfcc : int
        Number of MFCC coefficients to extract
    sr : int
        Sampling rate
    duration : float
        Maximum duration to load (seconds)

    Returns
    -------
    np.ndarray
        MFCC feature vector (shape: n_mfcc,)
    """
    audio, _ = librosa.load(path, sr=sr, duration=duration)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1)

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
