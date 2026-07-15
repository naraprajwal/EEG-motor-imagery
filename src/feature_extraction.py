import numpy as np

from scipy.signal import welch

NPERSEG = 256


def extract_bandpower_features(epochs, nperseg=NPERSEG):
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]

    if data.shape[-1] < nperseg:
        raise ValueError(
            f"Epoch length ({data.shape[-1]} samples) shorter than "
            f"nperseg ({nperseg})."
        )

    features = []
    for epoch in data:
        epoch_features = []
        for channel in epoch:
            freqs, psd = welch(channel, fs=sfreq, nperseg=nperseg)
            mu_mask = (freqs >= 8) & (freqs <= 12)
            beta_mask = (freqs >= 13) & (freqs <= 30)
            mu_power = np.trapezoid(psd[mu_mask], freqs[mu_mask])
            beta_power = np.trapezoid(psd[beta_mask], freqs[beta_mask])
            epoch_features.extend([mu_power, beta_power])
        features.append(epoch_features)
    return np.array(features)
