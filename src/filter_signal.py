import os
import mne

# This is your ONE established preprocessing decision (see
# docs/decisions.md): notch filter for powerline noise, then a
# 4-40 Hz bandpass covering mu (8-12Hz) and beta (12-30Hz) bands with
# margin on both sides. Use preprocess_raw() for all subjects/runs —
# do not mix this with a different bandpass range across subjects,
# since pooled cross-subject features must be processed identically.


def load_subject_run(data_dir, subject_id, run_number, preload=True):
    """
    Load a single EDF recording.
    """
    if subject_id < 1 or subject_id > 109:
        raise ValueError("subject_id must be between 1 and 109")
    if run_number < 1 or run_number > 14:
        raise ValueError("run_number must be between 1 and 14")

    subject = f"S{subject_id:03d}"
    file_name = f"{subject}R{run_number:02d}.edf"
    edf_path = os.path.join(data_dir, subject, file_name)

    if not os.path.exists(edf_path):
        raise FileNotFoundError(edf_path)

    raw = mne.io.read_raw_edf(
        edf_path, preload=preload, encoding="latin1", verbose=False
    )
    return raw


def preprocess_raw(raw, notch_freq=50, l_freq=4, h_freq=40):
    """
    The single, established preprocessing pipeline: notch filter for
    powerline noise, then 4-40 Hz bandpass.

    notch_freq: 50 Hz (India/most of world), 60 Hz (US/parts of Americas)
    l_freq/h_freq: 4-40 Hz bandpass — wide enough to retain mu (8-12Hz)
        and beta (12-30Hz) bands (the bands actually used downstream in
        feature extraction) while removing slow drift below 4Hz and
        muscle/electrical noise above 40Hz. See docs/decisions.md for
        the full reasoning.

    Always call this on the continuous raw signal BEFORE epoching —
    filtering after epoching introduces edge artifacts at trial
    boundaries.
    """
    processed = raw.copy()
    processed.notch_filter(notch_freq, verbose=False)
    processed.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    return processed
