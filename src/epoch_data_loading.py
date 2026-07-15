import os
import mne

REQUIRED_EVENT_NAMES = {"rest", "left", "right"}


def load_subject_run_epochs(epoch_dir, subject, run_code):
    """
    Load ONE already-saved, already-epoched .fif file for a single
    subject/run from disk. This is the load-time counterpart to
    epoching.py's extract_epochs() (which creates and saves epochs) —
    this function only loads what's already on disk.

    Parameters
    ----------
    epoch_dir : str
        Root directory containing per-subject epoch folders, e.g.
        ".../data/processed/epochs"
    subject : str
        e.g. "S001"
    run_code : str
        e.g. "R04"

    Returns
    -------
    mne.Epochs

    Raises
    ------
    FileNotFoundError if the expected .fif file doesn't exist
    ValueError if the loaded epochs don't have exactly the expected
        event names (rest/left/right), or have zero epochs
    """
    fif_path = os.path.join(epoch_dir, subject, f"{subject}{run_code}_epo.fif")

    if not os.path.exists(fif_path):
        raise FileNotFoundError(fif_path)

    epochs = mne.read_epochs(fif_path, preload=True, verbose=False)

    if set(epochs.event_id.keys()) != REQUIRED_EVENT_NAMES:
        raise ValueError(
            f"{subject}{run_code}: unexpected event names "
            f"{set(epochs.event_id.keys())}, expected {REQUIRED_EVENT_NAMES}"
        )

    if len(epochs) == 0:
        raise ValueError(f"{subject}{run_code}: 0 epochs in file")

    return epochs


def load_subject_train_test_epochs(epoch_dir, subject, train_runs, test_run):
    """
    Load and assemble one subject's train (concatenated multiple runs)
    and test (single run) epochs from saved .fif files.

    Verifies all loaded runs for this subject share the same event_id
    mapping before concatenating — catches the case where, for some
    reason, one run's annotations produced different integer codes
    than another run's for the same subject (shouldn't normally happen
    within one subject, but worth checking rather than assuming).

    Parameters
    ----------
    epoch_dir : str
    subject : str
        e.g. "S001"
    train_runs : list[str]
        e.g. ["R04", "R08"]
    test_run : str
        e.g. "R12"

    Returns
    -------
    train_epochs, test_epochs : mne.Epochs, mne.Epochs
    event_id_map : dict
        e.g. {"rest": 1, "left": 2, "right": 3} — this subject's
        actual mapping, needed downstream for canonical label remapping
    """
    train_epoch_objs = [
        load_subject_run_epochs(epoch_dir, subject, run) for run in train_runs
    ]
    test_epochs = load_subject_run_epochs(epoch_dir, subject, test_run)

    all_event_ids = [ep.event_id for ep in train_epoch_objs] + [test_epochs.event_id]
    if not all(eid == all_event_ids[0] for eid in all_event_ids):
        raise ValueError(
            f"{subject}: event_id mapping differs across runs "
            f"({list(zip(train_runs + [test_run], all_event_ids))})"
        )

    train_epochs = mne.concatenate_epochs(train_epoch_objs)
    event_id_map = train_epochs.event_id

    return train_epochs, test_epochs, event_id_map
