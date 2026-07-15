import os
import mne

REQUIRED_EVENT_NAMES = {"T0", "T1", "T2"}
CANONICAL_NAMES = {"T0": "rest", "T1": "left", "T2": "right"}
# NOTE: T1/T2 meaning depends on which run this is — for runs 5/6/9/10/13/14
# (bilateral fist/feet tasks) T1="both fists" and T2="both feet" instead.
# This mapping is only correct for unilateral runs (3,4,7,8,11,12).
# See docs/decisions.md run-protocol table.


def extract_epochs(
    data_dir,
    subject_ids,
    run_number,
    tmin=0.0,
    tmax=4.0,
    preload=True,
):
    """
    Extract epochs for a list of subjects, for a single run number.

    Labels are renamed to canonical names (rest/left/right) at
    extraction time, and any subject/run missing one of the three
    required event types is skipped with a clear message rather than
    silently producing a malformed epochs object or crashing the
    whole batch.

    Parameters
    ----------
    data_dir : str
        Root EEG dataset directory
    subject_ids : list[int]
        Example: [1, 2, 3, 4]
    run_number : int
        Example: 3, 4, 5, 6 (1-14; see run-protocol mapping in docs)
    tmin, tmax : float
        Epoch window relative to event onset. Default 0.0-4.0 matches
        the ~4s trial duration confirmed from this dataset's .event
        files — change deliberately, not casually.

    Returns
    -------
    epochs_dict : dict[str, mne.Epochs]
        {"S001": Epochs, "S002": Epochs, ...} — only for subjects that
        loaded successfully with all required event types present.
    failed : list[tuple[str, str]]
        (subject_id, reason) for any subject that failed or was skipped.
    """
    if run_number < 1 or run_number > 14:
        raise ValueError("run_number must be between 1 and 14")

    for s in subject_ids:
        if s < 1 or s > 109:
            raise ValueError(f"Invalid subject {s}. Valid range is 1-109.")

    is_bilateral_run = run_number in (5, 6, 9, 10, 13, 14)
    if is_bilateral_run:
        print(
            f"  NOTE: run {run_number} is a bilateral fist/feet run — "
            f"T1='both fists', T2='both feet', not left/right."
        )

    epochs_dict = {}
    failed = []

    for subject in subject_ids:
        subject_id = f"S{subject:03d}"
        file_name = f"{subject_id}R{run_number:02d}.edf"
        edf_path = os.path.join(data_dir, subject_id, file_name)

        try:
            if not os.path.exists(edf_path):
                raise FileNotFoundError(edf_path)

            print(f"Loading {file_name}")
            raw = mne.io.read_raw_edf(
                edf_path, preload=preload, encoding="latin1", verbose=False
            )

            events, event_id_raw = mne.events_from_annotations(raw, verbose=False)

            if not REQUIRED_EVENT_NAMES.issubset(event_id_raw.keys()):
                raise ValueError(
                    f"missing event codes (found {sorted(event_id_raw.keys())}, "
                    f"need {sorted(REQUIRED_EVENT_NAMES)})"
                )

            if is_bilateral_run:
                event_id = {
                    "rest": event_id_raw["T0"],
                    "both_fists": event_id_raw["T1"],
                    "both_feet": event_id_raw["T2"],
                }
            else:
                event_id = {
                    "rest": event_id_raw["T0"],
                    "left": event_id_raw["T1"],
                    "right": event_id_raw["T2"],
                }

            epochs = mne.Epochs(
                raw,
                events,
                event_id=event_id,
                tmin=tmin,
                tmax=tmax,
                baseline=None,
                preload=True,
                verbose=False,
            )

            if len(epochs) == 0:
                raise ValueError("0 epochs extracted")

            epochs_dict[subject_id] = epochs
            print(f"  {subject_id}: {len(epochs)} epochs, event_id={event_id}")

        except Exception as e:
            print(f"  FAILED {subject_id}: {e}")
            failed.append((subject_id, str(e)))
            continue

    return epochs_dict, failed
