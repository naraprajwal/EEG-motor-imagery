import os
import numpy as np

CANONICAL_EVENT_ID = {"rest": 1, "left": 2, "right": 3}


def load_subject_data(feature_dir, subject):
    """
    Load one subject's saved train/test features and labels.

    Labels are remapped to CANONICAL_EVENT_ID using the subject's own
    saved event_id_map.npy, so integer label codes are guaranteed
    consistent across subjects even if MNE assigned different raw
    codes per subject's file (this WILL happen across some subjects —
    confirmed necessary during 10-subject pooling).

    Also returns a subject_id array tagging every row, needed for
    subject-level train/test splitting once multiple subjects are
    pooled together.

    Returns
    -------
    X_train, y_train, X_test, y_test, subject_ids_train, subject_ids_test
    """
    save_dir = os.path.join(feature_dir, subject)

    X_train = np.load(os.path.join(save_dir, "X_train.npy"))
    y_train_raw = np.load(os.path.join(save_dir, "y_train.npy"))
    X_test = np.load(os.path.join(save_dir, "X_test.npy"))
    y_test_raw = np.load(os.path.join(save_dir, "y_test.npy"))

    event_id_path = os.path.join(save_dir, "event_id_map.npy")
    if not os.path.exists(event_id_path):
        raise FileNotFoundError(
            f"{event_id_path} not found — was this subject's features "
            f"extracted with event_id_map saving enabled?"
        )
    event_id_map = np.load(event_id_path, allow_pickle=True).item()

    if set(event_id_map.keys()) != set(CANONICAL_EVENT_ID.keys()):
        raise ValueError(
            f"{subject}: event_id_map keys {set(event_id_map.keys())} "
            f"don't match expected {set(CANONICAL_EVENT_ID.keys())}"
        )

    code_to_name = {v: k for k, v in event_id_map.items()}
    remap = {
        orig_code: CANONICAL_EVENT_ID[name]
        for orig_code, name in code_to_name.items()
    }

    y_train = np.array([remap[c] for c in y_train_raw])
    y_test = np.array([remap[c] for c in y_test_raw])

    subject_ids_train = np.full(len(y_train), subject)
    subject_ids_test = np.full(len(y_test), subject)

    return X_train, y_train, X_test, y_test, subject_ids_train, subject_ids_test


def load_and_pool_subjects(feature_dir, subjects):
    """
    Load multiple subjects and pool them, tracking subject_id for
    every row across both train and test sets combined (the split
    point — within-subject vs cross-subject — is decided downstream,
    not here).

    Returns
    -------
    X_all, y_all, subject_ids_all : pooled arrays
    skipped : list of subjects that failed to load, with reasons
    """
    X_list, y_list, subj_list = [], [], []
    skipped = []

    for subject in subjects:
        try:
            X_tr, y_tr, X_te, y_te, sid_tr, sid_te = load_subject_data(
                feature_dir, subject
            )
            X_list.append(np.concatenate([X_tr, X_te], axis=0))
            y_list.append(np.concatenate([y_tr, y_te], axis=0))
            subj_list.append(np.concatenate([sid_tr, sid_te], axis=0))
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping {subject}: {e}")
            skipped.append((subject, str(e)))

    if not X_list:
        raise RuntimeError("No subjects loaded successfully.")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    subject_ids_all = np.concatenate(subj_list, axis=0)

    return X_all, y_all, subject_ids_all, skipped
