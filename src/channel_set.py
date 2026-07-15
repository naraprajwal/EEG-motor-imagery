# Channel set definitions used for feature extraction. These store
# CANONICAL channel names (no trailing dots, mixed case as in the
# standard 10-20 system) — resolve_channel_names() in
# channel_resolution.py handles matching these against whatever exact
# naming the loaded dataset uses (PhysioNet uses trailing dots, e.g.
# "C3..", confirmed during S001 exploration).

# None signals "use all channels, no .pick() call" — handled explicitly
# wherever this is consumed, rather than silently iterating over None.
FULL64_CHANNELS = None

C3CZC4_CHANNELS = [
    "C3",
    "Cz",
    "C4",
]

MOTOR_15_CHANNELS = [
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C3", "C1", "Cz", "C2", "C4",
    "CP3", "CP1", "CPz", "CP2", "CP4",
]

# Convenience registry so extraction scripts can select a channel set
# by name (e.g. for looping over all three at each subject-count
# checkpoint) instead of importing each constant separately.
CHANNEL_SETS = {
    "full64": FULL64_CHANNELS,
    "c3_cz_c4": C3CZC4_CHANNELS,
    "motor15": MOTOR_15_CHANNELS,
}
