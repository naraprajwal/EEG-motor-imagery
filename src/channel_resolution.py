def resolve_channel_names(available_chs, wanted):
    """
    Match each wanted channel name (e.g. "C3", "Cz") against the actual
    channel names in a loaded dataset, which may carry trailing dots
    and inconsistent casing (PhysioNet EDF exports are inconsistent
    about both — confirmed "C3.." form during S001 exploration).

    Tries case-insensitive match ignoring trailing dots. Raises loudly
    if a channel can't be resolved, and raises if two different wanted
    names resolve to the same actual channel (would otherwise break
    epochs.pick() with a confusing error — this exact bug broke an
    earlier version of the 15-channel extraction script, where C3/Cz/C4
    were each listed twice).

    Parameters
    ----------
    available_chs : list[str]
        epochs.info["ch_names"] from a loaded MNE object
    wanted : list[str]
        Canonical channel names to resolve, e.g. MOTOR_15_CHANNELS

    Returns
    -------
    list[str] : actual channel names as they appear in available_chs,
        in the same order as `wanted`
    """
    lookup = {ch.lower().rstrip("."): ch for ch in available_chs}
    resolved = []
    seen_actual = set()

    for ch in wanted:
        key = ch.lower().rstrip(".")
        if key not in lookup:
            raise ValueError(
                f"Channel '{ch}' not found in dataset channels: {available_chs}"
            )
        actual = lookup[key]
        if actual in seen_actual:
            raise ValueError(
                f"Channel '{ch}' resolved to '{actual}', which was already "
                f"matched by an earlier entry in `wanted` — check for "
                f"duplicates."
            )
        seen_actual.add(actual)
        resolved.append(actual)

    return resolved
