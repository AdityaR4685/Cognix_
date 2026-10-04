"""Exact sealed official-label alignment function; no calibrator."""
def labels_from_table(table):
    import numpy as np
    ticks = table["tick"].to_numpy()
    labels = table["anomaly"].to_numpy()
    if ticks.dtype.kind not in "iu" or len(ticks) < 2 or not np.array_equal(ticks, np.arange(len(ticks))):
        raise ValueError("LABEL_TICKS_MUST_BE_COMPLETE_CONTIGUOUS_INTEGER_0_TO_N_MINUS_1")
    if labels.dtype.kind not in "biu" or not np.isin(labels, (0, 1)).all():
        raise ValueError("OFFICIAL_LABELS_MUST_BE_BOOLEAN_OR_INTEGER_0_1_NO_NULLS")
    return labels.astype(np.int64)
