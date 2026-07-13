"""Turn factor pair tables into a deduplicated list of (utt_id, condition,
split) feature requests -- the set of frozen-encoder forward passes actually
needed to satisfy every pair, so extraction never runs the same
(utterance, condition) through an encoder twice.
"""

import pandas as pd

from isplit.data.augment import encode_channel_condition, encode_environment_condition


def collect_feature_requests(pair_sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []

    for factor in ("speaker", "content"):
        df = pair_sets.get(factor)
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            rows.append({"utt_id": row["a_utt_id"], "condition": None, "split": row["split"]})
            rows.append({"utt_id": row["b_utt_id"], "condition": None, "split": row["split"]})

    env_df = pair_sets.get("environment")
    if env_df is not None and not env_df.empty:
        for _, row in env_df.iterrows():
            cond_a = encode_environment_condition(row["a_noise_id"], row["a_snr_db"])
            cond_b = encode_environment_condition(row["b_noise_id"], row["b_snr_db"])
            rows.append({"utt_id": row["base_utt_id"], "condition": cond_a, "split": row["split"]})
            rows.append({"utt_id": row["base_utt_id"], "condition": cond_b, "split": row["split"]})

    chan_df = pair_sets.get("channel")
    if chan_df is not None and not chan_df.empty:
        for _, row in chan_df.iterrows():
            cond_a = encode_channel_condition(row["a_channel"])
            cond_b = encode_channel_condition(row["b_channel"])
            rows.append({"utt_id": row["base_utt_id"], "condition": cond_a, "split": row["split"]})
            rows.append({"utt_id": row["base_utt_id"], "condition": cond_b, "split": row["split"]})

    if not rows:
        return pd.DataFrame(columns=["utt_id", "condition", "split"])
    return pd.DataFrame(rows).drop_duplicates(subset=["utt_id", "condition"]).reset_index(drop=True)
