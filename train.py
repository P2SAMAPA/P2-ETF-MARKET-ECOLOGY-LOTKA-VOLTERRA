import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from huggingface_hub import HfApi
import config
import data_manager as dm
from lotka_volterra import compute_ecological_scores

def normalize_scores(score_dict):
    scores = np.array(list(score_dict.values()))
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-12:
        return {k: 0.0 for k in score_dict}
    norm = (scores - min_s) / (max_s - min_s)
    return {ticker: float(norm[i]) for i, ticker in enumerate(score_dict.keys())}

def run_for_window(returns, window_days):
    if len(returns) < window_days:
        return None
    ret_window = returns.iloc[-window_days:]
    # Compute raw scores (predicted log returns)
    raw_scores = compute_ecological_scores(ret_window)
    norm_scores = normalize_scores(raw_scores)
    # Top 3 by normalized score
    sorted_norm = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
    top_etfs = [{"ticker": t, "ecological_score_norm": s, "raw_score": raw_scores[t]} for t, s in sorted_norm[:3]]
    return {
        "window": window_days,
        "top_etfs": top_etfs,
        "all_scores_raw": raw_scores,
        "all_scores_norm": norm_scores
    }

def main():
    print("Loading master data...")
    dm.load_master_data()
    results = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "windows": config.WINDOWS,
        "prediction_horizon": config.PREDICTION_HORIZON,
        "universes": {}
    }
    for uni_name in config.UNIVERSES.keys():
        print(f"Processing {uni_name}...")
        returns = dm.get_universe_returns(uni_name)
        if returns.empty:
            print(f"  No data -> skipping")
            continue
        per_window = []
        for w in config.WINDOWS:
            print(f"  Window {w} days")
            out = run_for_window(returns, w)
            if out:
                per_window.append(out)
            else:
                print(f"    Not enough data for window {w}")
        # Select best window = one with highest max absolute raw score
        best = None
        best_score = -np.inf
        best_data = None
        for pw in per_window:
            max_abs = max(abs(v) for v in pw["all_scores_raw"].values())
            if max_abs > best_score:
                best_score = max_abs
                best = pw["window"]
                best_data = pw
        if best_data:
            results["universes"][uni_name] = {
                "best_window": best,
                "best_window_data": {
                    "top_etfs": best_data["top_etfs"],
                    "all_scores_norm": best_data["all_scores_norm"],
                    "all_scores_raw": best_data["all_scores_raw"]
                }
            }
        else:
            results["universes"][uni_name] = None
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"output/ecological_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")
    api = HfApi(token=config.HF_TOKEN)
    try:
        api.upload_file(
            path_or_fileobj=out_file,
            path_in_repo=os.path.basename(out_file),
            repo_id=config.OUTPUT_REPO,
            repo_type="dataset"
        )
        print(f"Uploaded to {config.OUTPUT_REPO}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
