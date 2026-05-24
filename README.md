# Market Ecology – Lotka‑Volterra for ETFs

Ecological competition/cooperation model applied to ETF returns.  
Estimates interaction matrix α from rolling windows, simulates forward dynamics, and produces a **predicted growth score**.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63 to 5040 days)
- Lotka‑Volterra ODE: `dX_i/dt = r_i X_i + Σ_j α_ij X_i X_j`
- Regularised linear regression to estimate r and α
- Forward simulation for a fixed horizon (e.g., 5 days)
- Score = predicted log return (normalised per universe)
- Results stored on Hugging Face Datasets: `P2SAMAPA/p2-etf-lotka-volterra-results`
- Streamlit dashboard with refresh button

## Usage

1. Set `HF_TOKEN` environment variable.
2. Run training: `python train.py`
3. Launch dashboard: `streamlit run streamlit_app.py`
4. (Optional) GitHub Actions runs daily.

## Interpretation

- Positive α_ij means ETF j helps ETF i grow (cooperation).
- Negative α_ij means competition.
- The final score is the expected log return over the next `PREDICTION_HORIZON` days.
- A high score suggests the ETF occupies a favourable ecological niche.

## Requirements

See `requirements.txt`.
