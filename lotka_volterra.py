import numpy as np
from scipy.integrate import odeint
from sklearn.linear_model import Ridge
import config          # <-- added this import

def estimate_interaction_matrix(returns):
    """
    Estimate Lotka‑Volterra interaction matrix α (competition/cooperation)
    from daily returns.
    Model: dX_i / dt ≈ r_i X_i + Σ_j α_ij X_i X_j
    Discretised: ΔX_i / Δt ≈ r_i X_i + Σ_j α_ij X_i X_j
    Returns: α (n x n) with zero diagonal.
    """
    n = returns.shape[1]
    X = returns.values  # returns are already log returns, but we need actual price changes? Use cumulative returns.
    # Use cumulative price (start from 1)
    cum = np.exp(np.cumsum(X, axis=0))  # shape (T, n)
    # ΔX = diff(cum, axis=0)
    dX = np.diff(cum, axis=0)           # (T-1, n)
    X_mid = cum[:-1]                    # mid-point values for X
    # For each time t, features = X_mid[t] * X_mid[t] (outer product)
    # but we need per equation: dX_i = (r_i X_i + Σ_j α_ij X_i X_j) dt
    # Rearranged: dX_i / (X_i dt) = r_i + Σ_j α_ij X_j
    # Using log returns: dX_i / X_i = log_return_i (approx)
    # So log_return_i = r_i dt + Σ_j α_ij X_j dt
    # Let y_i = log_return_i (daily), features = X_j (the level)
    y = X[1:]  # daily log returns (T-1, n)
    X_level = cum[:-1]  # (T-1, n)
    # Solve for each i: y_i = r_i + Σ_j α_ij X_level_j
    coefs = []
    for i in range(n):
        reg = Ridge(alpha=config.LAMBDA_REG, fit_intercept=True)
        reg.fit(X_level, y[:, i])
        r_i = reg.intercept_
        alpha_i = reg.coef_
        alpha_i[i] = 0  # no self-interaction
        coefs.append(alpha_i)
    alpha = np.array(coefs)
    r = np.array([reg.intercept_ for reg in coefs])  # same as above, but safe
    return r, alpha

def lotka_volterra_ode(X, t, r, alpha):
    """dX/dt = r * X + X * (alpha @ X)   (elementwise)"""
    interaction = X * (alpha @ X)
    return r * X + interaction

def simulate(X0, r, alpha, days=5, steps_per_day=20):
    """
    Simulate future population (price level) starting from last observed X0.
    Returns final population vector.
    """
    t = np.linspace(0, days, days * steps_per_day + 1)
    sol = odeint(lotka_volterra_ode, X0, t, args=(r, alpha))
    return sol[-1]  # final population

def compute_ecological_scores(returns):
    """
    Estimate r, alpha from the training window, then simulate forward.
    Score for each ETF = predicted log return (or final population growth).
    Returns dictionary of scores.
    """
    r, alpha = estimate_interaction_matrix(returns)
    X0 = np.exp(np.cumsum(returns.iloc[-1].values))  # last observed price level (starting at 1)
    X_final = simulate(X0, r, alpha, days=config.PREDICTION_HORIZON)
    # Score = final log price (or log return relative to last)
    score = np.log(X_final / X0)
    tickers = returns.columns
    return {ticker: score[i] for i, ticker in enumerate(tickers)}
