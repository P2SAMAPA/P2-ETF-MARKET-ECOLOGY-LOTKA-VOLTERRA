import numpy as np
from scipy.integrate import odeint
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import config

def estimate_interaction_matrix(returns):
    """
    Estimate r and alpha from log returns.
    Uses standardised price levels as features to avoid numerical issues.
    Returns r, alpha, and the StandardScaler fitted to price levels.
    """
    # Price levels (start at 1)
    price = np.exp(returns.cumsum(axis=0)).values   # (T, n)
    y = returns.values                               # (T, n)
    # Remove rows with any NaN
    mask = ~(np.isnan(price).any(axis=1) | np.isnan(y).any(axis=1))
    price = price[mask]
    y = y[mask]
    if len(price) < 2:
        raise ValueError("Not enough clean data after removing NaNs")
    # Standardise price levels (each ETF separately)
    scaler = StandardScaler()
    price_scaled = scaler.fit_transform(price)
    n = returns.shape[1]
    r = np.zeros(n)
    alpha = np.zeros((n, n))
    for i in range(n):
        reg = Ridge(alpha=config.LAMBDA_REG, fit_intercept=True)
        reg.fit(price_scaled, y[:, i])
        r[i] = reg.intercept_
        coef = reg.coef_
        coef[i] = 0.0          # no self‑interaction
        alpha[i, :] = coef
    return r, alpha, scaler

def lotka_volterra_ode(X, t, r, alpha):
    """dX/dt = r * X + X * (alpha @ X)"""
    interaction = X * (alpha @ X)
    return r * X + interaction

def simulate(X0_scaled, r, alpha, days, steps_per_day=20):
    t = np.linspace(0, days, days * steps_per_day + 1)
    sol = odeint(lotka_volterra_ode, X0_scaled, t, args=(r, alpha))
    return sol[-1]

def compute_ecological_scores(returns):
    """
    Compute predicted log return (score) for each ETF.
    Handles NaN by dropping rows.
    """
    # Drop any row that contains NaN in returns
    returns_clean = returns.dropna()
    if len(returns_clean) < 10:
        raise ValueError("Insufficient clean returns data")
    r, alpha, scaler = estimate_interaction_matrix(returns_clean)
    # Last observed price (actual, not standardised)
    last_price = np.exp(returns_clean.cumsum(axis=0).iloc[-1].values).reshape(1, -1)
    # Standardise
    last_price_scaled = scaler.transform(last_price).flatten()
    # Simulate
    final_scaled = simulate(last_price_scaled, r, alpha, days=config.PREDICTION_HORIZON)
    # Inverse transform to actual price
    final_price = scaler.inverse_transform(final_scaled.reshape(1, -1)).flatten()
    # Score = log return over horizon
    score = np.log(final_price / last_price.flatten())
    tickers = returns_clean.columns
    return {ticker: score[i] for i, ticker in enumerate(tickers)}
