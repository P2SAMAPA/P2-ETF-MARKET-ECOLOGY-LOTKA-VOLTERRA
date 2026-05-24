import numpy as np
from scipy.integrate import odeint
from sklearn.linear_model import Ridge
import config

def estimate_interaction_matrix(returns):
    """
    Estimate Lotka‑Volterra interaction matrix α and growth rates r.
    Model: dX_i/dt = r_i X_i + Σ_j α_ij X_i X_j
    Discretized: log_return_i ≈ r_i dt + Σ_j α_ij X_j dt
    Returns:
        r : (n,) array of intrinsic growth rates
        alpha : (n,n) interaction matrix with zero diagonal
    """
    n = returns.shape[1]
    # Price levels (start at 1)
    price = np.exp(returns.cumsum(axis=0)).values   # (T, n)
    # Log returns (dependent variables)
    y = returns.values                               # (T, n)
    r = np.zeros(n)
    alpha = np.zeros((n, n))
    for i in range(n):
        reg = Ridge(alpha=config.LAMBDA_REG, fit_intercept=True)
        reg.fit(price, y[:, i])                     # use price levels as features
        r[i] = reg.intercept_
        coef = reg.coef_
        coef[i] = 0.0                               # no self-interaction
        alpha[i, :] = coef
    return r, alpha

def lotka_volterra_ode(X, t, r, alpha):
    """dX/dt = r * X + X * (alpha @ X)   (elementwise)"""
    interaction = X * (alpha @ X)
    return r * X + interaction

def simulate(X0, r, alpha, days=5, steps_per_day=20):
    """Forward simulation of ODE."""
    t = np.linspace(0, days, days * steps_per_day + 1)
    sol = odeint(lotka_volterra_ode, X0, t, args=(r, alpha))
    return sol[-1]

def compute_ecological_scores(returns):
    """
    Compute ecological scores for each ETF:
    - Estimate r, alpha from the rolling window
    - Simulate forward `PREDICTION_HORIZON` days
    - Score = predicted log return (log(final_price / last_price))
    """
    r, alpha = estimate_interaction_matrix(returns)
    last_price = np.exp(returns.cumsum(axis=0).iloc[-1].values)
    X_final = simulate(last_price, r, alpha, days=config.PREDICTION_HORIZON)
    score = np.log(X_final / last_price)
    tickers = returns.columns
    return {ticker: score[i] for i, ticker in enumerate(tickers)}
