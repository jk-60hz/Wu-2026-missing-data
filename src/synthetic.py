"""Benchmark model systems from the paper's Methods (Eq. 3 and Eq. 4 with i=1)."""
import numpy as np
from scipy.integrate import solve_ivp


def ecology(n=500, discard=100, x0=(0.4, 0.5, 0.6)):
    """3-species discrete ecology system, Eq. 3."""
    s = np.empty((n, 3))
    s[0] = x0
    for t in range(n - 1):
        x, y, z = s[t]
        s[t + 1] = (x * (4 - 4 * x - 2 * y - 0.4 * z),
                    y * (3.1 - 0.31 * x - 3.1 * y - 0.93 * z),
                    z * (2.12 + 0.636 * x + 0.636 * y - 2.12 * z))
    return s[discard:]


def lorenz(n=1600, discard=100, t_span=(10, 40), x0=(0.3, 0.33, 0.36), a=10, b=28, c=8 / 3):
    """Classic 3D Lorenz system sampled on [10, 40] (Eq. 4 with a single subsystem)."""
    f = lambda t, s: [a * (s[1] - s[0]), s[0] * (b - s[2]) - s[1], s[0] * s[1] - c * s[2]]
    t = np.linspace(*t_span, n)
    sol = solve_ivp(f, (0, t_span[1]), x0, t_eval=t, rtol=1e-9, atol=1e-9)
    return sol.y.T[discard:]
