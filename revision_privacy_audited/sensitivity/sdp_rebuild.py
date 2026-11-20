"""Rebuild Table S-DP for the published run (root pipeline.py, Aug 2026).
Noise calibration there: get_noise_multiplier(target_eps, delta=min(1e-5, 1/fit_n),
sample_rate=batch/fit_n, epochs=5*15, accountant='rdp', alphas=A).
Executed by Opacus make_private (Poisson): q = 1/len(loader), steps = len(loader)*75.
Realized epsilon = RDP accountant on the executed (sigma, q, steps)."""
import math, csv
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
A = [1 + x / 10 for x in range(1, 100)] + list(range(12, 128))
N = {"Hugo": 26, "Riaz": 49, "Gide": 73, "Liu": 119, "IMvigor210": 298}
rows = []
for c, n in N.items():
    fit = n - math.ceil(0.2 * n)               # sklearn test_size=0.2 rounds the test part up
    L = math.ceil(fit / 8); q = 1 / L; steps = L * 75; d = min(1e-5, 1 / fit)
    for t in [1, 2, 4, 8, 10]:
        nm = get_noise_multiplier(target_epsilon=t, target_delta=d, sample_rate=8 / fit,
                                  epochs=75, accountant="rdp", alphas=A)
        a = RDPAccountant(); a.history = [(nm, q, steps)]
        rows.append([c, fit, round(8 / fit, 3), round(q, 3), t, round(nm, 2), steps,
                     round(a.get_epsilon(d, alphas=A), 3)])
with open("Table_S-DP_published_run.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Cohort", "Fit_n", "Nominal_q", "Executed_q", "Target_eps", "Noise_multiplier", "DP_steps", "Realized_eps"])
    w.writerows(rows)
for r in rows: print(r)
