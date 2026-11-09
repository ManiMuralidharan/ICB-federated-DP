import warnings; warnings.filterwarnings("ignore")
import numpy as np, math
from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants import RDPAccountant

# CONFIG from pipeline.py
EPOCHS, ROUNDS, BATCH, DELTA = 5, 15, 8, 1e-5
dp_epochs = EPOCHS * ROUNDS   # 75
alphas = [1 + x/10.0 for x in range(1,100)] + list(range(12,128))

# representative training-partition sizes (len(Xf) ~ 80% of cohort fit partition).
# Use a spread so we see consistency across cohorts.
cohort_fit_sizes = {"small(~40)":40, "mid(~90)":90, "large(~240)":240}
targets = [10.0, 8.0, 4.0, 2.0, 1.0]

print(f"{'target eps':>10} {'fit_n':>10} {'q(sr)':>8} {'sigma':>8} {'steps_run':>10} {'eps_readback':>13} {'match':>7}")
for teps in targets:
    for name,n in cohort_fit_sizes.items():
        q = BATCH / n                      # sampling rate
        td = min(DELTA, 1.0/n)             # = 1e-5 for these sizes
        # sigma the pipeline derives (analytic, for the full 75-epoch trajectory)
        sigma = get_noise_multiplier(target_epsilon=teps, target_delta=td,
                                     sample_rate=q, epochs=dp_epochs,
                                     accountant="rdp", alphas=alphas)
        # ACTUAL steps the code runs: 75 epochs of Poisson-sampled minibatches,
        # expected steps per epoch = 1/q, so total T = round(dp_epochs / q)
        T = round(dp_epochs / q)
        # persistent accountant stepping through the ACTUAL trajectory at fixed sigma
        acct = RDPAccountant()
        for _ in range(T):
            acct.step(noise_multiplier=sigma, sample_rate=q)
        eps_rb = acct.get_epsilon(delta=td)
        ok = "OK" if eps_rb <= teps*1.05 + 1e-6 else "HIGH"
        print(f"{teps:>10.1f} {name:>10} {q:>8.3f} {sigma:>8.3f} {T:>10d} {eps_rb:>13.3f} {ok:>7}")
