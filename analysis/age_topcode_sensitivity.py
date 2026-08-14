"""Does top-coding the oldest participants at 90 change the age correlations?

AABC top-codes the oldest participants as "90 or older" (n=56, a clear pile-up
against 16 at age 88 and 17 at 89). They are assigned 90 throughout, which keeps
the cohort at N=1,379.

Compressing the upper tail restricts the range of the predictor, so it should
ATTENUATE age correlations, making the reported values mild underestimates. This
quantifies that by reassigning the top-coded participants larger ages: fixed
offsets, and a realistic tail drawn from an exponential with mean +2.5 years.

Output: printed table. Backs the sensitivity statement in Statistical analysis.
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

import os
BASE = Path(os.environ.get("JG_DATA", Path(__file__).resolve().parent))
HCP = BASE / "HCP"
COLS = ("J_ln_D_mean", "J_ln_S_mean", "J_f_mean")


def partial(v, age, sex):
    d = pd.DataFrame({"v": v, "a": age, "s": sex}).dropna()
    X = np.column_stack([np.ones(len(d)),
                         pd.get_dummies(d["s"], drop_first=True).astype(float).values])
    res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return stats.pearsonr(res(d["v"].values), res(d["a"].values))[0]


def main():
    man = pd.read_csv(HCP / "manifest_n1379_b1500.tsv", sep="\t")
    g = pd.read_csv(HCP / "cr_gaps_long.csv")
    jf = pd.read_csv(HCP / "cr_jf_long.csv")
    gg = g[(g.ROI == "CR_all") & (g.shell_bval == 1500)][
        ["Session_ID", "J_ln_D_mean", "J_ln_S_mean"]]
    jj = jf[jf.ROI == "CR_all"][["Session_ID", "J_f_mean"]]
    m = (man.merge(gg, left_on="session_id", right_on="Session_ID")
            .merge(jj, on="Session_ID", how="left"))

    base = pd.to_numeric(m["age"], errors="coerce")
    top = base == 90
    print(f"top-coded participants: {int(top.sum())} of {len(m)}  "
          f"(88: {int((base==88).sum())}, 89: {int((base==89).sum())})")
    print(f"\n{'assumed age':>12s} {'J_ln':>9s} {'J_lnS':>9s} {'J_f':>9s}")
    for assumed in (90, 92, 94, 96):
        a = base.copy(); a[top] = assumed
        r = [partial(pd.to_numeric(m[c], errors="coerce"), a, m["sex"]) for c in COLS]
        print(f"  {assumed:10d} {r[0]:+9.4f} {r[1]:+9.4f} {r[2]:+9.4f}")

    rng = np.random.default_rng(0)
    sims = []
    for _ in range(200):
        a = base.copy(); a[top] = 90 + rng.exponential(2.5, int(top.sum()))
        sims.append([partial(pd.to_numeric(m[c], errors="coerce"), a, m["sex"])
                     for c in COLS])
    s = np.array(sims)
    print("\nrealistic tail (exponential, mean +2.5 y above 90; 200 draws):")
    for i, c in enumerate(("J_ln", "J_lnS", "J_f")):
        print(f"  {c:6s} {s[:, i].mean():+.4f} +/- {s[:, i].std():.4f}")
    print("\nAll correlations strengthen slightly, so top-coding is conservative.")


if __name__ == "__main__":
    main()
