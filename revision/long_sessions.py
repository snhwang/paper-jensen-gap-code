"""Sessions of the longitudinal analysis, and those still to be put through
native_pass.py.

The longitudinal set is every visit (one session per subject-visit, the latest if
duplicated) of the participants with two or more visits. Sessions with a
non-standard acquisition (not 14/93/92 volumes) are flagged, as in the
cross-sectional cohort.

Writes out/long_sessions.csv (all) and out/long_todo_sessions.csv (not yet in
out/native_pass.csv). Usage:  python long_sessions.py
"""
import glob

import numpy as np
import pandas as pd

from config import DTI, HCP, OUT



def clean_age(s):
    return pd.to_numeric(s.astype(str).str.replace("90 or older", "90", regex=False),
                         errors="coerce")


dem = pd.read_csv(glob.glob(str(HCP / "AABC2_subjects_*.csv"))[0], low_memory=False)
dem = dem.dropna(subset=["id_event"]).copy()
dem["Subject_ID"] = dem.id_event.str.rsplit("_", n=1).str[0]
dem["Visit"] = dem.id_event.str.rsplit("_", n=1).str[1]
dem["age"] = clean_age(dem.age_open)

g = pd.read_csv(HCP / "cr_gaps_long.csv", usecols=["Subject_ID", "Visit", "Session_ID",
                                                   "ROI", "shell_bval", "J_ln_D_mean"])
g = g[(g.ROI == "CR_all") & (g.shell_bval == 1500)]
g = g.sort_values("Session_ID").groupby(["Subject_ID", "Visit"]).tail(1)
g = g.merge(dem[["Subject_ID", "Visit", "age", "sex", "days_from_V1"]], on=["Subject_ID", "Visit"])
g = g.dropna(subset=["age", "sex"])
g = g[g.groupby("Subject_ID").Visit.transform("nunique") >= 2]


def counts(sid):
    f = DTI / sid / "inputs" / "dwi.bval"
    if not f.exists():
        return np.nan, np.nan, np.nan
    b = np.loadtxt(f).ravel()
    return (int((b < 50).sum()), int((abs(b - 1500) < 100).sum()),
            int((abs(b - 3000) < 100).sum()))


acq = pd.DataFrame([(s, *counts(s)) for s in g.Session_ID],
                   columns=["Session_ID", "n_b0", "n_b1500", "n_b3000"])
g = g.merge(acq, on="Session_ID")
g["standard"] = (g.n_b0 == 14) & (g.n_b1500 == 93) & (g.n_b3000 == 92)
g[["Subject_ID", "Visit", "Session_ID", "age", "days_from_V1", "sex", "standard"]].to_csv(
    OUT / "long_sessions.csv", index=False)

done = set(pd.read_csv(OUT / "native_pass.csv", usecols=["Session_ID"]).Session_ID)
todo = g[~g.Session_ID.isin(done)]
todo[["Session_ID"]].to_csv(OUT / "long_todo_sessions.csv", index=False)
print(f"{g.Subject_ID.nunique()} participants, {len(g)} sessions; "
      f"non-standard {int((~g.standard).sum())}; already processed "
      f"{len(g) - len(todo)}; to do {len(todo)}")
