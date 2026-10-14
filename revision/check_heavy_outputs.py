"""Check that the stored outputs of the slow passes match the current scripts.

native_pass.py and jf_pass.py take hours, so run_all.py reuses their stored
outputs unless --heavy is given. This script reruns a few sessions of each
stored output with the current code and compares every numeric column except
the run time. Both
passes seed their noise per session, so a match should be exact up to
floating-point rounding.

Checked:  out/native_pass.csv, out/native_pass_long.csv, out/jf_pass.csv
          (the first N sessions of each, in file order)
Usage:    python check_heavy_outputs.py [N]          (default N = 3)
Exit code 1 if any value differs by more than 1e-9 relative.
"""
import sys

import numpy as np
import pandas as pd

import jf_pass
import native_pass
from config import OUT

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TOL = 1e-9
worst = 0.0
for name, module in (("native_pass.csv", native_pass), ("native_pass_long.csv", native_pass),
                     ("jf_pass.csv", jf_pass)):
    stored = pd.read_csv(OUT / name)
    for sid in stored.Session_ID.drop_duplicates().head(N):
        old = stored[stored.Session_ID == sid].set_index("ROI").sort_index()
        new = pd.DataFrame(module.one(sid)).set_index("ROI").sort_index()
        # "seconds" is the run time of the session, not a result.
        cols = [c for c in old.columns if c in new.columns and c != "seconds"
                and pd.api.types.is_numeric_dtype(old[c])]
        missing = sorted(set(old.columns) - set(new.columns))
        if "shell_bval" in cols:          # one row per ROI and shell
            old = old.set_index("shell_bval", append=True).sort_index()
            new = new.set_index("shell_bval", append=True).sort_index()
            cols.remove("shell_bval")
        a, b = old[cols].to_numpy(float), new.loc[old.index, cols].to_numpy(float)
        rel = np.nanmax(np.abs(a - b) / np.maximum(np.abs(a), 1e-12))
        nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
        worst = max(worst, rel, np.inf if nan_mismatch or missing else 0)
        print(f"{name:22s} {sid}: {len(cols)} columns x {len(old)} rows, max relative "
              f"difference {rel:.1e}" + (f", NaN mismatches {nan_mismatch}" if nan_mismatch else "")
              + (f", columns no longer produced {missing}" if missing else ""), flush=True)
print("OK: stored outputs match the current scripts" if worst <= TOL
      else "MISMATCH: rerun the slow passes (run_all.py --heavy)")
sys.exit(int(worst > TOL))
