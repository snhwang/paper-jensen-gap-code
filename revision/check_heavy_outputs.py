"""Check that the stored outputs of the slow passes match the current scripts.

native_pass.py and jf_pass.py take hours, so run_all.py reuses their stored
outputs unless --heavy is given. Two modes:

  python check_heavy_outputs.py [N]
      Spot check. Reruns the first N sessions (default 3) of each stored
      output with the current code and compares them. Under a minute.

  python check_heavy_outputs.py --paper ROOT
      Full check after a complete rerun (run_all.py --heavy), for example of
      the public copy of this code against the paper's own outputs. ROOT is
      the paper's source tree, with figures/, figures_supp/, tables_supp/ and
      revision/analysis/out/. Compares every session and region of the slow
      outputs and motion.csv, and requires the text summaries (out/*.txt), the
      supplementary tables and the figure PDFs to have identical content.

      Content is compared without three artefacts of the operating system and
      Python build, which do not change what is printed or drawn:
        - line endings (Windows writes CRLF, Linux LF),
        - PDF stream compression (zlib builds differ, so every stream is
          decompressed, and the stream lengths, the objects that store them
          and the byte-offset table are ignored),
        - the sign of zero in PDF coordinates ("-0" and "0" draw the same point).

Files:   out/native_pass.csv, out/native_pass_long.csv, out/jf_pass.csv
Both passes seed their noise per session, so a match should be exact up to
floating-point rounding. The "seconds" column is the run time of each session
and is not compared. Exit code 1 if any value differs by more than 1e-9
relative, if a session is in one set but not the other, or if any summary,
table or figure differs.
"""
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from config import FIGS, OUT, SUPP_FIGS, TABLES

TOL = 1e-9
FILES = {"native_pass.csv": ["Session_ID", "ROI", "shell_bval"],
         "native_pass_long.csv": ["Session_ID", "ROI", "shell_bval"],
         "jf_pass.csv": ["Session_ID", "ROI"]}


def compare(old, new, keys, label):
    """Largest relative difference between two tables aligned on keys."""
    old, new = old.set_index(keys).sort_index(), new.set_index(keys).sort_index()
    only_old, only_new = old.index.difference(new.index), new.index.difference(old.index)
    cols = [c for c in old.columns if c in new.columns and c != "seconds"
            and pd.api.types.is_numeric_dtype(old[c])]
    common = old.index.intersection(new.index)
    a, b = old.loc[common, cols].to_numpy(float), new.loc[common, cols].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.nanmax(np.abs(a - b) / np.maximum(np.abs(a), 1e-12)) if a.size else 0.0
    nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
    missing = sorted(set(old.columns) - set(new.columns))
    print(f"{label}: {len(common)} rows x {len(cols)} columns, max relative difference "
          f"{rel:.1e}"
          + (f", NaN mismatches {nan_mismatch}" if nan_mismatch else "")
          + (f", rows only in old {len(only_old)}" if len(only_old) else "")
          + (f", rows only in new {len(only_new)}" if len(only_new) else "")
          + (f", columns no longer produced {missing}" if missing else ""), flush=True)
    bad = nan_mismatch or len(only_old) or len(only_new) or missing
    return np.inf if bad else rel


def spot(n):
    import jf_pass
    import native_pass
    worst = 0.0
    for name, keys in FILES.items():
        module = jf_pass if name.startswith("jf") else native_pass
        stored = pd.read_csv(OUT / name)
        for sid in stored.Session_ID.drop_duplicates().head(n):
            new = pd.DataFrame(module.one(sid))
            worst = max(worst, compare(stored[stored.Session_ID == sid], new, keys,
                                       f"{name:22s} {sid}"))
    return worst


# Everything up to a stream (object text and the stream's dictionary), then the
# stream itself. Matching from one stream to the next covers the whole file.
PDF_STREAM = re.compile(rb"(.*?>>)(\s*stream\r?\n)(.*?)(\r?\nendstream)", re.S)
NEG_ZERO = re.compile(rb"(?<![\d.])-0(?![\d.])")


def content(path):
    """File bytes without the platform artefacts listed in the docstring."""
    raw = path.read_bytes()
    if path.suffix != ".pdf":
        return raw.replace(b"\r\n", b"\n")

    def plain(m):
        head, body = m.group(1), m.group(3)
        if b"FlateDecode" in head:
            body = zlib.decompress(body)
        return re.sub(rb"/Length \d+", b"", head) + b"\nstream\n" + NEG_ZERO.sub(b"0", body)

    text = PDF_STREAM.sub(plain, raw)
    # Objects that hold only an integer are compressed stream lengths.
    text = re.sub(rb"(\d+ 0 obj\s+)\d+(\s+endobj)", rb"\1\2", text)
    return re.sub(rb"\nxref\n.*", b"", text, flags=re.S)   # byte offsets of the objects


def same_content(new_dir, old_dir, pattern, label):
    names = sorted(p.name for p in new_dir.glob(pattern))
    differ = [n for n in names
              if not (old_dir / n).exists() or content(old_dir / n) != content(new_dir / n)]
    print(f"{label:22s} {len(names)} files, {len(differ)} differ"
          + (f": {', '.join(differ)}" if differ else ""), flush=True)
    return np.inf if differ or not names else 0.0


def full(root):
    paper_out = root / "revision" / "analysis" / "out"
    worst = 0.0
    for name, keys in {**FILES, "motion.csv": ["Subject_ID"]}.items():
        worst = max(worst, compare(pd.read_csv(paper_out / name), pd.read_csv(OUT / name),
                                   keys, f"{name:22s} all sessions"))
    worst = max(worst, same_content(OUT, paper_out, "*.txt", "text summaries"),
                same_content(TABLES, root / "tables_supp", "*.tex", "supplementary tables"),
                same_content(FIGS, root / "figures", "*.pdf", "figures"),
                same_content(SUPP_FIGS, root / "figures_supp", "*.pdf", "supplementary figures"))
    return worst


if __name__ == "__main__":
    if "--paper" in sys.argv:
        worst = full(Path(sys.argv[sys.argv.index("--paper") + 1]))
    else:
        worst = spot(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
    print("OK: the outputs match" if worst <= TOL else "MISMATCH: see the lines above")
    sys.exit(int(worst > TOL))
