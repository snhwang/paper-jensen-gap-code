"""Rebuild every result, figure and table of the revision, in dependency order.

  python run_all.py              light steps only. Reuses the outputs of the slow
                                 steps already in out/ (minutes).
  python run_all.py --heavy      also reruns the steps that read the raw images
                                 and zips (about 5 hours locally, disk-bound).
                                 Delete out/*_partial.csv first to start the
                                 native passes from scratch, otherwise they
                                 resume and skip sessions already done.
  python run_all.py --list       lists the steps.
  python run_all.py --from NAME  starts at step NAME.

See README.md for what each step produces and where each number, figure and
table of the paper comes from.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (name, arguments, heavy, what it produces)
STEPS = [
    ("cohort", ["v1_roi_stats.py"], False,
     "first-visit cohort; cross-sectional gaps, decomposition, b-dependence, spectrum"),
    ("motion", ["motion_covariates.py"], True,
     "head motion and eddy outlier fraction from the download zips"),
    ("native", ["native_pass.py"], True,
     "native pass over the 1,354 cohort sessions: gaps, tensor, noise simulation"),
    ("split_half_masks", ["split_half_masks.py"], False,
     "two balanced halves of each shell's gradient directions"),
    ("split_half", ["split_half_pass.py"], True,
     "gaps from split halves, a first noise-floor null, CSF floor, cohort sessions"),
    ("split_half_stats", ["split_half_stats.py"], False,
     "split-half gaps and the noise share of J_f; not used in the paper since S5 was cut (11/15)"),
    ("floor_null", ["floor_null_pass.py"], True,
     "noise-floor nulls compared (profile, b=3000 level, noise model), noise measures, cohort sessions"),
    ("floor_null_stats", ["floor_null_stats.py"], False,
     "the noise measures, and the noise-floor nulls compared (Sections 3.5.2, 4.4, S4)"),
    ("floor_null_sim", ["floor_null_sim.py", "--sample", "300"], True,
     "the nulls tested on simulated data with no true change, and the excess corrected (Section S4)"),
    ("post_native", ["post_native.py"], False,
     "FA correlations, beyond-tensor test, noise-corrected associations, summary"),
    ("long_sessions", ["long_sessions.py"], False,
     "longitudinal session list and the sessions still to process"),
    ("native_long", ["native_pass.py", "--sessions", "out/long_todo_sessions.csv",
                     "--tag", "long"], True,
     "native pass over the remaining longitudinal sessions"),
    ("longitudinal", ["longitudinal.py"], False,
     "within-person rates, measured and noise-corrected"),
    ("jf_pass", ["jf_pass.py", "--sample", "300"], True,
     "J_f on paired and on identical directions, random 300 sessions"),
    ("jf_stats", ["jf_stats.py"], False, "J_f against its noise null (not in the paper)"),
    ("example_maps", ["example_maps.py"], False, "example-participant voxel maps"),
    ("example_stats", ["example_stats.py"], False,
     "direction-set eigenvalues and example-participant numbers"),
    ("check_pairing", ["check_pairing.py"], False, "J_f direction-pairing angles (not in the paper)"),
    ("theory_table", ["theory_table.py"], False, "Supplementary Table S1"),
    ("crossing_sim", ["crossing_sim.py"], False, "crossing-fibre simulation (Section 2.5)"),
    ("sensitivity", ["sensitivity.py"], False, "Supplementary Table S2"),
    ("fig1", ["fig1_schematic.py"], False, "Figure 1"),
    ("fig_configurations", ["fig_configurations.py"], False, "Figure 2"),
    ("fig2", ["fig2_maps.py"], False, "Figure 3"),
    ("fig3", ["fig3_fa_relation.py"], False, "Figure 4"),
    ("heatmaps", ["fig_heatmaps.py", "both"], False, "Figure 5 and Supplementary Figure S1"),
    ("fig5_6", ["fig_region.py"], False, "Figures 6 and 7"),
    ("supp_tables", ["supp_tables.py"], False, "Supplementary Tables S3 and S4"),
]


def main():
    args = sys.argv[1:]
    if "--list" in args:
        for name, cmd, heavy, what in STEPS:
            print(f"{name:14s} {'[heavy] ' if heavy else ''}{' '.join(cmd):60s} {what}")
        return
    start = args[args.index("--from") + 1] if "--from" in args else STEPS[0][0]
    names = [s[0] for s in STEPS]
    for name, cmd, heavy, what in STEPS[names.index(start):]:
        if heavy and "--heavy" not in args:
            print(f"-- skip {name} (heavy; reusing existing output)")
            continue
        print(f"== {name}: {what}", flush=True)
        r = subprocess.run([sys.executable, *cmd], cwd=HERE)
        if r.returncode:
            sys.exit(f"step {name} failed with exit code {r.returncode}")


if __name__ == "__main__":
    main()
