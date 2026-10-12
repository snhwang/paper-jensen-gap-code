"""Paths shared by every analysis script of the revision.

Each path can be overridden with the environment variable of the same name, so
the pipeline can run on another machine without editing the scripts. By default
all data are looked for under one data root:

  JG_DATA      data root (default: ../data next to this folder)
  JG_HCP       per-session ROI tables and demographics (cr_gaps_long.csv, ...)
  JG_DTI       per-session processing folders (session_*/processed, inputs)
  JG_MAPS      example-participant voxel maps written by example_maps.py
  JG_OLDMAPS   example-participant maps from the original pipeline
  JG_ATLAS     FSL/JHU atlases (JHU-ICBM-labels-1mm, FMRIB58_FA_1mm)

Outputs are written inside this folder and are not committed:
  OUT          out/             intermediate tables and text summaries
  FIGS         figures/         main-text figures
  SUPP_FIGS    figures_supp/    supplementary figures
  TABLES       tables_supp/     LaTeX tables for the supplement
"""
import os
from pathlib import Path

# Fixed timestamp for matplotlib's PDF metadata, so that rerunning a figure
# script gives a byte-identical file when the figure itself has not changed.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")   # 2026-01-01

HERE = Path(__file__).resolve().parent
ROOT = HERE

DATA = Path(os.environ.get("JG_DATA", str(HERE.parent / "data")))
HCP = Path(os.environ.get("JG_HCP", str(DATA / "HCP")))
DTI = Path(os.environ.get("JG_DTI", str(DATA / "dti_output")))
MAPS = Path(os.environ.get("JG_MAPS", str(HCP / "voxel_maps_revision")))
OLDMAPS = Path(os.environ.get("JG_OLDMAPS", str(HCP / "voxel_maps")))
ATLAS = Path(os.environ.get("JG_ATLAS", str(DATA / "atlases")))

OUT = HERE / "out"
FIGS = ROOT / "figures"
SUPP_FIGS = ROOT / "figures_supp"
TABLES = ROOT / "tables_supp"
for _d in (OUT, FIGS, SUPP_FIGS, TABLES):
    _d.mkdir(exist_ok=True)

# Example participant of Figures 2 and 3 (a processing-session label, not a
# participant identifier).
EXAMPLE_SESSION = "session_20260222_031250"

# The 16 JHU ICBM-DTI-81 regions (label numbers) used throughout.
ROI_DEFS = {
    "CR_all": [23, 24, 25, 26, 27, 28], "ACR": [23, 24], "SCR": [25, 26],
    "PCR": [27, 28], "Genu": [3], "Body": [4], "Splenium": [5],
    "alic": [17, 18], "plic": [19, 20], "rlic": [21, 22], "ec": [33, 34],
    "slf": [41, 42], "ptr": [29, 30], "ss": [31, 32], "cingulum_cc": [35, 36],
    "tapetum": [47, 48],
}
ROIS = list(ROI_DEFS)
NAMES = {
    "CR_all": "Corona radiata (all)", "ACR": "Anterior corona radiata",
    "SCR": "Superior corona radiata", "PCR": "Posterior corona radiata",
    "Genu": "Genu of corpus callosum", "Body": "Body of corpus callosum",
    "Splenium": "Splenium of corpus callosum", "alic": "Internal capsule, anterior",
    "plic": "Internal capsule, posterior", "rlic": "Internal capsule, retrolenticular",
    "ec": "External capsule", "slf": "Superior longitudinal fasciculus",
    "ptr": "Posterior thalamic radiation", "ss": "Sagittal stratum",
    "cingulum_cc": "Cingulum (cingulate gyrus)", "tapetum": "Tapetum",
}
