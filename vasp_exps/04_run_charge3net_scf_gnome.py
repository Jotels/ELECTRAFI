#!/usr/bin/env python3
import os
import re
import shutil
import time
import argparse
import csv
import tempfile
import subprocess
import sys

import pandas as pd
import numpy as np
import lz4.frame
import wandb

from ase.io.vasp import read_vasp

# -------------------------
# Global monkey-patch: avoid spglib bug in get_space_group_info
# -------------------------
import pymatgen.core.structure as pmg_structure


def patch_lmaxmix_for_dftu(incar_path):
    """
    Enforce ChargE3Net-style LMAXMIX:
      - If any LDAUL = 3 (f), set LMAXMIX = 6
      - Else if any LDAUL = 2 (d), set LMAXMIX = 4
      - Only if DFT+U is actually on (LDAU = .TRUE.)
    """
    if not os.path.exists(incar_path):
        return

    with open(incar_path, "r") as f:
        lines = f.readlines()

    ldaul_line = None
    ldau_on = False
    for line in lines:
        key = line.split("=", 1)[0].strip().upper() if "=" in line else ""
        if key == "LDAU":
            if "T" in line.upper():  # .TRUE. or T
                ldau_on = True
        elif key == "LDAUL":
            ldaul_line = line

    if not ldau_on or ldaul_line is None:
        # No DFT+U → don't touch LMAXMIX
        return

    # Parse LDAUL integers, e.g. "LDAUL = 2 -1 -1"
    try:
        ldaul_vals = [
            int(x) for x in ldaul_line.split("=", 1)[1].split()
            if x.strip().lstrip("+-").isdigit()
        ]
    except Exception:
        return

    has_f = any(l == 3 for l in ldaul_vals)
    has_d = any(l == 2 for l in ldaul_vals)

    if not (has_f or has_d):
        return

    desired = 6 if has_f else 4

    new_lines = []
    saw_lmaxmix = False
    for line in lines:
        key = line.split("=", 1)[0].strip().upper() if "=" in line else ""
        if key == "LMAXMIX":
            new_lines.append(f"LMAXMIX = {desired}\n")
            saw_lmaxmix = True
        else:
            new_lines.append(line)

    if not saw_lmaxmix:
        new_lines.append(f"LMAXMIX = {desired}\n")

    with open(incar_path, "w") as f:
        f.writelines(new_lines)


def _fake_get_space_group_info(self, symprec=0.01, angle_tolerance=5.0):
    """
    Replacement for Structure.get_space_group_info.

    MPStaticSet / Kpoints.automatic_density_by_vol only check:

        structure.get_space_group_info()[0][0] == "F"

    to see if it's face-centered. Returning ("P1", 1) is enough and
    bypasses the buggy SpacegroupAnalyzer path that expects
    `_space_group_data` to have an `.international` attribute.
    """
    return ("P1", 1)


# Patch the canonical Structure class used everywhere in pymatgen
pmg_structure.Structure.get_space_group_info = _fake_get_space_group_info

# -------------------------
# Now safe to import pymatgen I/O layers
# -------------------------
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.io.vasp.outputs import Oszicar, Outcar, Chgcar
from pymatgen.electronic_structure.core import Spin

# -------------------------
# Helpers
# -------------------------

_float_re = re.compile(r'[-+]?\d*\.?\d+(?:[Ee][+-]?\d+)?')


def atoms_from_chgcar(chgcar_path):
    """
    Load an ASE Atoms from a CHGCAR or CHGCAR.lz4.
    We don't care about the density here, only the structure.
    """
    if chgcar_path.endswith(".lz4"):
        tmpfd, tmppath = tempfile.mkstemp(prefix="tmp_chgcar_")
        os.close(tmpfd)
        with lz4.frame.open(chgcar_path, "rb") as src, open(tmppath, "wb") as dst:
            shutil.copyfileobj(src, dst)
        atoms = read_vasp(tmppath)
        os.remove(tmppath)
    else:
        atoms = read_vasp(chgcar_path)
    return atoms


def _get_chgcar_dim(chg: Chgcar):
    """
    Robustly get (nx, ny, nz) for a Chgcar.

    Handles:
      - chg.dim property (newer pymatgen)
      - chg.data being dict[Spin, np.ndarray]
      - chg.data being a single np.ndarray
    """
    dim = getattr(chg, "dim", None)
    if dim is not None:
        dim = tuple(dim)
        if len(dim) == 3:
            return dim

    data = chg.data

    if isinstance(data, dict) and len(data) > 0:
        arr = np.asarray(next(iter(data.values())))
    else:
        arr = np.asarray(data)

    if arr.ndim != 3:
        raise RuntimeError(f"Expected 3D grid, got shape {arr.shape}")

    return tuple(arr.shape)


def get_chgcar_grid_dims(chgcar_path: str):
    """
    Parse (NGXF, NGYF, NGZF) from a standard VASP CHGCAR using pymatgen.
    Assumes chgcar_path is an uncompressed CHGCAR (not .lz4).
    """
    chg = Chgcar.from_file(chgcar_path)
    return _get_chgcar_dim(chg)


def patch_incar_ngxf(incar_path, grid_dims):
    """
    Overwrite/add NGXF/NGYF/NGZF in an INCAR file to match `grid_dims`.
    (Currently not called; kept for potential future use.)
    """
    if grid_dims is None:
        return

    nx, ny, nz = grid_dims
    try:
        with open(incar_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return

    new_lines = []
    for line in lines:
        if "=" not in line:
            new_lines.append(line)
            continue
        key = line.split("=", 1)[0].strip().upper()
        if key in {"NGXF", "NGYF", "NGZF"}:
            # drop any existing NGX* lines
            continue
        new_lines.append(line)

    new_lines.append(f"NGXF = {nx}\n")
    new_lines.append(f"NGYF = {ny}\n")
    new_lines.append(f"NGZF = {nz}\n")

    with open(incar_path, "w") as f:
        f.writelines(new_lines)


def count_scf_breakdown_from_oszicar(osz_path: str):
    """
    Count DAV, RMM, and any other iteration types in OSZICAR.

    Returns:
        dav (int)
        rmm (int)
        total (int)  # DAV + RMM + all other tags
        other_counts (dict[str, int])  # e.g. {"HMM": 3, "CG": 1}
    """
    dav = 0
    rmm = 0
    other_counts: dict[str, int] = {}

    try:
        with open(osz_path, "r") as f:
            for line in f:
                ls = line.lstrip()
                m = re.match(r"([A-Z]+):", ls)
                if not m:
                    continue
                tag = m.group(1)

                if tag == "DAV":
                    dav += 1
                elif tag == "RMM":
                    rmm += 1
                else:
                    other_counts[tag] = other_counts.get(tag, 0) + 1
    except FileNotFoundError:
        pass

    total = dav + rmm + sum(other_counts.values())
    return dav, rmm, total, other_counts


# -------------------------
# Core runner using MPStaticSet (approx MP settings for GNoME)
# -------------------------

def run_vasp_pymatgen(atoms,
                      workdir,
                      icharg,
                      chgcar_path=None,
                      vasp_cmd=("vasp_std",),
                      grid_dims=None):
    """
    VASP runner using pymatgen's MPStaticSet with approximate MP settings,
    similar to the ChargE3Net GNoME setup.

    Returns on success:
        (energy, dav_steps, rmm_steps, total_steps, wall_time)

    Returns on any VASP/OUTCAR failure:
        (None, dav_steps, rmm_steps, total_steps, wall_time)
    """

    os.makedirs(workdir, exist_ok=True)

    # ASE -> pymatgen Structure
    structure = AseAtomsAdaptor.get_structure(atoms)

    # INCAR overrides relative to MP defaults (ChargE3Net-style static settings)
    user_incar = {
        "ICHARG": icharg,  # distinguishes SAD vs CHGCAR vs ML init
        "ISTART": 0,       # always start from scratch, not from stray WAVECAR
        "LCHARG": True,    # write CHGCAR
        "LWAVE": False,    # do not write WAVECAR
    }

    # Let MPStaticSet build INCAR/KPOINTS/POTCAR using its MP config + PMG_VASP_PSP_DIR
    mp_set = MPStaticSet(
        structure,
        user_incar_settings=user_incar,
    )
    mp_set.write_input(workdir)

    # Patch LMAXMIX for DFT+U, ChargE3Net style
    incar_path = os.path.join(workdir, "INCAR")
    patch_lmaxmix_for_dftu(incar_path)

    # Optional CHGCAR init
    if chgcar_path is not None:
        dest = os.path.join(workdir, "CHGCAR")

        # If src and dest are the same file, do nothing
        try:
            if os.path.exists(dest) and os.path.samefile(chgcar_path, dest):
                pass
            else:
                if chgcar_path.endswith(".lz4"):
                    with lz4.frame.open(chgcar_path, "rb") as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                else:
                    shutil.copy(chgcar_path, dest)
        except FileNotFoundError:
            # os.path.samefile can raise if one path doesn't exist yet
            if chgcar_path.endswith(".lz4"):
                with lz4.frame.open(chgcar_path, "rb") as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            else:
                shutil.copy(chgcar_path, dest)

    # Run VASP
    start_time = time.time()
    try:
        subprocess.run(vasp_cmd, cwd=workdir, check=True)
    except subprocess.CalledProcessError as e:
        wall_time = time.time() - start_time
        print(f"❌ VASP run failed (return code {e.returncode}) in {workdir}")
        osz_path = os.path.join(workdir, "OSZICAR")
        dav_steps, rmm_steps, total_steps, other_counts = count_scf_breakdown_from_oszicar(osz_path)
        return None, dav_steps, rmm_steps, total_steps, other_counts, wall_time

    wall_time = time.time() - start_time

    # Parse SCF steps (DAV/RMM/total)
    osz_path = os.path.join(workdir, "OSZICAR")
    dav_steps, rmm_steps, total_steps, other_counts = count_scf_breakdown_from_oszicar(osz_path)

    # Parse final energy – be defensive because OUTCAR might be truncated
    outcar_path = os.path.join(workdir, "OUTCAR")
    if not os.path.isfile(outcar_path):
        print(f"❌ OUTCAR missing in {workdir}; treating as failed run.")
        return None, dav_steps, rmm_steps, total_steps, other_counts, wall_time

    try:
        outcar = Outcar(outcar_path)
        energy = outcar.final_energy
    except Exception as e:
        print(f"❌ Failed to parse OUTCAR in {workdir}: {e!r}")
        energy = None

    return energy, dav_steps, rmm_steps, total_steps, other_counts, wall_time


# -------------------------
# CHGCAR patching via pymatgen: replace total density but preserve magnetization
# -------------------------



def build_ml_initialized_chgcar(
    template_true_chgcar: str,
    ml_chgcar_path: str,
    out_path: str,
):
    """
    Build a CHGCAR for ML-init on GNoME:

      - Load *template_true_chgcar* (default run CHGCAR) via Chgcar.from_file.
      - Parse *ml_chgcar_path* (ChargE3Net CHGCAR) manually using the grid-dim line.
      - Replace the *total* charge density with the ML grid.
      - For ISPIN=2 (spin-polarized), preserve the *true* magnetization:
          n_up^ML = (n_ML + m_true) / 2
          n_dn^ML = (n_ML - m_true) / 2
      - Keep augmentation (data_aug) and all other metadata untouched.
    """
    # Load template (true) CHGCAR
    true_chg = Chgcar.from_file(template_true_chgcar)

    # Determine grid shape and Ngrid from template
    dim = _get_chgcar_dim(true_chg)  # (nx, ny, nz)
    n_needed = int(dim[0] * dim[1] * dim[2])

    # Read ML grid as flat array, then reshape to template shape
    ml_chg = Chgcar.from_file(ml_chgcar_path)
    ml_tot, ml_diff = _extract_total_and_diff(ml_chg)

    if ml_tot.shape != dim:
        raise RuntimeError(f"ML grid shape {ml_tot.shape} != template shape {dim}")

    ml_grid = ml_tot
    ml_grid = _rescale_ml_to_true_charge(ml_grid, true_chg)

    data_obj = true_chg.data

    if isinstance(data_obj, dict):
        # Case A: explicit spin channels
        if (Spin.up in data_obj) and (Spin.down in data_obj):
            n_up_true = np.asarray(data_obj[Spin.up])
            n_dn_true = np.asarray(data_obj[Spin.down])

            if n_up_true.shape != dim or n_dn_true.shape != dim:
                raise RuntimeError(
                    f"Spin channels have shapes {n_up_true.shape}, {n_dn_true.shape}, "
                    f"but ML grid shape is {dim}"
                )

            m_true = n_up_true - n_dn_true
            n_up_ml = 0.5 * (ml_grid + m_true)
            n_dn_ml = 0.5 * (ml_grid - m_true)

            true_chg.data = {Spin.up: n_up_ml, Spin.down: n_dn_ml}

        # Case B: pymatgen "total"/"diff" representation
        elif "total" in data_obj:
            new_data = dict(data_obj)
            new_data["total"] = ml_grid
            # keep new_data["diff"] untouched (true magnetization)
            true_chg.data = new_data

        # Case C: unknown dict layout → broadcast (last resort)
        else:
            new_data = {}
            for key, arr in data_obj.items():
                arr_np = np.asarray(arr)
                if arr_np.shape != dim:
                    raise RuntimeError(
                        f"Template CHGCAR channel {key} has shape {arr_np.shape}, "
                        f"but ML grid shape is {dim}"
                    )
                new_data[key] = ml_grid.copy()
            true_chg.data = new_data
    else:
        true_chg.data = ml_grid.copy()

    # Augmentation (true_chg.data_aug) and all metadata remain untouched
    true_chg.write_file(out_path)

def _extract_total_and_diff(chg: Chgcar):
    """
    Returns (total, diff_or_None) as 3D numpy arrays.

    Supports:
      - dict with Spin.up/Spin.down
      - dict with keys "total" and optional "diff"
      - single ndarray
    """
    data = chg.data

    if isinstance(data, dict):
        if (Spin.up in data) and (Spin.down in data):
            up = np.asarray(data[Spin.up])
            dn = np.asarray(data[Spin.down])
            total = up + dn
            diff = up - dn
            return total, diff

        if "total" in data:
            total = np.asarray(data["total"])
            diff = np.asarray(data["diff"]) if "diff" in data else None
            return total, diff

        # fallback: take first channel as total, no diff
        arr = np.asarray(next(iter(data.values())))
        return arr, None

    # non-spin: single 3D array
    arr = np.asarray(data)
    return arr, None


def nmae(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Normalized Mean Absolute Error:
        mean(|pred - true|) / (mean(|true|) + eps)
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    return float(np.sum(np.abs(pred - true)) / (np.sum(np.abs(true))))


def compute_nmae_between_chgcars(true_chgcar_path: str, pred_chgcar_path: str):
    """
    Compute NMAE for total density, and (if present) the diff/magnetization channel.
    Returns: (nmae_total, nmae_diff_or_None)
    """
    chg_true = Chgcar.from_file(true_chgcar_path)
    chg_pred = Chgcar.from_file(pred_chgcar_path)

    tot_true, diff_true = _extract_total_and_diff(chg_true)
    tot_pred, diff_pred = _extract_total_and_diff(chg_pred)

    if tot_true.shape != tot_pred.shape:
        raise RuntimeError(f"Total grid shape mismatch: {tot_true.shape} vs {tot_pred.shape}")

    n_total = nmae(tot_pred, tot_true)

    n_diff = None
    if (diff_true is not None) and (diff_pred is not None):
        if diff_true.shape != diff_pred.shape:
            raise RuntimeError(f"Diff grid shape mismatch: {diff_true.shape} vs {diff_pred.shape}")
        n_diff = nmae(diff_pred, diff_true)

    return n_total, n_diff
def _rescale_ml_to_true_charge(ml_grid, true_chg):
    """
    Rescale ml_grid so that its integrated charge matches that of true_chg.

    Handles:
      - true_chg.data is ndarray (non-spin)
      - true_chg.data is dict with Spin.up/Spin.down
      - true_chg.data is dict with "total" (and possibly "diff")
    """
    data = true_chg.data

    if isinstance(data, dict):
        # Case 1: explicit spin channels
        if (Spin.up in data) and (Spin.down in data):
            n_up = np.asarray(data[Spin.up])
            n_dn = np.asarray(data[Spin.down])
            true_tot = n_up + n_dn
        # Case 2: pymatgen total/diff representation
        elif "total" in data:
            true_tot = np.asarray(data["total"])
        else:
            # Fallback: take the first channel as "total"
            arr = np.asarray(next(iter(data.values())))
            true_tot = arr
    else:
        # Non-spin-polarized: single 3D array
        true_tot = np.asarray(data)

    dim = true_tot.shape
    assert ml_grid.shape == dim, f"ML grid shape {ml_grid.shape} != template shape {dim}"

    # Voxel volume = cell_volume / number_of_grid_points
    cell_vol = true_chg.structure.lattice.volume
    npoints = int(dim[0] * dim[1] * dim[2])
    dv = cell_vol / npoints

    q_true = true_tot.sum() * dv
    q_ml   = ml_grid.sum()   * dv

    if q_ml == 0:
        raise RuntimeError("ML grid has zero integrated charge, cannot rescale")

    scale = q_true / q_ml
    # Optional debug:
    # print(f"[rescale] q_true={q_true:.6f}, q_ml={q_ml:.6f}, scale={scale:.6f}")

    return ml_grid * scale
# -------------------------
# Main execution: ML-INIT on GNoME (ICHARG=1 from ChargE3Net-patched CHGCAR)
# -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run VASP SCF for one GNoME entry using MPStaticSet-style inputs, "
            "initialized from a CHGCAR where the total charge density has been "
            "replaced by a ChargE3Net prediction (ICHARG=1). "
            "Header, atom lines, magnetization, and PAW augmentation come from the "
            "ground-truth CHGCAR of the default/SAD run."
        )
    )

    parser.add_argument(
        "--csv", "-C",
        dest="CSV_PATH",
        required=True,
        help="GNoME task CSV with at least ['CHGCAR_PATH','performed_default','performed_true_init','performed_ml_init']",
    )
    parser.add_argument(
        "--default_results_csv", "-D",
        dest="DEFAULT_RESULTS_CSV",
        required=True,
        help="CSV with default SCF results including ['CHGCAR_PATH','scf_steps_total_default','time_default','is_magnetic']",
    )
    parser.add_argument(
        "--outdir", "-o",
        dest="BASE_DIR",
        required=True,
        help="Base directory under which per-structure ML-INIT folders will be created",
    )
    parser.add_argument(
        "--results_csv", "-R",
        dest="RESULTS_CSV",
        required=True,
        help="CSV file to append ML-init results to (e.g. CHARGE3NET_ML_INIT_GNOME.csv)",
    )
    parser.add_argument(
        "--sad_runs_dir",
        dest="SAD_RUNS_DIR",
        default="SAD_VASP_RUNS_GNOME",
        help=(
            "Directory containing prior SAD/default runs, organized as "
            "SAD_RUNS_DIR/<base>/run_default/CHGCAR. "
            "Defaults to 'SAD_VASP_RUNS_GNOME' relative to CWD."
        ),
    )
    parser.add_argument(
        "--ml_chgcar_root",
        dest="ML_CHGCAR_ROOT",
        required=True,
        help=(
            "Root directory containing ChargE3Net-predicted GNoME CHGCARs. "
            "This script expects files named <base>.chgcar or <base> inside this directory, "
            "where <base> is the basename of the original GNoME CHGCAR_PATH without .lz4."
        ),
    )
    parser.add_argument(
        "--vasp_cmd",
        nargs=argparse.REMAINDER,
        default=["vasp_std"],
        help="Command used to run VASP (e.g. mpirun -np 32 vasp_std). "
             "MUST be the last option on the command line.",
    )
    parser.add_argument(
        "--start_row",
        type=int,
        default=0,
        help="Inclusive start row index in the task CSV slice.",
    )
    parser.add_argument(
        "--end_row",
        type=int,
        default=None,
        help="Exclusive end row index in the task CSV slice. Defaults to len(df).",
    )

    args = parser.parse_args()

    # Load task CSV
    df = pd.read_csv(args.CSV_PATH, dtype={"CHGCAR_PATH": str})
    n_rows = len(df)

    # Load default-results CSV with is_magnetic, steps, time
    df_default = pd.read_csv(
        args.DEFAULT_RESULTS_CSV,
        dtype={"CHGCAR_PATH": str},
    )
    # Normalize CHGCAR paths for robust matching
    df_default["CHGCAR_PATH_norm"] = df_default["CHGCAR_PATH"].astype(str)

    # Build set of non-magnetic CHGCAR paths
    if "is_magnetic" not in df_default.columns:
        raise RuntimeError(
            f"'is_magnetic' column not found in DEFAULT_RESULTS_CSV={args.DEFAULT_RESULTS_CSV}"
        )

    nonmag_paths = set(
        df_default.loc[df_default["is_magnetic"] == False, "CHGCAR_PATH_norm"]
        .astype(str)
        .tolist()
    )

    # Ensure columns exist
    for col in ["performed_default", "performed_true_init", "performed_ml_init"]:
        if col not in df.columns:
            df[col] = False

    # Clamp slice
    start = max(args.start_row, 0)
    end = n_rows if args.end_row is None else min(args.end_row, n_rows)

    if start >= end:
        print(f"❌ Invalid slice [{start}:{end}) for CSV with {n_rows} rows.")
        raise SystemExit(1)

    print(f"Using slice [{start}:{end}) out of {n_rows}")

    df_slice = df.iloc[start:end]

    # --------- De-dup using existing ML-INIT results CSV ---------
    if os.path.exists(args.RESULTS_CSV):
        try:
            done_df = pd.read_csv(args.RESULTS_CSV, usecols=["CHGCAR_PATH"])
            done_paths = set(done_df["CHGCAR_PATH"].astype(str))
        except Exception as e:
            print(f"⚠️ Could not read results CSV {args.RESULTS_CSV}: {e!r}")
            done_paths = set()
    else:
        done_paths = set()

    if done_paths:
        mask_done = df_slice["CHGCAR_PATH"].astype(str).isin(done_paths)
        if mask_done.any():
            df.loc[df_slice.index[mask_done], "performed_ml_init"] = True
            df_slice = df.iloc[start:end]

    # Non-magnetic mask from default results
    chg_paths_slice = df_slice["CHGCAR_PATH"].astype(str)
    nonmag_mask = chg_paths_slice.isin(nonmag_paths)

    # Now pick pending rows:
    #  - require performed_default == True (so default run & true CHGCAR exist)
    #  - require performed_ml_init == False
    #  - not already in ML-INIT results CSV
    #  - non-magnetic according to DEFAULT_RESULTS_CSV
    pending_mask = (
        (~df_slice["CHGCAR_PATH"].astype(str).isin(done_paths)) &
        nonmag_mask
    )

    pending = df_slice[pending_mask]

    if pending.empty:
        print(f"No pending ML-INIT GNoME rows in slice [{start}:{end}) (after non-magnetic filter).")
        raise SystemExit(2)

    # idx is still the original CSV index
    idx = pending.index[0]
    chgcar_path = df.at[idx, "CHGCAR_PATH"]

    print(
        f"Selected row idx={idx} (slice [{start}:{end})), "
        f"CHGCAR={chgcar_path}"
    )

    # Derive base name, e.g. "74b63d06aa.chgcar" from ".../74b63d06aa.chgcar.lz4"
    base_name = os.path.splitext(os.path.basename(chgcar_path))[0]

    # Path to prior SAD/default run that holds the *ground-truth* CHGCAR
    sad_runs_dir = args.SAD_RUNS_DIR
    sad_run_base = os.path.join(sad_runs_dir, base_name)
    sad_run_default_dir = os.path.join(sad_run_base, "run_default")
    true_chgcar_path = os.path.join(sad_run_default_dir, "CHGCAR")

    if not os.path.isfile(true_chgcar_path):
        print(f"❌ Ground-truth CHGCAR not found at {true_chgcar_path}")
        # Mark as attempted (so we don't spin forever), but signal a per-structure failure.
        df.at[idx, "performed_ml_init"] = True
        df.to_csv(args.CSV_PATH, index=False)
        raise SystemExit(3)

    # Path to ChargE3Net-predicted CHGCAR for GNoME
    ml_chgcar_root = args.ML_CHGCAR_ROOT
    orig_base = os.path.basename(chgcar_path)  # "1b891ed189.chgcar.lz4"
    stem, ext = os.path.splitext(orig_base)  # ("1b891ed189.chgcar", ".lz4")
    # We try two possibilities:
    #   <root>/<base_name>.chgcar
    #   <root>/<base_name>
    if stem.endswith(".chgcar"):
        base_id = stem[:-7]  # "1b891ed189"
    else:
        base_id = stem
    ml_chgcar_root = args.ML_CHGCAR_ROOT

    # For your layout:
    #   <root>/<base_id>/CHGCAR
    # Optionally keep a couple of fallbacks if you want
    cand1 = os.path.join(ml_chgcar_root, base_id, "CHGCAR")  # main pattern
    cand2 = os.path.join(ml_chgcar_root, f"{base_id}.chgcar")  # optional alt
    cand3 = os.path.join(ml_chgcar_root, base_id)  # optional alt

    if os.path.isfile(cand1):
        ml_chgcar_path = cand1
    elif os.path.isfile(cand2):
        ml_chgcar_path = cand2
    elif os.path.isfile(cand3):
        ml_chgcar_path = cand3
    else:
        print(f"❌ ML-predicted CHGCAR not found at any of:")
        print(f"    {cand1}")
        print(f"    {cand2}")
        print(f"    {cand3}")
        df.at[idx, "performed_ml_init"] = True
        df.to_csv(args.CSV_PATH, index=False)
        raise SystemExit(3)

    # Mark as performed (bookkeeping) regardless of VASP outcome for this structure
    df.at[idx, "performed_ml_init"] = True
    df.to_csv(args.CSV_PATH, index=False)

    # Directory setup for the ML-INIT run:
    run_base = os.path.join(args.BASE_DIR, base_name)
    ml_init_dir = os.path.join(run_base, "run_ml_init")

    if os.path.exists(ml_init_dir):
        shutil.rmtree(ml_init_dir)
    os.makedirs(ml_init_dir, exist_ok=True)

    # Get CHGCAR grid dims from the *true* CHGCAR (for logging + Ngrid)
    grid_dims = get_chgcar_grid_dims(true_chgcar_path)

    # Build merged CHGCAR in ml_init_dir using true CHGCAR as template + ML grid
    merged_chgcar_path = os.path.join(ml_init_dir, "CHGCAR")
    try:
        build_ml_initialized_chgcar(
            template_true_chgcar=true_chgcar_path,
            ml_chgcar_path=ml_chgcar_path,
            out_path=merged_chgcar_path,
        )
    except Exception as e:
        print(f"❌ Failed to build ML-initialized CHGCAR for base={base_name}: {e!r}")
        # already marked performed_ml_init; treat as per-structure failure
        raise SystemExit(3)
    try:
        nmae_total, nmae_diff = compute_nmae_between_chgcars(true_chgcar_path, merged_chgcar_path)
        print(f"NMAE(total) ML vs TRUE: {nmae_total:.6e}  ({100 * nmae_total:.4f} %)")
        if nmae_diff is not None:
            print(f"NMAE(diff)  ML vs TRUE: {nmae_diff:.6e}  ({100 * nmae_diff:.4f} %)")
    except Exception as e:
        print(f"⚠️ NMAE computation failed: {e!r}")
        nmae_total, nmae_diff = None, None

    # Load structure from the original GNoME CHGCAR(/lz4) for MPStaticSet
    try:
        atoms = atoms_from_chgcar(chgcar_path)
    except FileNotFoundError:
        print(f"❌ GNoME CHGCAR not found at {chgcar_path}")
        raise SystemExit(3)

    # W&B init
    wandb.init(
        project="VASP_SCF_ChargE3Net_GNOME",
        name=f"ml_init_gnome_{base_name}",
        config={
            "csv_index": int(idx),
            "slice_start": int(start),
            "slice_end": int(end),
            "chgcar_path_gnome": chgcar_path,
            "workdir": ml_init_dir,
            "grid_dims": grid_dims,
            "sad_run_default_dir": sad_run_default_dir,
            "true_chgcar_path": true_chgcar_path,
            "ml_chgcar_path": ml_chgcar_path,
            "merged_chgcar_path": merged_chgcar_path,
            "icharg": 1,
        },
    )

    # Run ML-INIT SCF (ICHARG = 1, CHGCAR init from merged CHGCAR)
    energy, dav, rmm, total, other_counts, wall = run_vasp_pymatgen(
        atoms=atoms,
        workdir=ml_init_dir,
        icharg=1,
        chgcar_path=merged_chgcar_path,
        vasp_cmd=tuple(args.vasp_cmd),
        grid_dims=grid_dims,
    )

    other_total = sum(other_counts.values()) if other_counts else 0
    status = "ok" if energy is not None else "failed"

    # --- Look up baseline (DEFAULT / SAD) steps & time for this CHGCAR_PATH ---
    base_steps = base_time = None
    step_delta = time_delta = None
    step_reduction_pct = time_reduction_pct = None

    base_rows = df_default.loc[
        df_default["CHGCAR_PATH_norm"] == str(chgcar_path)
    ]
    if not base_rows.empty:
        base_row = base_rows.iloc[0]
        try:
            base_steps = float(base_row["scf_steps_total_default"])
            base_time = float(base_row["time_default"])

            step_delta = base_steps - total if base_steps is not None else None
            time_delta = base_time - wall if base_time is not None else None

            if base_steps and base_steps > 0:
                step_reduction_pct = 100.0 * step_delta / base_steps
            if base_time and base_time > 0:
                time_reduction_pct = 100.0 * time_delta / base_time
        except KeyError as e:
            print(f"⚠️ Baseline columns missing in DEFAULT_RESULTS_CSV: {e!r}")

    # Log to W&B
    wandb.log({
        "status_ml_init": status,
        "energy_ml_init": energy,
        "scf_steps_dav_ml_init": dav,
        "scf_steps_rmm_ml_init": rmm,
        "scf_steps_other_ml_init": other_total,
        "scf_steps_total_ml_init": total,
        "time_ml_init": wall,

        # Baseline (default/SAD) values
        "baseline_steps_default": base_steps,
        "baseline_time_default": base_time,

        # Absolute improvements vs default
        "delta_steps_vs_default": step_delta,
        "delta_time_vs_default": time_delta,

        # Percentage improvements vs default
        "pct_steps_reduction_vs_default": step_reduction_pct,
        "pct_time_reduction_vs_default": time_reduction_pct,
        "nmae_total_ml_vs_true": nmae_total,
        "nmae_total_ml_vs_true_pct": None if nmae_total is None else 100.0 * nmae_total,
        "nmae_diff_ml_vs_true": nmae_diff,
        "nmae_diff_ml_vs_true_pct": None if nmae_diff is None else 100.0 * nmae_diff,
    })
    wandb.finish()

    # Append to ML-INIT results CSV
    header = [
        "CHGCAR_PATH",
        "csv_index",
        "slice_start",
        "slice_end",
        "status_ml_init",
        "energy_ml_init",
        "scf_steps_dav_ml_init",
        "scf_steps_rmm_ml_init",
        "scf_steps_other_ml_init",
        "scf_steps_total_ml_init",
        "time_ml_init",
        "nmae_total_ml_vs_true",
        "nmae_total_ml_vs_true_pct",
        "nmae_diff_ml_vs_true",
        "nmae_diff_ml_vs_true_pct",
    ]

    row_out = [
        chgcar_path,
        int(idx),
        int(start),
        int(end),
        status,
        "" if energy is None else energy,
        dav,
        rmm,
        other_total,
        total,
        wall,
        "" if nmae_total is None else nmae_total,
        "" if nmae_total is None else 100.0 * nmae_total,
        "" if nmae_diff is None else nmae_diff,
        "" if nmae_diff is None else 100.0 * nmae_diff,
    ]

    write_header = not os.path.exists(args.RESULTS_CSV)
    with open(args.RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row_out)

    print("Done (GNoME ML-INIT with patched CHGCAR via pymatgen, preserving magnetization).")
    print(f"  CSV index: {idx}")
    print(f"  Slice:     [{start}:{end})")
    print(f"  CHGCAR (GNoME): {chgcar_path}")
    print(f"  True CHGCAR (SAD): {true_chgcar_path}")
    print(f"  ML CHGCAR: {ml_chgcar_path}")
    print(f"  Merged CHGCAR: {merged_chgcar_path}")
    print(f"  Grid dims (NGXF,NGYF,NGZF): {grid_dims}")
    print(f"  Energy (ml-init): {energy}")
    print(f"  DAV steps: {dav}")
    print(f"  RMM steps: {rmm}")
    print(f"  TOTAL SCF steps: {total}")
    print(f"  Wall time [s]: {wall}")
    print(f"Appended ML-INIT results to {args.RESULTS_CSV}")

    sys.exit(0)
