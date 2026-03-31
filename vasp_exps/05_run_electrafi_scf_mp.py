#!/usr/bin/env python3
import os
import re
import shutil
import time
import argparse
import csv
import subprocess
import sys

import pandas as pd
import lz4.frame
import wandb
import numpy as np

from pymatgen.io.vasp.inputs import Incar, Poscar, Kpoints, Potcar
from pymatgen.io.vasp.outputs import Oszicar, Outcar, Chgcar
from pymatgen.electronic_structure.core import Spin
from mp_api.client import MPRester

_float_re = re.compile(r'[-+]?\d*\.?\d+(?:[Ee][+-]?\d+)?')

# --------------------------------------------------------------------------------------
# Helpers: CHGCAR grid dims via pymatgen
# --------------------------------------------------------------------------------------

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
    """
    chg = Chgcar.from_file(chgcar_path)
    return _get_chgcar_dim(chg)


# --------------------------------------------------------------------------------------
# Parse SCF breakdown from OSZICAR
# --------------------------------------------------------------------------------------

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


# --------------------------------------------------------------------------------------
# Materials Project: Retrieve the *exact* CHGCAR-generating TaskDoc
# --------------------------------------------------------------------------------------

def get_exact_chg_taskdoc(mpid: str):
    """
    Retrieve the exact VASP TaskDoc that produced the MP CHGCAR.

    Uses:
        mpr.get_charge_density_from_material_id(mpid, inc_task_doc=True)
    """
    with MPRester() as mpr:
        result = mpr.get_charge_density_from_material_id(
            mpid,
            inc_task_doc=True
        )
        if result is None:
            raise RuntimeError(f"No charge density / task document for {mpid}")

        chgcar, taskdoc = result  # noqa: F841 (chgcar unused)
        return taskdoc


# --------------------------------------------------------------------------------------
# Write exact MP inputs (POSCAR, INCAR, KPOINTS, POTCAR)
# --------------------------------------------------------------------------------------

def write_mp_inputs_for_mpid(mpid: str, workdir: str):
    os.makedirs(workdir, exist_ok=True)

    # The exact task that produced this CHGCAR
    tdoc = get_exact_chg_taskdoc(mpid)

    # Handle dict vs document model
    vinput = tdoc["input"] if isinstance(tdoc, dict) else tdoc.input

    # POSCAR
    structure = vinput["structure"] if isinstance(vinput, dict) else vinput.structure
    Poscar(structure).write_file(os.path.join(workdir, "POSCAR"))

    # INCAR
    if isinstance(vinput, dict):
        incar_dict = vinput.get("parameters", {}) or {}
    else:
        incar_dict = vinput.parameters or {}
    Incar(incar_dict).write_file(os.path.join(workdir, "INCAR"))

    # KPOINTS (only write if we actually have them)
    if isinstance(vinput, dict):
        k_raw = vinput.get("kpoints", None)
    else:
        k_raw = getattr(vinput, "kpoints", None)

    kpoints = None
    if isinstance(k_raw, Kpoints):
        kpoints = k_raw
    elif isinstance(k_raw, dict):
        kpoints = Kpoints.from_dict(k_raw)
    # else: leave kpoints=None and DO NOT write KPOINTS

    if kpoints is not None:
        kpoints.write_file(os.path.join(workdir, "KPOINTS"))

    # POTCAR from potcar_spec if present
    if isinstance(vinput, dict):
        potcar_spec = vinput.get("potcar_spec", None)
    else:
        potcar_spec = getattr(vinput, "potcar_spec", None)

    if potcar_spec:
        symbols: list[str] = []
        for ps in potcar_spec:
            if hasattr(ps, "symbol") and ps.symbol is not None:
                symbols.append(str(ps.symbol))
            elif isinstance(ps, dict) and "symbol" in ps:
                symbols.append(str(ps["symbol"]))
            elif hasattr(ps, "titel"):
                titel = str(ps.titel)
                parts = titel.split()
                if len(parts) > 1:
                    symbols.append(parts[1])
                else:
                    raise RuntimeError(f"Unexpected POTCAR titel format: {titel}")
            elif isinstance(ps, dict) and "titel" in ps:
                titel = str(ps["titel"])
                parts = titel.split()
                if len(parts) > 1:
                    symbols.append(parts[1])
                else:
                    raise RuntimeError(f"Unexpected POTCAR titel format (dict): {titel}")
            else:
                raise RuntimeError(f"Unrecognized potcar_spec entry type: {ps!r}")

        potcar = Potcar(symbols)
        potcar.write_file(os.path.join(workdir, "POTCAR"))
    else:
        raise RuntimeError(f"POTCAR metadata missing for {mpid} (no potcar_spec)")


# --------------------------------------------------------------------------------------
# Convenience util: derive MP ID from CSV row or path
# --------------------------------------------------------------------------------------

def derive_mpid(row: pd.Series, chgcar_path: str) -> str:
    for col in ("MP_ID", "mp_id", "material_id", "materialId"):
        if col in row.index:
            val = str(row[col]).strip()
            if val.startswith("mp-"):
                return val.lower()

    m = re.search(r"(mp-\d+)", chgcar_path, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()

    raise RuntimeError(f"Could not infer MP ID for CHGCAR_PATH={chgcar_path}")

def find_ml_chgcar_for_mpid(ml_root: str, mpid: str) -> str | None:
    """
    Recursively search ml_root for files ending with 'CHGCAR' whose filename prefix
    (before first underscore) matches mpid (case-insensitive).

    Example match:
      mpid='mp-1015814' matches 'mp-1015814_AlLiMg2_..._test_1270.CHGCAR'
    """
    mpid_l = mpid.lower()

    for dirpath, _, filenames in os.walk(ml_root):
        for fn in filenames:
            if not fn.upper().endswith("CHGCAR"):
                continue
            prefix = fn.split("_", 1)[0].lower()
            if prefix == mpid_l:
                return os.path.join(dirpath, fn)

    return None

# --------------------------------------------------------------------------------------
# Run VASP using MP inputs, with optional CHGCAR init
# --------------------------------------------------------------------------------------

def run_vasp_mp(
    mpid: str,
    workdir: str,
    icharg: int,
    chgcar_path: str | None,
    vasp_cmd,
    grid_dims,
):
    os.makedirs(workdir, exist_ok=True)

    # Write exact MP input files (POSCAR/INCAR/KPOINTS/POTCAR)
    write_mp_inputs_for_mpid(mpid, workdir)

    # Modify INCAR minimally
    incar_path = os.path.join(workdir, "INCAR")
    incar = Incar.from_file(incar_path)

    # Remove parallelization hints
    for bad in ["NPAR", "NCORE", "KPAR", "NSIM"]:
        if bad in incar:
            incar.pop(bad)

    # Only touch what we need for the SCF experiment
    incar["ICHARG"] = icharg  # 2 = SAD, 1 = CHGCAR init
    incar["ISTART"] = 0       # always fresh SCF (no WAVECAR reuse)
    incar["LCHARG"] = True    # always write CHGCAR
    incar.write_file(incar_path)

    # If CHGCAR init is requested, copy it in; otherwise assume it already exists
    if chgcar_path is not None:
        dest = os.path.join(workdir, "CHGCAR")
        if chgcar_path.endswith(".lz4"):
            with lz4.frame.open(chgcar_path, "rb") as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.copy(chgcar_path, dest)

    # Run VASP
    start = time.time()
    try:
        subprocess.run(vasp_cmd, cwd=workdir, check=True)
    except subprocess.CalledProcessError:
        wall = time.time() - start
        osz_path = os.path.join(workdir, "OSZICAR")
        dav, rmm, total, other_counts = count_scf_breakdown_from_oszicar(osz_path)
        return None, dav, rmm, total, other_counts, wall

    wall = time.time() - start

    # Parse SCF steps
    osz_path = os.path.join(workdir, "OSZICAR")
    dav, rmm, total, other_counts = count_scf_breakdown_from_oszicar(osz_path)

    # Parse energy
    outcar_path = os.path.join(workdir, "OUTCAR")
    if not os.path.isfile(outcar_path):
        return None, dav, rmm, total, other_counts, wall

    try:
        outcar = Outcar(outcar_path)
        energy = outcar.final_energy
    except Exception:
        energy = None

    return energy, dav, rmm, total, other_counts, wall


# --------------------------------------------------------------------------------------
# Rescale ML grid to match total charge of true CHGCAR
# --------------------------------------------------------------------------------------

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



# --------------------------------------------------------------------------------------
# CHGCAR patching via pymatgen: replace total density but preserve magnetization
# --------------------------------------------------------------------------------------

def _read_ml_grid_flat(ml_chgcar_path: str, n_needed: int):
    """
    Read n_needed floats from a ELECTRAFI-style CHGCAR:

      - Look for a line with three integers nx ny nz giving the grid dims
      - Require nx * ny * nz == n_needed
      - Then read floats from the subsequent lines

    Returns:
        1D numpy array of length n_needed
    """
    with open(ml_chgcar_path, "r") as f:
        lines = f.readlines()

    n_line_idx = None
    n_grid = None

    for i, line in enumerate(lines):
        toks = line.split()
        if len(toks) == 3 and all(t.isdigit() for t in toks):
            nx, ny, nz = map(int, toks)
            cand_n_grid = nx * ny * nz
            if cand_n_grid == n_needed:
                n_line_idx = i
                n_grid = cand_n_grid
                break

    if n_line_idx is None or n_grid is None:
        raise RuntimeError(
            f"Could not find grid-dim line with three integers whose product is Ngrid={n_needed} in {ml_chgcar_path}"
        )

    rest = "".join(lines[n_line_idx + 1:])

    vals = []
    for m in _float_re.finditer(rest):
        vals.append(float(m.group(0)))
        if len(vals) == n_grid:
            break

    if len(vals) < n_grid:
        raise RuntimeError(
            f"Only found {len(vals)} density values (expected {n_grid}) "
            f"in {ml_chgcar_path}"
        )

    return np.array(vals, dtype=float)


def build_ml_initialized_chgcar(
    template_true_chgcar: str,
    ml_chgcar_path: str,
    out_path: str,
):
    """
    Build a CHGCAR for ML-init:

      - Load *template_true_chgcar* (default run CHGCAR) via Chgcar.from_file.
      - Parse *ml_chgcar_path* (ELECTRAFI-style CHGCAR) manually using the grid-dim line.
      - Rescale the ML grid so its integrated charge matches the true CHGCAR.
      - Replace the *total* charge density with the rescaled ML grid.

      Spin handling:
        * If data is Spin.up/Spin.down: we assume non-magnetic and split ML total
          equally between the two channels:
              n_up^ML = n_dn^ML = 0.5 * n_ML
        * If data has "total"/"diff": replace only "total" with ML grid, keep "diff".
        * If data is a single 3D array: overwrite directly.

      Augmentation (data_aug) and all metadata remain untouched.
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
        raise RuntimeError(f"ML CHGCAR grid shape {ml_tot.shape} != template shape {dim}")

    ml_grid = ml_tot

    # --- Normalize electron count to match true CHGCAR ---
    ml_grid = _rescale_ml_to_true_charge(ml_grid, true_chg)

    data_obj = true_chg.data

    if isinstance(data_obj, dict):
        # Case A: explicit spin channels (ISPIN=2)
        if (Spin.up in data_obj) and (Spin.down in data_obj):
            n_up_true = np.asarray(data_obj[Spin.up])
            n_dn_true = np.asarray(data_obj[Spin.down])

            if n_up_true.shape != dim or n_dn_true.shape != dim:
                raise RuntimeError(
                    f"Spin channels have shapes {n_up_true.shape}, {n_dn_true.shape}, "
                    f"but ML grid shape is {dim}"
                )

            # We are working on a non-magnetic subset. Instead of preserving m_true,
            # just split the ML total density equally between up and down.
            m_true = n_up_true - n_dn_true
            n_up_ml = 0.5 * (ml_grid + m_true)
            n_dn_ml = 0.5 * (ml_grid - m_true)
            true_chg.data = {Spin.up: n_up_ml, Spin.down: n_dn_ml}

        # Case B: pymatgen "total"/"diff" representation
        elif "total" in data_obj:
            total_true = np.asarray(data_obj["total"])
            if total_true.shape != dim:
                raise RuntimeError(
                    f"'total' channel has shape {total_true.shape}, "
                    f"but ML grid shape is {dim}"
                )

            new_data = dict(data_obj)
            # Replace only the total density; keep diff (magnetization) as is
            new_data["total"] = ml_grid
            true_chg.data = new_data

        else:
            # Fallback: unknown dict structure, just broadcast ML grid to all channels
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
        # Non-spin-polarized: single 3D array
        arr_np = np.asarray(data_obj)
        if arr_np.shape != dim:
            raise RuntimeError(
                f"Template CHGCAR data shape {arr_np.shape} "
                f"mismatch with ML grid shape {dim}"
            )
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
    return float(np.sum(np.abs(pred - true)) / (np.sum(true)))


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



# --------------------------------------------------------------------------------------
# Main execution: ML-INIT (ICHARG=1 from ELECTRAFI-patched CHGCAR)
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run VASP SCF for one MP entry using exact MP inputs, "
            "initialized from a CHGCAR where the valence grid has been "
            "replaced by a ELECTRAFI prediction (ICHARG=1), and rescaled "
            "so that the integrated charge matches the ground-truth CHGCAR. "
            "Header, atom lines, and PAW augmentation come from the prior "
            "ground-truth CHGCAR of the default/SAD run."
        )
    )

    parser.add_argument(
        "--csv", "-C",
        dest="CSV_PATH",
        required=True,
        help="Task CSV with at least ['CHGCAR_PATH','MP_ID','performed_default']"
    )
    parser.add_argument(
        "--default_results_csv", "-D",
        dest="DEFAULT_RESULTS_CSV",
        required=True,
        help="CSV with default SCF results including 'MP_ID' and 'is_magnetic'. "
             "Only rows with is_magnetic == False will be run."
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
        help="CSV file to append ML-init results to (e.g. MP_ML_INIT.csv)",
    )
    parser.add_argument(
        "--sad_runs_dir",
        dest="SAD_RUNS_DIR",
        default="SAD_VASP_RUNS_MP",
        help=(
            "Directory containing prior SAD/default runs, organized as "
            "SAD_RUNS_DIR/<base>/run_default/CHGCAR. "
            "Defaults to 'SAD_VASP_RUNS_MP' relative to CWD."
        ),
    )
    parser.add_argument(
        "--ml_chgcar_root",
        dest="ML_CHGCAR_ROOT",
        required=True,
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
        help="Inclusive start row index in the task CSV slice."
    )
    parser.add_argument(
        "--end_row",
        type=int,
        default=None,
        help="Exclusive end row index in the task CSV slice. Defaults to len(df)."
    )

    args = parser.parse_args()

    # Load task CSV
    df = pd.read_csv(args.CSV_PATH, dtype={"CHGCAR_PATH": str, "MP_ID": str})
    n_rows = len(df)

    # Load default-results CSV with is_magnetic
    df_default = pd.read_csv(
        args.DEFAULT_RESULTS_CSV,
        dtype={"CHGCAR_PATH": str, "MP_ID": str}
    )
    # Convenience: lowercase MP IDs for robust matching
    df_default["MP_ID_lower"] = df_default["MP_ID"].astype(str).str.lower()

    # Build set of non-magnetic MP IDs
    nonmag_ids = set(
        df_default.loc[df_default["is_magnetic"] == False, "MP_ID"]
        .astype(str)
        .str.lower()
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
    mp_ids_slice = df_slice["MP_ID"].astype(str).str.lower()
    nonmag_mask = mp_ids_slice.isin(nonmag_ids)

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
        print(f"No pending ML-INIT rows in slice [{start}:{end}) (after non-magnetic filter).")
        raise SystemExit(2)

    # idx is still the original CSV index
    idx = pending.index[0]
    chgcar_path = df.at[idx, "CHGCAR_PATH"]
    row = df.loc[idx]

    mpid = derive_mpid(row, chgcar_path)

    print(f"Selected idx={idx}, CHGCAR={chgcar_path}, MPID={mpid}")

    # Derive base name as before (e.g. 'mp-1218746.chgcar')
    base = os.path.splitext(os.path.basename(chgcar_path))[0]

    # Path to prior SAD/default run that holds the *ground-truth* CHGCAR
    sad_runs_dir = args.SAD_RUNS_DIR
    sad_run_base = os.path.join(sad_runs_dir, base)
    sad_run_default_dir = os.path.join(sad_run_base, "run_default")
    true_chgcar_path = os.path.join(sad_run_default_dir, "CHGCAR")

    if not os.path.isfile(true_chgcar_path):
        print(f"❌ Ground-truth CHGCAR not found at {true_chgcar_path}")
        df.at[idx, "performed_ml_init"] = True
        df.to_csv(args.CSV_PATH, index=False)
        raise SystemExit(3)

    # Path to ELECTRAFI-predicted CHGCAR (new naming/layout)
    ml_chgcar_root = args.ML_CHGCAR_ROOT
    ml_chgcar_path = find_ml_chgcar_for_mpid(ml_chgcar_root, mpid)

    if (ml_chgcar_path is None) or (not os.path.isfile(ml_chgcar_path)):
        print(f"❌ ML-predicted CHGCAR not found under {ml_chgcar_root} for MPID={mpid}")
        df.at[idx, "performed_ml_init"] = True
        df.to_csv(args.CSV_PATH, index=False)
        raise SystemExit(3)

    # Mark as performed (bookkeeping) regardless of VASP outcome for this structure
    df.at[idx, "performed_ml_init"] = True
    df.to_csv(args.CSV_PATH, index=False)

    # Directory setup for the ML-INIT run:
    run_base = os.path.join(args.BASE_DIR, base)
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
        print(f"❌ Failed to build ML-initialized CHGCAR for {mpid}: {e!r}")
        raise SystemExit(3)
    # Compute NMAE between true CHGCAR and the merged (ML-grid) CHGCAR
    try:
        nmae_total, nmae_diff = compute_nmae_between_chgcars(true_chgcar_path, merged_chgcar_path)
        print(f"NMAE(total) ML vs TRUE: {nmae_total:.6e}  ({100*nmae_total:.4f} %)")
        if nmae_diff is not None:
            print(f"NMAE(diff)  ML vs TRUE: {nmae_diff:.6e}  ({100*nmae_diff:.4f} %)")
    except Exception as e:
        print(f"⚠️ NMAE computation failed: {e!r}")
        nmae_total, nmae_diff = None, None

    # W&B init
    wandb.init(
        project="VASP_SCF_ELECTRAFI_MP",
        name=f"ml_init_{base}",
        config={
            "csv_index": int(idx),
            "chgcar_path_mp": chgcar_path,
            "mpid": mpid,
            "workdir": ml_init_dir,
            "grid_dims": grid_dims,
            "slice_start": int(start),
            "slice_end": int(end),
            "sad_run_default_dir": sad_run_default_dir,
            "true_chgcar_path": true_chgcar_path,
            "ml_chgcar_path": ml_chgcar_path,
            "merged_chgcar_path": merged_chgcar_path,
            "icharg": 1,
        },
    )

    # Run VASP (ICHARG = 1 = read CHGCAR and start SCF from it)
    # Note: we pass chgcar_path=None because CHGCAR is already written in ml_init_dir
    energy, dav, rmm, total, other_counts, wall = run_vasp_mp(
        mpid=mpid,
        workdir=ml_init_dir,
        icharg=1,
        chgcar_path=None,
        vasp_cmd=tuple(args.vasp_cmd),
        grid_dims=grid_dims,
    )

    other_total = sum(other_counts.values()) if other_counts else 0

    status = "ok" if energy is not None else "failed"

    # --- Look up baseline (DEFAULT / SAD) steps & time for this MPID ---
    base_steps = base_time = None
    step_delta = time_delta = None
    step_reduction_pct = time_reduction_pct = None

    base_rows = df_default.loc[df_default["MP_ID_lower"] == mpid.lower()]
    if not base_rows.empty:
        base_row = base_rows.iloc[0]
        try:
            base_steps = float(base_row["scf_steps_total_default"])
            base_time = float(base_row["time_default"])

            step_delta = base_steps - total
            time_delta = base_time - wall

            if base_steps > 0:
                step_reduction_pct = 100.0 * step_delta / base_steps
            if base_time > 0:
                time_reduction_pct = 100.0 * time_delta / base_time
        except KeyError as e:
            print(f"⚠️ Baseline columns missing in DEFAULT_RESULTS_CSV: {e!r}")

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

        # Absolute improvements vs default (positive = better than default)
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
        "MP_ID",
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
        mpid,
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

    print("Done (ML-INIT with patched, charge-normalized CHGCAR via pymatgen, preserving magnetization).")
    print(f"Energy (ml-init): {energy}")
    print(f"DAV: {dav}, RMM: {rmm}, OTHER: {other_total}, TOTAL: {total}")
    print(f"Time: {wall:.2f}s")
