#!/usr/bin/env python3
import os
import re
import shutil
import time
import argparse
import csv
import subprocess
import sys
import tempfile

import pandas as pd
import lz4.frame
import wandb

from pymatgen.io.vasp.inputs import Incar, Poscar, Kpoints, Potcar
from pymatgen.io.vasp.outputs import Oszicar, Outcar
from mp_api.client import MPRester


# --------------------------------------------------------------------------------------
# Helpers: CHGCAR grid (we still log this, but do NOT use it to override INCAR anymore)
# --------------------------------------------------------------------------------------

def get_chgcar_grid_dims(chgcar_path):
    """
    Parse (NGXF, NGYF, NGZF) from a CHGCAR or CHGCAR.lz4.

    These grid dims are logged to W&B and CSV, but no longer inserted into INCAR.
    """
    path = chgcar_path
    tmp_created = False

    if chgcar_path.endswith(".lz4"):
        tmpfd, tmppath = tempfile.mkstemp(prefix="tmp_chgcar_grid_")
        os.close(tmpfd)
        with lz4.frame.open(chgcar_path, "rb") as src, open(tmppath, "wb") as dst:
            shutil.copyfileobj(src, dst)
        path = tmppath
        tmp_created = True

    nx = ny = nz = None
    try:
        with open(path, "r") as f:
            # Skip header + atom counts
            for _ in range(7):
                try:
                    next(f)
                except StopIteration:
                    break

            # First line with exactly 3 integers
            for line in f:
                toks = line.split()
                if len(toks) == 3 and all(t.isdigit() for t in toks):
                    nx, ny, nz = map(int, toks)
                    break
    finally:
        if tmp_created:
            os.remove(path)

    if nx is None:
        raise RuntimeError(f"Could not find NGXF/NGYF/NGZF in {chgcar_path}")

    return nx, ny, nz


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
                # Match leading ALLCAPS tag with colon, e.g. "DAV:", "RMM:", "HMM:", "CG:"
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
    Retrieve the exact VASP TaskDoc that produced the CHGCAR you downloaded.

    Uses:
        mpr.get_charge_density_from_material_id(mpid, inc_task_doc=True)

    which returns:
        (Chgcar, TaskDoc | dict)

    We ignore the Chgcar here and just return the TaskDoc.
    """
    with MPRester() as mpr:
        result = mpr.get_charge_density_from_material_id(
            mpid,
            inc_task_doc=True
        )
        if result is None:
            raise RuntimeError(f"No charge density / task document for {mpid}")

        # Unpack (chgcar, taskdoc)
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
    # else: leave kpoints=None and DO NOT write a KPOINTS file.
    # This lets VASP use KSPACING/KGAMMA from INCAR as in MP.

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
            # Try the most explicit attribute first
            if hasattr(ps, "symbol") and ps.symbol is not None:
                symbols.append(str(ps.symbol))
            # Fallback: dict-like
            elif isinstance(ps, dict) and "symbol" in ps:
                symbols.append(str(ps["symbol"]))
            # Fallback: parse from titel (e.g. "PAW_PBE Ti_pv 06Sep2000")
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

    # If CHGCAR init is requested
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
# Main execution: TRUE-DENSITY INIT (ICHARG=1 from ground-truth CHGCAR)
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VASP SCF for one MP entry using exact MP inputs, "
                    "initialized from ground-truth CHGCAR of a prior converged run."
    )

    parser.add_argument(
        "--csv", "-C",
        dest="CSV_PATH",
        required=True,
        help="Task CSV with at least ['CHGCAR_PATH','MP_ID','performed_default']"
    )
    parser.add_argument(
        "--outdir", "-o",
        dest="BASE_DIR",
        required=True,
        help="Base directory under which per-structure TRUE-INIT folders will be created",
    )
    parser.add_argument(
        "--results_csv", "-R",
        dest="RESULTS_CSV",
        required=True,
        help="CSV file to append true-init results to (e.g. MP_TRUE_INIT.csv)",
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

    # Ensure columns exist (we do NOT rely on performed_true_init for de-dup)
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

    # --------- De-dup using existing TRUE-INIT results CSV ---------
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
        # Mark these as performed_true_init for bookkeeping only; we don't *use* it for de-dup.
        mask_done = df_slice["CHGCAR_PATH"].astype(str).isin(done_paths)
        if mask_done.any():
            df.loc[df_slice.index[mask_done], "performed_true_init"] = True
            df_slice = df.iloc[start:end]

    # Now pick pending rows:
    #  - must have performed_default == True (so the ground-truth CHGCAR exists)
    #  - must not already appear in the TRUE-INIT results CSV
    if done_paths:
        pending_mask = (
            (~df_slice["CHGCAR_PATH"].astype(str).isin(done_paths))
        )
    else:
        pending_mask = (df_slice["performed_default"] == True)

    pending = df_slice[pending_mask]

    if pending.empty:
        print(f"No pending TRUE-INIT rows in slice [{start}:{end}).")
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
        # Mark as attempted (so we don't spin forever), but signal a per-structure failure.
        df.at[idx, "performed_true_init"] = True
        df.to_csv(args.CSV_PATH, index=False)
        raise SystemExit(3)

    # Mark as performed (bookkeeping)
    df.at[idx, "performed_true_init"] = True
    df.to_csv(args.CSV_PATH, index=False)

    # Directory setup for the TRUE-INIT run:
    # mirror the SAD_VASP_RUNS_MP structure but under BASE_DIR
    run_base = os.path.join(args.BASE_DIR, base)
    true_init_dir = os.path.join(run_base, "run_true_init")

    if os.path.exists(true_init_dir):
        shutil.rmtree(true_init_dir)
    os.makedirs(true_init_dir, exist_ok=True)

    # Get CHGCAR grid dims from the *original* MP CHGCAR (for logging only)
    grid_dims = get_chgcar_grid_dims(chgcar_path)

    # W&B init
    wandb.init(
        project="VASP_SCF_TrueInit_MP",
        name=f"true_init_{base}",
        config={
            "csv_index": int(idx),
            "chgcar_path_mp": chgcar_path,
            "mpid": mpid,
            "workdir": true_init_dir,
            "grid_dims": grid_dims,
            "slice_start": int(start),
            "slice_end": int(end),
            "sad_run_default_dir": sad_run_default_dir,
            "true_chgcar_path": true_chgcar_path,
            "icharg": 1,
        },
    )

    # Run VASP (ICHARG = 1 = read CHGCAR and start SCF from it)
    energy, dav, rmm, total, other_counts, wall = run_vasp_mp(
        mpid=mpid,
        workdir=true_init_dir,
        icharg=1,
        chgcar_path=true_chgcar_path,
        vasp_cmd=tuple(args.vasp_cmd),
        grid_dims=grid_dims,
    )

    other_total = sum(other_counts.values()) if other_counts else 0

    # Record results
    status = "ok" if energy is not None else "failed"

    wandb.log({
        "status_true_init": status,
        "energy_true_init": energy,
        "scf_steps_dav_true_init": dav,
        "scf_steps_rmm_true_init": rmm,
        "scf_steps_other_true_init": other_total,
        "scf_steps_total_true_init": total,
        "time_true_init": wall,
    })
    wandb.finish()

    # Append to TRUE-INIT results CSV
    header = [
        "CHGCAR_PATH",
        "MP_ID",
        "csv_index",
        "slice_start",
        "slice_end",
        "status_true_init",
        "energy_true_init",
        "scf_steps_dav_true_init",
        "scf_steps_rmm_true_init",
        "scf_steps_other_true_init",
        "scf_steps_total_true_init",
        "time_true_init",
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
    ]

    write_header = not os.path.exists(args.RESULTS_CSV)
    with open(args.RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row_out)

    print("Done (TRUE-INIT).")
    print(f"Energy (true-init): {energy}")
    print(f"DAV: {dav}, RMM: {rmm}, OTHER: {other_total}, TOTAL: {total}")
    print(f"Time: {wall:.2f}s")
