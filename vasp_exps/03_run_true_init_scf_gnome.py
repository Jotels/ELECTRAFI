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
from pymatgen.io.vasp.outputs import Oszicar, Outcar


# -------------------------
# Helpers
# -------------------------

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


def get_chgcar_grid_dims(chgcar_path):
    """
    Parse (NGXF, NGYF, NGZF) from a CHGCAR or CHGCAR.lz4.

    The CHGCAR format has the atom counts (e.g., '1 1 1') on line 7
    and the FFT grid (NGXF/NGYF/NGZF) much later after the coordinate section.
    We must skip the atom count line.

    Returns:
        (nx, ny, nz) as ints
    """
    path = chgcar_path
    tmp_created = False

    if chgcar_path.endswith(".lz4"):
        # Decompress lz4 file to a temporary location for line-by-line reading
        tmpfd, tmppath = tempfile.mkstemp(prefix="tmp_chgcar_grid_")
        os.close(tmpfd)
        with lz4.frame.open(chgcar_path, "rb") as src, open(tmppath, "wb") as dst:
            shutil.copyfileobj(src, dst)
        path = tmppath
        tmp_created = True

    nx = ny = nz = None
    try:
        with open(path, "r") as f:
            # Skip first 7 lines (0..6), which include atom counts
            for _ in range(7):
                try:
                    next(f)
                except StopIteration:
                    break

            # Now look for the first line with exactly 3 integers
            for line in f:
                toks = line.split()
                if len(toks) == 3 and all(t.isdigit() for t in toks):
                    nx, ny, nz = map(int, toks)
                    break
    finally:
        if tmp_created:
            os.remove(path)

    if nx is None:
        raise RuntimeError(
            f"Could not find NGXF/NGYF/NGZF line in {chgcar_path} "
            f"(search failed after line 7 skip)"
        )

    return nx, ny, nz


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
    Count DAV, RMM, and total electronic iterations in OSZICAR.
    If OSZICAR is missing or truncated, just return zeros.
    """
    dav = 0
    rmm = 0
    try:
        with open(osz_path, "r") as f:
            for line in f:
                ls = line.lstrip()
                if ls.startswith("DAV:"):
                    dav += 1
                elif ls.startswith("RMM:"):
                    rmm += 1
    except FileNotFoundError:
        pass
    total = dav + rmm
    return dav, rmm, total


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

    # Optional CHGCAR init (used here for TRUE-INIT runs)
    if chgcar_path is not None:
        dest = os.path.join(workdir, "CHGCAR")
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
        dav_steps, rmm_steps, total_steps = count_scf_breakdown_from_oszicar(osz_path)
        return None, dav_steps, rmm_steps, total_steps, wall_time

    wall_time = time.time() - start_time

    # Parse SCF steps (DAV/RMM/total)
    osz_path = os.path.join(workdir, "OSZICAR")
    dav_steps, rmm_steps, total_steps = count_scf_breakdown_from_oszicar(osz_path)

    # Parse final energy – be defensive because OUTCAR might be truncated
    outcar_path = os.path.join(workdir, "OUTCAR")
    if not os.path.isfile(outcar_path):
        print(f"❌ OUTCAR missing in {workdir}; treating as failed run.")
        return None, dav_steps, rmm_steps, total_steps, wall_time

    try:
        outcar = Outcar(outcar_path)
        energy = outcar.final_energy
    except Exception as e:
        print(f"❌ Failed to parse OUTCAR in {workdir}: {e!r}")
        energy = None

    return energy, dav_steps, rmm_steps, total_steps, wall_time


# -------------------------
# Main: TRUE-INIT GNOME runs
# -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run true-density-init VASP SCF for one CHGCAR from a GNoME-style task CSV, "
            "using MPStaticSet with approximate MP settings and initializing from the "
            "ground-truth CHGCAR of a prior converged run."
        )
    )
    parser.add_argument(
        "--csv", "-C",
        dest="CSV_PATH",
        required=True,
        help="CSV with columns ['CHGCAR_PATH','performed_default','performed_true_init','performed_ml_init']",
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
        help="CSV file to append TRUE-INIT results to (e.g. GNOME_TRUE_INIT.csv)",
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
    df = pd.read_csv(args.CSV_PATH, dtype={"CHGCAR_PATH": str})
    n_rows = len(df)

    # Ensure columns exist (backwards safety)
    for col in ["performed_default", "performed_true_init", "performed_ml_init"]:
        if col not in df.columns:
            df[col] = False

    # Clamp slice
    start = max(args.start_row, 0)
    end = n_rows if args.end_row is None else min(args.end_row, n_rows)

    if start >= end:
        print(f"❌ Invalid slice [{start}:{end}) for CSV with {n_rows} rows.")
        raise SystemExit(1)

    print(f"Using slice [{start}:{end}) out of {n_rows} rows")

    # Restrict to our slice **before** any filtering
    df_slice = df.iloc[start:end]

    # --------- De-dup using existing TRUE-INIT results CSV ---------
    if os.path.exists(args.RESULTS_CSV):
        try:
            res_df = pd.read_csv(args.RESULTS_CSV, usecols=["CHGCAR_PATH"])
            done_paths = set(res_df["CHGCAR_PATH"].astype(str))
        except Exception as e:
            print(f"⚠️ Could not read results CSV {args.RESULTS_CSV}: {e!r}")
            done_paths = set()
    else:
        done_paths = set()

    if done_paths:
        # Mark any rows in the slice whose CHGCAR_PATH is already in TRUE-INIT results
        mask_done = df_slice["CHGCAR_PATH"].astype(str).isin(done_paths)
        if mask_done.any():
            done_indices = df_slice.index[mask_done]
            print(f"Marking {len(done_indices)} rows as performed_true_init=True "
                  f"because they already appear in {args.RESULTS_CSV}.")
            df.loc[done_indices, "performed_true_init"] = True
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
        print(f"✅ No pending TRUE-INIT runs in slice [{start}:{end}).")
        # exit code 2: slice exhausted
        raise SystemExit(2)

    # idx is still the original CSV index (not 0..len(slice))
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
        df.at[idx, "performed_true_init"] = True
        df.to_csv(args.CSV_PATH, index=False)
        raise SystemExit(3)

    # Mark this row as performed_true_init, so we don't re-run it
    df.at[idx, "performed_true_init"] = True
    df.to_csv(args.CSV_PATH, index=False)

    # Directory setup for the TRUE-INIT run:
    # mirror the SAD_VASP_RUNS_GNOME structure but under BASE_DIR
    run_base = os.path.join(args.BASE_DIR, base_name)
    true_init_dir = os.path.join(run_base, "run_true_init")

    if os.path.exists(true_init_dir):
        shutil.rmtree(true_init_dir)
    os.makedirs(true_init_dir, exist_ok=True)

    # Load structure from the original GNOME CHGCAR(/lz4)
    # Load structure from the original GNOME CHGCAR(/lz4)
    try:
        atoms = atoms_from_chgcar(chgcar_path)
    except FileNotFoundError:
        print(f"❌ GNOME CHGCAR not found at {chgcar_path}")
        # Mark as attempted (so we don't spin forever on this row)
        df.at[idx, "performed_true_init"] = True
        df.to_csv(args.CSV_PATH, index=False)

        # Optionally also append a row to the TRUE-INIT results CSV
        header = [
            "CHGCAR_PATH",
            "csv_index",
            "slice_start",
            "slice_end",
            "status_true_init",
            "energy_true_init",
            "scf_steps_dav_true_init",
            "scf_steps_rmm_true_init",
            "scf_steps_total_true_init",
            "time_true_init",
        ]
        row_out = [
            chgcar_path,
            int(idx),
            int(start),
            int(end),
            "missing_gnome_chgcar",
            "",  # no energy
            0,
            0,
            0,
            0.0,
        ]

        write_header = not os.path.exists(args.RESULTS_CSV)
        with open(args.RESULTS_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row_out)

        # Exit with code 3 so the bash loop treats this as a per-structure failure and continues
        sys.exit(3)

    # Parse FFT grid from the same CHGCAR (for logging)
    grid_dims = get_chgcar_grid_dims(chgcar_path)

    # W&B init
    wandb.init(
        project="VASP_SCF_TrueInit_Gnome",
        name=f"true_init_{base_name}",
        config={
            "csv_index": int(idx),
            "slice_start": int(start),
            "slice_end": int(end),
            "chgcar_path_gnome": chgcar_path,
            "workdir": true_init_dir,
            "grid_dims": grid_dims,
            "sad_run_default_dir": sad_run_default_dir,
            "true_chgcar_path": true_chgcar_path,
            "icharg": 1,
        },
    )

    # Run TRUE-INIT SCF (ICHARG = 1, CHGCAR init from ground-truth SAD run)
    energy, dav_steps, rmm_steps, total_steps, wall_time = run_vasp_pymatgen(
        atoms=atoms,
        workdir=true_init_dir,
        icharg=1,
        chgcar_path=true_chgcar_path,
        vasp_cmd=tuple(args.vasp_cmd),
        grid_dims=grid_dims,
    )

    # Decide success vs failure based on energy
    if energy is None:
        status = "failed"

        wandb.log({
            "status_true_init": status,
            "scf_steps_dav_true_init": dav_steps,
            "scf_steps_rmm_true_init": rmm_steps,
            "scf_steps_total_true_init": total_steps,
            "time_true_init": wall_time,
        })
        wandb.finish()

        # Append a row with empty energy to TRUE-INIT results CSV
        header = [
            "CHGCAR_PATH",
            "csv_index",
            "slice_start",
            "slice_end",
            "status_true_init",
            "energy_true_init",
            "scf_steps_dav_true_init",
            "scf_steps_rmm_true_init",
            "scf_steps_total_true_init",
            "time_true_init",
        ]
        row_out = [
            chgcar_path,
            int(idx),
            int(start),
            int(end),
            status,
            "",  # energy missing
            dav_steps,
            rmm_steps,
            total_steps,
            wall_time,
        ]

        write_header = not os.path.exists(args.RESULTS_CSV)
        with open(args.RESULTS_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row_out)

        print("❌ TRUE-INIT run FAILED (OUTCAR/VASP issue).")
        print(f"  CSV index: {idx}")
        print(f"  Slice:     [{start}:{end})")
        print(f"  CHGCAR (GNOME): {chgcar_path}")
        print(f"  True CHGCAR (SAD): {true_chgcar_path}")
        print(f"  Grid dims (NGXF,NGYF,NGZF): {grid_dims}")
        print(f"  DAV steps: {dav_steps}")
        print(f"  RMM steps: {rmm_steps}")
        print(f"  TOTAL SCF steps: {total_steps}")
        print(f"  Wall time [s]: {wall_time}")
        print(f"Marked performed_true_init=True and appended FAILED row to {args.RESULTS_CSV}")
        sys.exit(3)

    # If we get here, it's a successful TRUE-INIT run
    status = "ok"

    # Log to W&B
    wandb.log({
        "status_true_init": status,
        "energy_true_init": energy,
        "scf_steps_dav_true_init": dav_steps,
        "scf_steps_rmm_true_init": rmm_steps,
        "scf_steps_total_true_init": total_steps,
        "time_true_init": wall_time,
    })
    wandb.finish()

    # Append to TRUE-INIT results CSV
    header = [
        "CHGCAR_PATH",
        "csv_index",
        "slice_start",
        "slice_end",
        "status_true_init",
        "energy_true_init",
        "scf_steps_dav_true_init",
        "scf_steps_rmm_true_init",
        "scf_steps_total_true_init",
        "time_true_init",
    ]
    row_out = [
        chgcar_path,
        int(idx),
        int(start),
        int(end),
        status,
        energy,
        dav_steps,
        rmm_steps,
        total_steps,
        wall_time,
    ]

    write_header = not os.path.exists(args.RESULTS_CSV)
    with open(args.RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row_out)

    print("✅ TRUE-INIT GNoME-style run done.")
    print(f"  CSV index: {idx}")
    print(f"  Slice:     [{start}:{end})")
    print(f"  CHGCAR (GNOME): {chgcar_path}")
    print(f"  True CHGCAR (SAD): {true_chgcar_path}")
    print(f"  Grid dims (NGXF,NGYF,NGZF): {grid_dims}")
    print(f"  Energy (true-init): {energy}")
    print(f"  DAV steps: {dav_steps}")
    print(f"  RMM steps: {rmm_steps}")
    print(f"  TOTAL SCF steps: {total_steps}")
    print(f"  Wall time [s]: {wall_time}")
    print(f"Appended TRUE-INIT results to {args.RESULTS_CSV}")
