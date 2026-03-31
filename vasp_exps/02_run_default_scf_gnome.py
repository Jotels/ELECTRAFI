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
import numpy as np  # not strictly needed here, but you probably have it anyway
from pymatgen.core.structure import Structure

# Keep a handle to the original method if you ever want to call it.
_ORIG_GET_SG_INFO = Structure.get_space_group_info

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

def _patched_get_space_group_info(self, symprec: float = 0.01, angle_tolerance: float = 5.0):
    """
    Replacement for Structure.get_space_group_info that:
      * Uses the new get_symmetry_dataset API
      * Reads 'international_symbol' / 'number'
      * Falls back gracefully if anything breaks.

    Returns:
        (spacegroup_symbol, international_number)
    """
    try:
        # Use pymatgen's symmetry dataset helper (spglib backend by default)
        ds = self.get_symmetry_dataset(
            backend="spglib",
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            return_raw_dataset=False,  # <- important: we want the normalized dict if available
        )

        # ds may be a dict (normalized) or a raw dataset-like object
        symbol = None
        number = None

        if isinstance(ds, dict):
            # Pymatgen / spglib typical keys
            symbol = (
                    ds.get("international")  # main HM symbol
                    or ds.get("international_short")  # sometimes present
                    or ds.get("spacegroup_symbol")  # fallback
                    or ds.get("symbol")  # generic fallback
                    or ds.get("international_symbol")  # in case of weird variants
            )
            number = ds.get("number")
        else:
            # Raw SpglibDataset-like object
            symbol = (
                    getattr(ds, "international_symbol", None)
                    or getattr(ds, "international", None)
                    or getattr(ds, "symbol", None)
            )
            number = getattr(ds, "number", None)

        if not symbol:
            # Last-ditch: at least return *something* valid
            symbol = "P1"
        if number is None:
            number = 1
        print(f"[Patch] get_space_group_info: symbol={symbol}, number={number}")

        return str(symbol), int(number)

    except Exception:
        # If anything in the new path explodes, fall back to something safe.
        # You can also call the original method here in a try/except if you like.
        print(f"[Patch] get_space_group_info: failed")
        return ("P1", 1)

# Install the monkey patch
Structure.get_space_group_info = _patched_get_space_group_info

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
        "ISTART": 0,  # always start from scratch, not from stray WAVECAR
        "LCHARG": True,  # you need CHGCAR for later runs
        "LWAVE": False,  # you don’t care about WAVECAR here
    }

    # Let MPStaticSet build INCAR/KPOINTS/POTCAR using its MP config + PMG_VASP_PSP_DIR
    mp_set = MPStaticSet(
        structure,
        user_incar_settings=user_incar,
    )
    mp_set.write_input(workdir)

    # Optionally enforce GNOME FFT grid (so CHGCARs are compatible grid-wise)
    incar_path = os.path.join(workdir, "INCAR")
    patch_lmaxmix_for_dftu(incar_path)  # <--- add this

    # Optional CHGCAR init (not used for default run, but kept for symmetry)
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
# Main
# -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run default-from-scratch VASP SCF for one CHGCAR from a GNoME-style task CSV, "
            "using MPStaticSet with approximate MP settings."
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
        help="Base directory under which per-structure folders will be created",
    )
    parser.add_argument(
        "--results_csv", "-R",
        dest="RESULTS_CSV",
        required=True,
        help="CSV file to append default-run results to",
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

    # Clamp slice
    start = max(args.start_row, 0)
    end = n_rows if args.end_row is None else min(args.end_row, n_rows) if False else (
        n_rows if args.end_row is None else min(args.end_row, n_rows)
    )

    if start >= end:
        print(f"❌ Invalid slice [{start}:{end}) for CSV with {n_rows} rows.")
        raise SystemExit(1)

    print(f"Using slice [{start}:{end}) out of {n_rows} rows")

    # Ensure columns exist (backwards safety)
    for col in ["performed_default", "performed_true_init", "performed_ml_init"]:
        if col not in df.columns:
            df[col] = False

    # Restrict to our slice **before** any filtering
    df_slice = df.iloc[start:end]

    # --------- De-dup using existing results CSV ---------
    if os.path.exists(args.RESULTS_CSV):
        try:
            res_df = pd.read_csv(args.RESULTS_CSV, usecols=["CHGCAR_PATH"])
            already_done_paths = set(res_df["CHGCAR_PATH"].astype(str))
        except Exception as e:
            print(f"⚠️ Could not read results CSV {args.RESULTS_CSV}: {e!r}")
            already_done_paths = set()
    else:
        already_done_paths = set()

    if already_done_paths:
        # Mark any rows in the slice whose CHGCAR_PATH is already in results as done
        mask_already = df_slice["CHGCAR_PATH"].isin(already_done_paths)
        if mask_already.any():
            done_indices = df_slice.index[mask_already]
            print(f"Marking {len(done_indices)} rows as performed_default=True "
                  f"because they already appear in {args.RESULTS_CSV}.")
            df.loc[done_indices, "performed_default"] = True
            # Refresh df_slice to reflect this
            df_slice = df.iloc[start:end]

    # Now pick pending rows
    pending = df_slice[df_slice["performed_default"] == False]
    if pending.empty:
        print(f"✅ No pending default runs in slice [{start}:{end}).")
        # exit code 2: slice exhausted
        raise SystemExit(2)

    # idx is still the original CSV index (not 0..len(slice))
    idx = pending.index[0]
    chgcar_path = df.at[idx, "CHGCAR_PATH"]

    print(
        f"Selected row idx={idx} (slice [{start}:{end})), "
        f"CHGCAR={chgcar_path}"
    )

    # Immediately mark this row as performed_default, so we don't re-run it
    df.at[idx, "performed_default"] = True
    df.to_csv(args.CSV_PATH, index=False)

    # Use basename as ID for folders
    base_name = os.path.splitext(os.path.basename(chgcar_path))[0]
    run_base = os.path.join(args.BASE_DIR, base_name)
    default_dir = os.path.join(run_base, "run_default")

    # Prepare directory
    if os.path.exists(default_dir):
        shutil.rmtree(default_dir)
    os.makedirs(default_dir, exist_ok=True)

    # Load structure from CHGCAR(/lz4)
    atoms = atoms_from_chgcar(chgcar_path)

    # Parse FFT grid from the same CHGCAR so that this run
    # uses the GNOME/ECD grid (for later CHGCAR-initialized runs).
    grid_dims = get_chgcar_grid_dims(chgcar_path)

    # W&B init
    wandb.init(
        project="VASP_SCF_Default_Gnome_v2",
        name=f"default_{base_name}",
        config={
            "csv_index": int(idx),
            "slice_start": int(start),
            "slice_end": int(end),
            "chgcar_path": chgcar_path,
            "workdir": default_dir,
            "grid_dims": grid_dims,
        },
    )

    # Run default SCF (ICHARG = 2, no CHGCAR init, MPStaticSet handles POTCAR)
    energy, dav_steps, rmm_steps, total_steps, wall_time = run_vasp_pymatgen(
        atoms=atoms,
        workdir=default_dir,
        icharg=2,
        chgcar_path=None,
        vasp_cmd=tuple(args.vasp_cmd),
        grid_dims=grid_dims,
    )

    # Decide success vs failure based on energy
    if energy is None:
        status = "failed"

        wandb.log({
            "status": status,
            "scf_steps_dav_default": dav_steps,
            "scf_steps_rmm_default": rmm_steps,
            "scf_steps_total_default": total_steps,
            "time_default": wall_time,
        })
        wandb.finish()

        # Append a row with empty energy to results CSV
        header = [
            "CHGCAR_PATH",
            "csv_index",
            "slice_start",
            "slice_end",
            "status_default",
            "energy_default",
            "scf_steps_dav_default",
            "scf_steps_rmm_default",
            "scf_steps_total_default",
            "time_default",
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

        print("❌ Default run FAILED (OUTCAR/VASP issue).")
        print(f"  CSV index: {idx}")
        print(f"  Slice:     [{start}:{end})")
        print(f"  CHGCAR:    {chgcar_path}")
        print(f"  Grid dims (NGXF,NGYF,NGZF): {grid_dims}")
        print(f"  DAV steps: {dav_steps}")
        print(f"  RMM steps: {rmm_steps}")
        print(f"  TOTAL SCF steps: {total_steps}")
        print(f"  Wall time [s]: {wall_time}")
        print(f"Marked performed_default=True and appended FAILED row to {args.RESULTS_CSV}")
        sys.exit(3)

    # If we get here, it's a successful run
    status = "ok"

    # Log to W&B — log all three counters
    wandb.log({
        "status": status,
        "energy_default": energy,
        "scf_steps_dav_default": dav_steps,
        "scf_steps_rmm_default": rmm_steps,
        "scf_steps_total_default": total_steps,
        "time_default": wall_time,
    })
    wandb.finish()

    # Append to results CSV (with all three counters)
    header = [
        "CHGCAR_PATH",
        "csv_index",
        "slice_start",
        "slice_end",
        "status_default",
        "energy_default",
        "scf_steps_dav_default",
        "scf_steps_rmm_default",
        "scf_steps_total_default",
        "time_default",
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

    print("✅ Default GNoME-style run done.")
    print(f"  CSV index: {idx}")
    print(f"  Slice:     [{start}:{end})")
    print(f"  CHGCAR:    {chgcar_path}")
    print(f"  Grid dims (NGXF,NGYF,NGZF): {grid_dims}")
    print(f"  Energy:    {energy}")
    print(f"  DAV steps: {dav_steps}")
    print(f"  RMM steps: {rmm_steps}")
    print(f"  TOTAL SCF steps: {total_steps}")
    print(f"  Wall time [s]: {wall_time}")
    print(f"Appended results to {args.RESULTS_CSV}")
