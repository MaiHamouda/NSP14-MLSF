#!/usr/bin/env python3
"""
Fingerprint generator: ECFP4 or PLEC.

Just run:  python fingerprints.py
and answer the questions. Advanced users can also pass options, e.g.

  python fingerprints.py --type ecfp4 --mol_path ligands.mol2 --output_csv out.csv
  python fingerprints.py --type plec --mol_path docked.mol2 --receptor_path prot.mol2 --output_csv out.csv

Needs: numpy, pandas, joblib, tqdm, oddt
Install with:  pip install numpy pandas joblib tqdm oddt
"""

import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

import oddt
from oddt.fingerprints import ECFP, PLEC

ECFP_SIZE = 2048
ECFP_DEPTH = 2          # depth 2 = radius 2 = ECFP4

PLEC_SIZE = 4096
PLEC_DEPTH_PROTEIN = 4
PLEC_DEPTH_LIGAND = 2
PLEC_DISTANCE_CUTOFF = 4.5

KNOWN_FORMATS = {".mol2": "mol2", ".sdf": "sdf", ".smi": "smi"}


# ---------- asking the user ----------

def clean_path(text):
    # handles drag-and-drop paths, which often come wrapped in quotes
    return text.strip().strip('"').strip("'")


def ask_choice(question, options):
    print(f"\n{question}")
    for i, (_, label) in enumerate(options, 1):
        print(f"  {i}) {label}")
    while True:
        answer = input("Type a number: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1][0]
        print("  Please type one of the numbers above.")


def ask_existing_file(question):
    while True:
        path = clean_path(input(f"\n{question}\n> "))
        if os.path.isfile(path):
            return path
        print("  File not found. Check the path and try again.")


def ask_output_path(default):
    answer = clean_path(input(f"\nWhere should the results be saved? [press Enter for '{default}']\n> "))
    return answer or default


def guess_format(path):
    return KNOWN_FORMATS.get(os.path.splitext(path)[1].lower())


def interactive_setup(args):
    print("=" * 50)
    print("  Molecular fingerprint generator")
    print("=" * 50)

    if not args.type:
        args.type = ask_choice(
            "Which fingerprint do you want to calculate?",
            [("ecfp4", "ECFP4  (compound only, no protein needed)"),
             ("plec", "PLEC   (docked compounds + receptor)")],
        )

    if not args.mol_path:
        if args.type == "plec":
            args.mol_path = ask_existing_file("Path to the docked ligands file (.mol2):")
        else:
            args.mol_path = ask_existing_file("Path to your compounds file (.mol2, .sdf or .smi):")

    if args.type == "plec" and not args.receptor_path:
        args.receptor_path = ask_existing_file("Path to the receptor file (.mol2):")

    if args.type == "ecfp4" and not args.format:
        args.format = guess_format(args.mol_path)
        if not args.format:
            args.format = ask_choice(
                "Could not tell the file type. Which is it?",
                [("mol2", "mol2"), ("sdf", "sdf"), ("smi", "smi")],
            )

    if not args.output_csv:
        default = f"{args.type}_features.csv"
        args.output_csv = ask_output_path(default)

    return args


# ---------- reading files ----------

def get_name(mol, index):
    name = getattr(mol, "title", None)
    if not name and hasattr(mol, "data"):
        name = mol.data.get("_Name")
    return str(name) if name else f"mol_{index}"


def load_molecules(path, file_format):
    logging.info(f"Reading molecules from {path} ({file_format})")
    mols, names = [], []
    for mol in oddt.toolkit.readfile(file_format, path):
        if mol is not None:
            names.append(get_name(mol, len(names)))
            mols.append(mol)
    if not mols:
        raise ValueError(f"No valid molecules found in {path}")
    logging.info(f"Loaded {len(mols)} molecules")
    return mols, names


def load_receptor(path):
    logging.info(f"Reading receptor from {path}")
    try:
        return next(oddt.toolkit.readfile("mol2", path))
    except StopIteration:
        raise ValueError(f"Could not read a receptor from {path}")


# ---------- fingerprints ----------

def ecfp4_one(mol):
    return ECFP(mol, depth=ECFP_DEPTH, size=ECFP_SIZE, sparse=False)


def plec_one(mol, receptor):
    return PLEC(mol, protein=receptor, size=PLEC_SIZE,
                depth_protein=PLEC_DEPTH_PROTEIN, depth_ligand=PLEC_DEPTH_LIGAND,
                distance_cutoff=PLEC_DISTANCE_CUTOFF, sparse=False)


def calculate(fp_type, mols, receptor, n_jobs):
    label = fp_type.upper()
    logging.info(f"Calculating {label} using {n_jobs} CPU core(s)")
    if fp_type == "ecfp4":
        jobs = (delayed(ecfp4_one)(m) for m in tqdm(mols, desc=label))
    else:
        jobs = (delayed(plec_one)(m, receptor) for m in tqdm(mols, desc=label))
    results = Parallel(n_jobs=n_jobs, backend="loky")(jobs)
    matrix = np.array(results)
    logging.info(f"Feature matrix shape: {matrix.shape}")
    return matrix


def save(names, matrix, fp_type, out_path):
    folder = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(folder, exist_ok=True)
    prefix = fp_type.upper()
    columns = [f"{prefix}_{i}" for i in range(matrix.shape[1])]
    df = pd.DataFrame(matrix, columns=columns)
    df.insert(0, "compound_name", names)
    df.to_csv(out_path, index=False)
    logging.info(f"Saved to {os.path.abspath(out_path)}")


# ---------- main ----------

def parse_args():
    cores = os.cpu_count() or 1
    default_jobs = max(1, cores // 2)

    p = argparse.ArgumentParser(description="Generate ECFP4 or PLEC fingerprints.")
    p.add_argument("--type", choices=["ecfp4", "plec"], help="Fingerprint type")
    p.add_argument("--mol_path", help="Compounds file (ECFP4: mol2/sdf/smi, PLEC: docked mol2)")
    p.add_argument("--receptor_path", help="Receptor mol2 file (PLEC only)")
    p.add_argument("--output_csv", help="Where to save the CSV")
    p.add_argument("--format", help="Compound file format for ECFP4 (mol2, sdf, smi)")
    p.add_argument("--n_jobs", type=int, default=default_jobs,
                   help=f"CPU cores to use (default: {default_jobs})")
    p.add_argument("--verbose", action="store_true", help="Show more detail")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    try:
        args = interactive_setup(args)
        if args.type == "plec":
            args.format = "mol2"

        for path in (args.mol_path, args.receptor_path):
            if path and not os.path.isfile(path):
                raise FileNotFoundError(f"File not found: {path}")
        if args.type == "plec" and not args.receptor_path:
            raise ValueError("PLEC needs a receptor file (--receptor_path)")

        print()
        mols, names = load_molecules(args.mol_path, args.format or guess_format(args.mol_path) or "mol2")
        receptor = load_receptor(args.receptor_path) if args.type == "plec" else None
        matrix = calculate(args.type, mols, receptor, args.n_jobs)
        save(names, matrix, args.type, args.output_csv)
        print("\nDone.")
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
