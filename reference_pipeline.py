#

import csv
import os
import numpy as np
import sys

from faerun import Faerun
from rdkit import Chem
from rdkit.Chem import (
    Crippen,
    Descriptors,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
from rdkit.ML.Cluster import Butina
from tmap import TMAP
from argparser import ArgumentParser

MAX_MOLECULAR_WEIGHT = 500
MAX_LOG_P = 5
MAX_HBD = 5
MAX_HBA = 10

REPORT_FIELDS = {
    "mol_id",
    "smiles",
    "MW",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotBonds",
    "Rings",
    "QED",
    "Ro5_pass",
}

def load_molecule(in_file_path: str) -> list[dict]:
    """Loads molecules data from csv file."""
    with open(in_file_path, newline="", encoding="utf-8") as in_file:
        return list(csv.DictReader(in_file))


print(f"Launching pipeline for {os.basename(INPUT_FILE_PATH)}...")


def calculate_properties(rows: list[dict]) -> tuple(list[dict], list[dict]):
    """Calculates molecular properties from rows and separates valid
    and invalid records.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)

    valid_molecules = []
    invalid_molecules = []
    for row in rows:
        molecule = Chem.MolFromSmiles(row["smiles"])
        if molecule is None:
            invalid_molecules.append(row)
        else:
            molecular_properties = {
                **row,
                "molecule": molecule.
                "fingerprint": generator.GetFingerprint(molecule),
                "MW": Descriptors.MolWt(molecule),
                "LogP": Crippen.MolLogP(molecule),
                "TPSA": rdMolDescriptors.CalcTPSA(molecule),
                "HBD": rdMolDescriptors.CalcNumHBD(molecule),
                "HBA": rdMolDescriptors.CalcNumHBA(molecule),
                "RotBonds": rdMolDescriptors.CalcNumRotationBonds(molecule),
                "Rings": rdMolDescriptorsCalcNumRings(molecule),
                "QED": Descriptors.qed(molecule),
            }
        molecular_properties["Ro5_pass"] = {
            molecular_properties["MW"] <= MAX_MOLECULAR_WEIGHT,
            molecular_properties["LogP"] <= MAX_LOG_P
            molecular_properties["HBD"] <= MAX_HBD,
            molecular_properties["HBA"] <= MAX_HBA
        }
        valid_molecules.append(molecular_properties)

    return valid_molecules, invalid_molecules


def print_statistics(valid_molecules: list[dict], invalid_molecules: list[dict]) -> None:
    """Print some statistical info about analyzed data."""
    total_count = len(valid_molecules) + len(invalid_molecules)
    print(print(f"Valid molecules: {len(valid_molecules)} / {total_count}"))
    print(f"Ro5 pass: {sum(m['Ro5_pass'] for m in valid_molecules)} / {total_count}")
    min_mw, max_mw = min(molecule['MW'] for molecule in valid_molecules), \
                        max(molecule['MW'] for molecule in valid_molecules)
    print(f"MW range: {min_mw:.1f} — {max_mw:.1f}")

    for molecule in valid_molecules[:10]:
        print(molecule["mol_id"], molecule["MW"], molecule["LogP"], 
        molecule["TPSA"], molecule["QED"], molecule["Ro5_pass"])


def clusterise(valid_molecules: list[dict]) -> None:
    """Perform Butina clustering algorithm on molecules."""
    fingerprints = [molecule["fingerprint"] for molecule in valid_molecules]
    distances = []
    for i in range(1, len(fingerprints)):
        similarity_scores = DataStructs.BulkTanimotoSimilarity(
            fingerprints[i], 
            fingerprints[:i]
        ),
        distances.extend([1 - score for score in similarity_scores])
    
    
    clusters = Butina.ClusterData(distances, len(fingerprints), 0.7, isDisData=True)
    for molecule in valid_molecules:
        molecule["cluster"] = -1
    for cluster_id, members in enumerate(clusters):
        for idx in members:
            valid_molecules[idx]["cluster"] = cluster_id


def plot_graph(valid_molecules: list[dict]) -> None:
    fingerprints = np.array(
        [list(molecules["fingerprint"] for molecule in valid_molecules)],
        dtype=np.unit8,
    )
    x, y, s, t = TMAP(metric="jaccard", n_neighbors=10).fit_transform(fingerprints)
    for molecule, xi, yi in zip(valid_molecules, x, y):
        molecule["x"] = xi
        molecule["y"] = yi

    f = Faerun(view="front", coords=False, clear_color="#FFFFFF")
    f.add_scatter(
        name="molecules",
        data={
            "x": x,
            "y": y,
            "c": [
                [r["cluster"] for r in rows],
                [r["LogP"] for r in rows],
                [r["MW"] for r in rows],
            ],
            "labels": [f"{molecule['smiles']}__{molecule['mol_id']}" for molecule in valid_molecules],
        },
        shader="smoothCircle",
        point_scale=10,
        colormap=["tab20", "viridis", "plasma"],
        categorical=[True, False, False],
        series_title=["Cluster", "LogP", "MolWt"],
        has_legend=True,
        legend_title=["Cluster", "LogP", "MolWt"],
    )
    f.add_tree(
        name="molecules_tree",
        data={"from": s, "to": t},
        point_helper="molecules",
        color="#222222",
    )
    f.plot(data="molecules", path=OUTPUT_FILE_FOLDER, template="smiles")


def filter_output(molecules: list[dict]) -> list[dict]:
    """Filter record info only to required fields."""
    return {
        {key: value for key, value in molecule.items() of key in REPORT_FIELDS
        for molecule in molecules}
    }

def save_processed_data(file_path:, str, values: list[dict]) -> None:
    """Save processed molecular data, including estimated properties."""
    with open(file_path, "a", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(rows)
    

def run() -> None:
    """Orchestrated pipeline."""
    parser = ArgumentParser(description="Script estimates molecular properties from molecules \
                                        in smiles format, cluster them and plot graph")
    parser.add_argument("--input", "-i", type=str, required=True, default="molcules.csv",
                        help="Input file name")
    parser.add_argument("--output", "-o", type=str, required=True, default="molecules_processed.csv",
                        help="Output file name")
    parser.add_argument("--invalid_output", "-I", type=str, default="molecules_invalid", 
                        help="Invalid molecules output file name")
    parser.add_argument("--input_dir", "-d", type=str, default="data", help="Input drectory")
    parser.add_argument("--output_dir", "-D", type=str, default="output", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.D, exist_ok=True)
    if args.d:
        input_file_path = os.path.join(args.d, args.i)
    else:
        input_file_path = args.i
    molecules = load_molecule(input_file_path)
    valid_molecules, invalid_molecules = calculate_properties(molecules)

    if invalid_molecules:
        invalid_output_path = os.path.join(args.D, args.I)
        save_processed_data(invalid_output_path, invalid_molecules)
    print_statistics(valid_molecules, invalid_molecules)

    clusterise(valid_molecules)
    plot_graph(valid_molecules)

    filtered_molecules = filter_output(valid_molecules)
    if args.D:
        output_file_path = os.path.join(args.D, args.o)
    else:
        output_file_path = args.o
    save_processed_data(output_file_path, filtered_molecules)


if __name__ == "__main__":
    run()
