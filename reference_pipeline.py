#

import csv
import os
import numpy as np
from argparse import ArgumentParser
from typing import Self

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

class Molecule:
    MAX_MOLECULAR_WEIGHT = 500
    MAX_LOG_P = 5
    MAX_HBD = 5
    MAX_HBA = 10

    FINGERPRINT_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=4096
    )

    def __init__(self, mol_id: str, smiles: str, molecule: Chem.Mol) -> None:
        self.mol_id = mol_id
        self.smiles = smiles

        self.fingerprint = self.FINGERPRINT_GENERATOR.GetFingerprint(molecule)
        self.MW =  Descriptors.MolWt(molecule)
        self.LogP = Crippen.MolLogP(molecule)
        self.TPSA = rdMolDescriptors.CalcTPSA(molecule)
        self.HBD = rdMolDescriptors.CalcNumHBD(molecule)
        self.HBA = rdMolDescriptors.CalcNumHBA(molecule)
        self.RotBonds = rdMolDescriptors.CalcNumRotatableBonds(molecule)
        self.Rings = rdMolDescriptors.CalcNumRings(molecule)
        self.QED = Descriptors.qed(molecule)

    @classmethod
    def from_row(cls, row: dict) -> Self | None:
        smiles = row["smiles"]
        if not smiles or not smiles.strip():
            return None
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        return cls(row["mol_id"], smiles, molecule)

    @property
    def ro5_pass(self) -> bool:
        return (
            self.MW <= self.MAX_MOLECULAR_WEIGHT
            and self.LogP <= self.MAX_LOG_P
            and self.HBD <= self.MAX_HBD
            and self.HBA <= self.MAX_HBA
        )

    def to_report(self) -> dict:
        """Build one row of the processed CSV."""
        return {
            "mol_id": self.mol_id,
            "smiles": self.smiles,
            "MW": self.MW,
            "LogP": self.LogP,
            "TPSA": self.TPSA,
            "HBD": self.HBD,
            "HBA": self.HBA,
            "RotBonds": self.RotBonds,
            "Rings": self.Rings,
            "QED": self.QED,
            "Ro5_pass": self.ro5_pass,
        }

class GraphGenerator():

    def __init__(
            self, 
            input_file_path: str, output_file_path: str, 
            invalid_output_file_path: str
        ) -> None:
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.invalid_output_file_path = invalid_output_file_path
        self.output_folder = os.path.dirname(output_file_path) or "." 

        self._rows = []
        self._valid_molecules = []
        self._invalid_molecules = []
        self._molecules = []



    def load_molecule(self):
        """Loads molecules data from csv file."""
        with open(self.input_file_path, newline="", encoding="utf-8") as in_file:
            self._rows = list(csv.DictReader(in_file))


    def calculate_properties(self) -> None:
        """Calculates molecular properties from rows and separates valid
        and invalid records.
        """
        self._valid_molecules = []
        self._invalid_molecules = []

        for row in self._rows:
            molecule = Molecule.from_row(row)
            if molecule is None:
                self._invalid_molecules.append(row)
            else:
                self._valid_molecules.append(molecule)


    def print_statistics(self) -> None:
        """Print some statistical info about analyzed data."""
        valid_count = len(self._valid_molecules)
        invalid_count = len(self._invalid_molecules)
        total_count = valid_count + invalid_count
        print(f"Valid molecules: {valid_count} / {total_count}")
        ro5_pass_count = sum(molecule.ro5_pass for molecule in self._valid_molecules)
        print(f"Ro5 pass: {ro5_pass_count} / {total_count}")
        
        if not self._valid_molecules:
            print("No valid molecules to analyze.")
            return
        
        min_mw, max_mw = min(molecule.MW for molecule in self._valid_molecules), \
                            max(molecule.MW for molecule in self._valid_molecules)
        print(f"MW range: {min_mw:.1f} — {max_mw:.1f}")

        for molecule in self._valid_molecules[:10]:
            print(molecule.mol_id, molecule.MW, molecule.LogP, 
            molecule.TPSA, molecule.QED, molecule.ro5_pass)


    def clusterise(self) -> None:
        """Perform Butina clustering algorithm on molecules."""
        fingerprints = [molecule.fingerprint for molecule in self._valid_molecules]
        distances = []
        for i in range(1, len(fingerprints)):
            similarity_scores = BulkTanimotoSimilarity(
                fingerprints[i], 
                fingerprints[:i]
            )
            distances.extend([1 - score for score in similarity_scores])
        
        clusters = Butina.ClusterData(
            distances, len(fingerprints), 0.7, isDistData=True
        )
        self._cluster_labels = [-1] * len(self._valid_molecules)
        for cluster_id, members in enumerate(clusters):
            for idx in members:
                self._cluster_labels[idx] = cluster_id


    def plot_graph(self) -> None:
        fingerprints = np.array(
            [list(molecule.fingerprint) for molecule in self._valid_molecules],
            dtype=np.uint8,
        )
        n_neighbors = min(10, len(fingerprints) - 1) # limit neighbors to avoid errors with small datasets
        x, y, s, t = TMAP(metric="jaccard", n_neighbors=n_neighbors).fit_transform(fingerprints)

        f = Faerun(view="front", coords=False, clear_color="#FFFFFF")
        f.add_scatter(
            name="molecules",
            data={
                "x": x,
                "y": y,
                "c": [
                    self._cluster_labels,
                    [molecule.LogP for molecule in self._valid_molecules],
                    [molecule.MW for molecule in self._valid_molecules],
                ],
                "labels": [f"{molecule.smiles}__{molecule.mol_id}"
                           for molecule in self._valid_molecules],
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
        f.plot("molecules", path=self.output_folder, template="smiles")


    def filter_output(self) -> list[dict]:
        """Filter record info only to required fields."""
        return [molecule.to_report() for molecule in self._valid_molecules]
  

    @staticmethod # useful as a static method since it doesn't depend on instance state
    def save_processed_data(file_path: str, values: list[dict]) -> None:
        """Save processed molecular data, including estimated properties."""
        with open(file_path, "w", newline="", encoding="utf-8") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
        

    def run(self) -> None:
        """Orchestrated pipeline."""
        
        self.load_molecule()
        self.calculate_properties()

        self.print_statistics()
        if self._invalid_molecules:
            self.save_processed_data(self.invalid_output_path, self._invalid_molecules)
        if not self._valid_molecules:
            print("No valid molecules to process. Exiting.")
            return
        
        filtered_molecules = self.filter_output()
        self.save_processed_data(self.output_file_path, filtered_molecules)
        if len(self._valid_molecules) == 1:
            print("Only one valid molecule found. Skipping clustering and graph plotting.")
            return

        self.clusterise()
        self.plot_graph()


if __name__ == "__main__":
    parser = ArgumentParser(description="Script estimates molecular properties from molecules \
                                                in smiles format, cluster them and plot graph")
    parser.add_argument("--input", "-i", type=str, default="molecules.csv",
                        help="Input file name")
    parser.add_argument("--output", "-o", type=str, default="molecules_processed.csv",
                        help="Output file name")
    parser.add_argument("--invalid_output", "-I", type=str, default="molecules_invalid", 
                        help="Invalid molecules output file name")
    parser.add_argument("--input_dir", "-d", type=str, default="data", help="Input drectory")
    parser.add_argument("--output_dir", "-D", type=str, default="output", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    if args.input_dir:
        input_file_path = os.path.join(args.input_dir, args.input)
    else:
        input_file_path = args.input
    if args.output_dir:
        output_file_path = os.path.join(args.output_dir, args.output)
    else:
        output_file_path = args.output
    invalid_output_file_path = os.path.join(args.output_dir, args.invalid_output)

    print(f"Launching pipeline for {os.path.basename(input_file_path)}...")
    generator = GraphGenerator(input_file_path, output_file_path, invalid_output_file_path)
    generator.run()
