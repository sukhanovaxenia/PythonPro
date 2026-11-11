# PythonPro2026

## Description

Pipeline for chemoinformatics analysis of molecules dataset (based on RDKit):

1. parses molecules in SMILES format from users data (default: `data/molecules.csv`, columns `mol_id`, `smiles`);
2. validates structures and obtains descriptors: MW, LogP, TPSA, HBD, HBA, number of rotating bonds, amount of rings, QED;
3. estimates correspondence to rule of 5 Lipinsk (`Ro5_pass`);
4. clusterises molecules by Butina algorithm based on Tannimoto similarity (Morgan-отпечатки, radius=2, 4096 bit);
5. builds TMAP dimensionality reduction and saves interactive visualisation (Faerun) and table of results into provided output file (default: `output/molecules_new.csv`).

## Structure:

```
data/molecules.csv       # input data
output/                  # results (CSV and HTML-visualisation)
reference_pipeline.py    # main script
pyproject.toml, uv.lock  # dependencies
```

## System requirements

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) — package manager and environments
- Windows, Linux and macOS
- Main dependencies (installed automatically): `rdkit`, `tmap2`, `faerun`, `numpy`, `matplotlib<3.9`
- Modern browser for results viewing

## Instruction

1. Clone repository and swithc to project directory:

   ```bash
   git clone <URL репозитория>
   cd PythonPro
   ```

2. Create environment and install dependecies:

   ```bash
   uv sync
   ```

3. Launch pipeline:

   ```bash
   uv run python reference_pipeline.py

   # or with user files
   uv run python reference_pipeline.py -i <User input file path> -o <User output file path>

   # or with user files and assigned directories
   uv run python reference_pipeline.py -i <User input file path> -o <User output file path> -d <input directory> -D <output directory>
   ```

4. Results appear in provided directory or defaults `output/` one: table `molecules_new.csv` (or user's if assigned) and HTML-file woth visualisation `molecules.html`, which should be opened in web-browser.

## Contact information

- Author: Xenia Sukhanova
- Email: sukhanovaxenia@gmail.com
