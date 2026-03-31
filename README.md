# Promotion, Demand, and Fulfillment

Semester final project scaffold for analyzing JD.com transaction-level data with a tight IE/OR scope.

## Project focus

- Promotion effects on order economics and order timing
- Fulfillment outcomes under local vs. remote warehouse assignment
- A simplified warehouse assignment optimization model using observed inventory availability

## Quick start

1. Create a Python environment and install `requirements.txt`.
2. Put the JD CSV files in `data/raw/`, or point the notebook to your local dataset folder.
3. Open [notebooks/01_jd_project_starter.ipynb](C:\Users\Qiang Gao\Desktop\IE5404-JD\notebooks\01_jd_project_starter.ipynb).
4. Run the data audit, feature engineering, and candidate-assignment cells.

## Core files

- `src/jd_project/data.py`: table loaders, validation helpers, and base joins
- `src/jd_project/features.py`: engineered variables for analysis/modeling
- `src/jd_project/optimization.py`: simplified warehouse assignment MILP in PuLP
- `notebooks/01_jd_project_starter.ipynb`: main executable workflow for the class project
