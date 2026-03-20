#!/bin/bash
#SBATCH --account=def-mkoz_cpu
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=mean_comparison
#SBATCH --output=logs/%x_%j.out

mkdir -p logs/

module load python/3.11 rdkit/2023.09.5 cuda/12.6

source venv/bin/activate

python run.py -m +experiment=mean_comparison/sweep.yaml
python run.py -m +experiment=mean_comparison/baseline.yaml
