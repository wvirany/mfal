# Data Directory

```
data/
├── raw/
│   ├── results.csv                    # 59,356 molecules with docking + MMGBSA scores
│   ├── results_benchmark.csv          # Experimental dataset
│   └── similarity_scores.csv          # MCS similarity scores from ZINC22 screening
│
└── processed/
    └── (preprocessing outputs go here)
```

## Key Columns

**results.csv:**
- `prot_smiles` - Use this for embeddings
- `vina_score` - Docking score
- `mmgbsa_score` - MMGBSA score 
- `mmgbsa_score_sem` - Standard error from 20 MD replicates

**results_benchmark.csv:**
- `exp_dg` - Experimental binding free energy
- `mmgbsa_score_blind` - MMGBSA prediction
- `vina_score_blind` - Docking prediction

**Note:** Lower (more negative) scores = stronger binding
