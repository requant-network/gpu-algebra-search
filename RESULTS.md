# Results and evidence index

A research project by [Requant Network](https://requant.network).

The complete experimental program and its limits are documented in [FINAL_REPORT.md](FINAL_REPORT.md).

| Evidence | Observed result |
| --- | --- |
| [CPU controls](results/cpu-study.json) | Four attributed certificates, 656 direct products, 900 exact transition checks. |
| [Controlled GPU transitions](results/transition-validation.json) | 1,536 transitions match independent CPU transformations; all tensors exact. |
| [Primary search](results/gpu-study.json) | 112 ensembles, seven policies, two shapes, all walk-best checks pass. |
| [Second cohort](results/gpu-replication.json) | 112 further ensembles, retained failures and exact certificates. |
| [Derived table](results/summary.csv) | 28 cohort/shape/policy summaries with censoring and success uncertainty. |
| [Build record](results/build.json) | Actual compiler, flags, CUDA binary and raw-record bindings. |
| [Environment](results/environment.json) | Non-identifying measured hardware/software metadata. |

The 4x4 rank-47 target is reached in 16 of 112 square-search ensembles across all tested policies. The 3x4x3 rank-32 target is reached in all 112 rectangular ensembles. Every one of 57,344 runtime walk-best certificates passed; the 224 archived ensemble-best certificates are independently rechecked by `verify_certificates.py`.

The best observed square-target policy by total hit count is `group1` (4/8 and 2/8). `fixed2`, `group2` and the tested `pool1` schedule do not reach rank 47 within these budgets. This does not establish impossibility or reject all population-search designs. The cohort samples are small and confidence intervals overlap. No new rank record, FP8 speedup, production miner or measured-energy result is claimed.
