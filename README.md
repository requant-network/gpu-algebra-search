# GPU Algebra Search

A research project by [Requant Network](https://requant.network).

**Completed research campaign:** exact GF(2) matrix multiplication discovery on RTX 5060 Ti 16 GB, with two independent seed cohorts, retained failures and independently verified certificates.

The GPU search reproduces the known **rank-47 4x4 decomposition from the naive rank-64 scheme**. The tested one-probe group-reduction policy reaches it in 4/8 and 2/8 ensembles under nominal five-second budgets. All 112 rectangular 3x4 by 4x3 ensembles reach the rank-32 target. These are bounded experimental findings, not a new world record or a hardware GEMM speedup claim.

- [Final research report, tables and conclusions](FINAL_REPORT.md)
- [Experimental scope and completion evidence](RESEARCH.md)
- [Methods and limitations](METHODS.md)
- [Results and raw-data index](RESULTS.md)
- [Derived policy/cohort table](results/summary.csv)
- [Reference certificate attribution and licenses](fixtures/README.md)

## CPU verification

Python 3.12, standard library only; run without `-O`.

```sh
python cpu_study.py --output build/cpu-check.json
python verify_composition.py
python verify_certificates.py
python verify_results.py
python summarize_results.py
```

The CPU verifier rechecks every archived GPU best certificate and rejects altered coefficients. CI performs these CPU/reference, provenance and table checks; GPU discovery is a separately executed hardware campaign.

## GPU reproduction

See the [full commands and environment](FINAL_REPORT.md#reproduction). Compile `cuda/search.cu` into `build/search.cubin` with the tested CUDA compiler and install [requirements-gpu.txt](requirements-gpu.txt) into a separate environment. `gpu_study.py` runs complete seeded ensembles; `validate_transitions.py` matches actual GPU transitions against independent CPU transformations.

## License and related research

Original code is MIT licensed. Reference data retains its separately recorded Apache-2.0 or CC BY 4.0 license.

- [GPU Wagner research](https://github.com/requant-network/gpu-wagner-research)
- [GPU erasure-code research](https://github.com/requant-network/gpu-erasure-research)
- [Earlier finite-field/proving campaign](https://github.com/requant-network/gpu-zk-research)
