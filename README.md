# GPU Algebra Search

Certifiable matrix-multiplication algorithm discovery over GF(2) on consumer GPUs.

A research project by [Requant Network](https://requant.network).

**Research campaign completed: CPU references, two independent GPU cohorts, retained certificates and final conclusions published.**

The GPU search reproduces the known **rank-47 decomposition of 4x4 matrix multiplication from the naive rank-64 scheme**. Shared-factor reductions with one probe reach that target in 4/8 and 2/8 ensembles in the two cohorts. All 112 rectangular 3x4 by 4x3 ensembles reach the selected rank-32 target. Every retained best scheme satisfies the complete tensor identity under independent CPU verification.

Read the [final research report](FINAL_REPORT.md) for the full mathematical contract, policy definitions, timing boundaries and conclusions. The [policy/cohort table](results/summary.csv) retains failures and uncertainty alongside successful searches.

## Research question

Can an exact warp-local flip-graph search discover useful binary matrix-multiplication decompositions on a consumer GPU, and which transition policies improve the probability and time of reaching a target rank?

The optimization target is algebraic rank: the number of scalar multiplications in a verified bilinear decomposition. A lower rank does not by itself establish a faster compiled GEMM, lower energy consumption or a secure proof-of-useful-work protocol.

## Mathematical scope and implementation

A scheme is a list of triples of binary factor masks. Each triple contributes a rank-one tensor; XOR of all triples must equal the exact matrix-multiplication tensor. Inputs and outputs use row-major indexing. Arithmetic is over GF(2), with no floating-point tolerance.

The independent [CPU reference](experiments/tensor.py) expands the full tensor into an arbitrary-precision bitset and also evaluates schemes on matrices against direct multiplication. It implements tensor-preserving flips, plus insertions and generalized reductions among up to five terms sharing a factor. The original [CUDA search](cuda/search.cu) uses warp-local matching and periodically reshuffles slots to expose additional matches.

The complete campaign covers two shapes, seven policies, eight ensembles per policy/shape and two independent seed cohorts: **224 ensembles of 256 walks**. All reported searches start from naive schemes. The optional compositional rank-49 start is independently checked on CPU, but is not used in the reported policy comparisons.

## Measured findings

Rank-47 attainment for the square task, with a nominal five-second budget per ensemble:

| Policy | Cohort 20261109 | Cohort 20261110 | Observed decision |
| --- | --- | --- | --- |
| Ordinary flips, one probe (`fixed1`) | 1/8 | 1/8 | Valid baseline; occasional target hits. |
| Ordinary flips, two probes (`fixed2`) | 0/8 | 0/8 | No target hit within the tested budget. |
| Group reductions, one probe (`group1`) | 4/8 | 2/8 | Strongest observed attainment in these cohorts. |
| Group reductions, two probes (`group2`) | 0/8 | 0/8 | More probing did not improve this target result. |
| Adaptive probing (`adaptive2`) | 3/8 | 2/8 | Sometimes successful; schedule-dependent. |
| Elite reseeding (`pool1`) | 0/8 | 0/8 | The tested pooling schedule did not help. |
| Smaller state capacity (`capacity96`) | 1/8 | 2/8 | A viable alternative with limited evidence. |

The rectangular target is reached in every ensemble. Rank 47 is a known bound reproduced here, not a new world record. With eight ensembles per cell, uncertainty remains broad; the [derived table](results/summary.csv) includes Wilson 95% intervals, restricted target-time scores and explicitly labeled successful-only medians. A negative result for one schedule does not rule out other implementations or budgets.

## Correctness evidence

- Four attributed CPU controls: AlphaTensor ranks 7, 23 and 47 for 2x2, 3x3 and 4x4, and the paper's rank-245 certificate for 7x7. The 7x7 control is not a GPU discovery result.
- 656 independent matrix-product checks and 900 CPU transition invariants.
- 2,400 intermediate GPU tensor snapshots checked before each cohort.
- 1,536 controlled GPU transitions matched with independent CPU flip, plus and generalized-reduction functions, including deliberately exercised rare branches.
- 57,344 runtime checks of walk-best certificates; all 224 ensemble-best certificates are archived for CPU replay. Altered coefficients are rejected.

Reference data is committed with [attribution and licenses](fixtures/README.md). Importers decode numerical data; they do not execute the paper's supplied programs. The GPU implementation is original and was not benchmarked against the paper's H200 binary.

## Reproduce the CPU checks

Requires Python 3.12 and its standard library. Run without `-O`, because assertions are correctness gates. Published fixtures are sufficient; regeneration with the optional importers is not required.

```sh
python cpu_study.py --output build/cpu-check.json
python verify_composition.py
python verify_certificates.py
python verify_results.py
python summarize_results.py
git diff --exit-code -- results/summary.csv
```

This writes the fresh CPU check to `build/`, rechecks retained GPU certificates and source/result bindings, and regenerates the summary from the published measurements. CI runs these checks without CUDA; it does not rerun GPU discovery.

## Prepare and reproduce the GPU campaign

Measured on Linux x86_64, Ryzen 5 5500 and RTX 5060 Ti 16 GB (SM 120), using Python 3.12, NumPy 2.2.6, CuPy 13.6.0 and CUDA 13.3.73 nvcc. CuPy uses the pinned CUDA 12.x runtime/header environment; the CUBIN is compiled separately with CUDA 13.3. See [environment.json](results/environment.json) and [build.json](results/build.json) for the exact measured stack and binary bindings.

Install dependencies in a separate environment and provide compatible CUDA 12.x headers:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-gpu.txt
mkdir -p build results
/path/to/cuda-13.3/bin/nvcc -O3 -std=c++17 -arch=sm_120 -cubin cuda/search.cu -o build/search.cubin
export CUDA_PATH=/path/to/cuda-12.8
export CUDA_VISIBLE_DEVICES=0
export PYTHON=python
bash run_gpu.sh
bash run_gpu.sh --seed 20261110 --output results/gpu-replication.json
STUDY=validate_transitions.py bash run_gpu.sh
STUDY=record_environment.py bash run_gpu.sh
python verify_certificates.py
python summarize_results.py
```

The default campaign runs both shapes and all seven policies. It writes raw measurements to `results/`; each case retains the seed, configuration, trajectory, target observation, walk-rank distribution, operation counts and best certificate. Run timing campaigns sequentially on a free GPU. The post-run build collector assumes the measured compiler location; see the [full reproduction notes](FINAL_REPORT.md#reproduction) before adapting it to another installation.

## Measurement rules

Trial i uses base seed + 1009i; policies share that trial's initial random state. The two base seeds are 20261109 and 20261110. An ensemble, rather than each of its 256 walks, is the statistical trial; pool walks additionally share elite schemes.

Targets are observed after 4,096-step chunks, making recorded hit times upper bounds. A run may exceed its nominal budget by one chunk. Unsuccessful ensembles remain right-censored. Complete verified wall time includes final device-to-host transfer and exact CPU checks of every walk-best, while allocation, initialization and a separate warm-up state are outside the timer. Device kernel intervals and host wall time are recorded separately.

## Repository and research record

| Artifact | Purpose |
| --- | --- |
| [RESEARCH.md](RESEARCH.md) | Original required experiments and completion evidence. |
| [METHODS.md](METHODS.md), [RESULTS.md](RESULTS.md), [FINAL_REPORT.md](FINAL_REPORT.md) | Methodology, evidence index and full interpretation. |
| [gpu_study.py](gpu_study.py), [cuda/search.cu](cuda/search.cu) | Complete seeded GPU search and transition kernels. |
| [CPU controls](results/cpu-study.json), [transition validation](results/transition-validation.json) | Independent correctness evidence. |
| [Primary GPU cohort](results/gpu-study.json), [replication](results/gpu-replication.json) | Raw successful and censored experiments. |
| [summary.csv](results/summary.csv) | Reproducible policy/cohort outcomes and uncertainty. |
| [build.json](results/build.json), [environment.json](results/environment.json) | Source/result/binary bindings and hardware/software metadata. |

## Limits and next research decisions

The current GPU factors fit in uint32 masks, and the kernel uses local stack storage as well as registers. Finite capacity, transition choices and non-cryptographic random states shape the search distribution. Results concern exact binary decompositions; they do not automatically transfer to real arithmetic, FP8, a faster matrix kernel or a permissionless mining lottery. Energy and profitability were not measured.

The strongest next experiments are larger independent cohorts, population schedules that preserve diversity, 64-bit factor masks and compiling verified schemes into measured GF(2) kernels. These are follow-up questions rather than results claimed by this campaign.

## License and contributions

Original code is [MIT licensed](LICENSE). Reference data retains its separately recorded Apache-2.0 or CC BY 4.0 license. Contributions should include a reproducible command, source revision, environment, independent certificate checks, all unsuccessful trials and explicit timing boundaries. Reproducible negative findings are welcome.

## Primary references

- [AlphaTensor factorizations](https://github.com/google-deepmind/alphatensor/tree/1949163da3bef7e3eb268a3ac015fd1c2dbfc767)
- [Medley et al., arXiv:2609.17533v2](https://arxiv.org/html/2609.17533v2)
- [Imported certificate conventions and attribution](fixtures/README.md)

## Related Requant Network research

- [GPU Wagner research](https://github.com/requant-network/gpu-wagner-research)
- [GPU erasure-code research](https://github.com/requant-network/gpu-erasure-research)
- [GPU ZK research](https://github.com/requant-network/gpu-zk-research)
- [GPU FHE research](https://github.com/requant-network/gpu-fhe-research)
- [GPU proof-of-useful-work research](https://github.com/requant-network/gpu-pouw-research)
