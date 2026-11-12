# Certifiable GPU algebra search: final research report

A research project by [Requant Network](https://requant.network). Research date: 2026-11-09, Asia/Yekaterinburg; raw timestamps use UTC.

## Finding

An original GPU flip-graph implementation on RTX 5060 Ti reproduces the known **rank-47 decomposition of 4x4 matrix multiplication over GF(2)**, starting from the naive rank-64 scheme. All archived best schemes satisfy the complete tensor identity, checked independently on CPU. This reproduces a known rank bound; it is not a new world record.

Within two cohorts of eight five-second searches per policy, shared-factor reductions with one probe per axis reach rank 47 in 4/8 and 2/8 ensembles. Ordinary one-probe flips reach it in 1/8 and 1/8. Two-probe policies reach rank 49 but do not reach 47 within these budgets. Adaptive probing and smaller state capacity sometimes succeed; the tested elite-reseeding schedule does not. The small sample sizes support a practical follow-up, not a precise universal probability or speedup claim.

For the rectangular 3x4 by 4x3 task, all 112 ensembles reach the chosen rank-32 target. This is a second complete search workload, not merely a transition microbenchmark.

## Hypotheses and decisions

| Hypothesis | Test | Decision |
| --- | --- | --- |
| Warp-local matching can execute complete valid algebraic search on a consumer GPU. | Full walks from naive schemes; every walk-best checked as an exact tensor. | Supported on both tested shapes. |
| Generalized shared-factor reductions improve target attainment. | Fixed probing with and without group reductions, paired initial seeds. | Better observed attainment at one probe; no corresponding success at two probes within this budget. |
| More probing is always beneficial. | One versus two probes and adaptive reduction after stagnation. | Rejected for the tested policy family: probing also changes escape behavior. |
| Sharing elite schemes necessarily helps. | Worst-quarter reseeding at the recorded interval. | Not supported by this configuration; other intervals, fractions and diversity controls remain untested. |
| Lower algebraic rank implies a faster hardware GEMM. | No hardware compiler/GEMM comparison in this campaign. | Not established. |

## Mathematical contract

A scheme is a list of triples of nonzero binary masks (A_r,B_r,C_r). It represents the exact identity M = XOR_r A_r tensor B_r tensor C_r for the matrix multiplication tensor. Input A is row-major m by k, input B is row-major k by n, and output C is row-major m by n. The reported rank is the number of retained rank-one terms in this particular decomposition; this is an upper bound on the tensor's minimum rank.

[experiments/tensor.py](experiments/tensor.py) independently expands the full tensor into an arbitrary-precision bitset and compares it with the elementary matrix-product tensor. A separate evaluator computes scalar GF(2) matrix products from the masks and compares them with a direct matrix multiplication implementation. There is no floating-point tolerance.

Flips, plus insertions and shared-factor reductions preserve the tensor exactly. Generalized reductions detect XOR dependencies among up to five terms sharing one factor, update the remaining factors, and delete zero terms. A different bit order or a changed coefficient must not be accepted merely because a random product happens to agree.

## Reference controls and attribution

The CPU campaign verifies four attributed controls: AlphaTensor ranks 7, 23 and 47 for 2x2, 3x3 and 4x4, and the rank-245 7x7 certificate in Appendix A of [Medley et al., arXiv:2609.17533v2](https://arxiv.org/html/2609.17533v2). The latter was decoded as numerical data and verified with our own implementation; the paper's supplied program was not executed. The 7x7 result is a CPU control, not a GPU search result in this repository.

AlphaTensor data comes from revision `1949163da3bef7e3eb268a3ac015fd1c2dbfc767`; the third factor is transposed from its trace convention to our output convention. [Fixture attribution and licenses](fixtures/README.md) identify Apache-2.0 and CC BY 4.0 data. Original research code is MIT licensed.

The recent paper motivates warp-local matching and generalized reductions. Its H200 search binary was not used for these measurements. Our result is an independently implemented search campaign on a different device, with explicit random seeds and retained failures. No comparison against its reported H200 throughput is claimed.

## Implementations and correctness evidence

[cuda/search.cu](cuda/search.cu) stores a bounded collection of factor masks per walk and uses `__match_any_sync` to find shared factors. A warp applies one transition at a time. Periodic per-slot shuffles expose matches between terms previously held on the same lane. Masks are uint32; the GPU campaign uses shapes whose factors fit in 32 bits.

The campaign validates 2,400 intermediate tensor snapshots before each cohort. A separate [transition validation](results/transition-validation.json) matches **1,536 actual GPU transitions** against the independent CPU flip, plus and generalized-reduction functions. Forced insertion and deliberately added zero-sum triads exercise branches that are rare in an ordinary short trace. All tensors remain exact. Some GPU group reductions are also expressible as ordinary pairwise flips; the validation records both classifications rather than requiring them to coincide.

During the two complete cohorts, every walk's best scheme is CPU-verified: 224 ensembles times 256 walks gives **57,344 runtime certificate checks**. Each ensemble retains its best certificate, full walk-rank distribution, first-hit-step markers, operation counts and trajectory. All **224 archived certificates** can be rechecked without a GPU using [verify_certificates.py](verify_certificates.py); changing a retained coefficient is rejected. The reference campaign also passes 656 direct matrix-product checks and 900 CPU transition invariants. The optional compositional rank-49 starting scheme has an exact CPU check, but the reported search comparisons all start from naive schemes.

## Policies and experimental design

Each cohort uses base seed 20261109 or 20261110. Trial i uses base + 1009i. Policies share each trial's initial random state. Each trial is an ensemble of 256 pseudo-random walks, with a nominal five-second wall budget and a final GPU-chunk observation. Walks are not treated as 256 independent statistical trials; policy tables use eight ensembles per cohort. Pool walks additionally share elite schemes.

`fixed1` and `fixed2` use ordinary flips and fallback plus transitions, with one or two probes per axis. `group1` and `group2` add dependency reductions. `adaptive2` starts at two probes and drops to one after 2,048 steps without improving a walk's best rank. `pool1` uses one probe/group reduction and replaces the worst quarter with the current elite every 65,536 step bound, keeping the other three quarters. `capacity96` tests 96 slots instead of 128. These schedules and caps define the tested algorithms; a negative result is not a rejection of all adaptive or population search.

The device uses non-cryptographic xorshift states. This is an optimization experiment, not a claim of cryptographic unpredictability or a secure mining lottery. Search stopping and finite state capacity affect the explored distribution.

## Results

| Policy | Rank-47 hits, cohort 20261109 | Rank-47 hits, cohort 20261110 | Restricted mean target time, first / second (s) |
| --- | --- | --- | --- |
| fixed1 | 1/8 | 1/8 | 4.930 / 4.818 |
| fixed2 | 0/8 | 0/8 | 5.000 / 5.000 |
| group1 | 4/8 | 2/8 | 4.466 / 4.530 |
| group2 | 0/8 | 0/8 | 5.000 / 5.000 |
| adaptive2 | 3/8 | 2/8 | 4.806 / 4.583 |
| pool1 | 0/8 | 0/8 | 5.000 / 5.000 |
| capacity96 | 1/8 | 2/8 | 4.896 / 4.590 |

[summary.csv](results/summary.csv) includes each cohort/shape/policy, success counts, Wilson 95% intervals, final ranks, and a restricted target-time score that charges every failed search its full nominal budget. Successful-only medians are explicitly labeled; they must not be interpreted as unconditional time-to-solution. With only eight ensembles per cell, uncertainty is substantial.

Raw records: [primary cohort](results/gpu-study.json), [second cohort](results/gpu-replication.json), [CPU controls](results/cpu-study.json), [GPU/CPU transition checks](results/transition-validation.json), [compiler/binary bindings](results/build.json), [hardware/software](results/environment.json).

## Timing and resource boundaries

Target observation times are host-observed upper bounds after 4,096-step chunks. A nominal-budget run can exceed its budget by one chunk. The raw trajectory and actual wall time are retained. Search stops when an ensemble has reached its target; unsuccessful ensembles remain right-censored. Complete verified wall time additionally includes final D2H and exact CPU checking of every walk's best. Allocation, initialization and a separate warm-up state are outside that timer.

Device-event totals sum actual kernel intervals; host search includes submission and rank observations. The implementation uses local stack storage as well as registers, and should not be described as a register-only reproduction of the paper. No energy measurement or full-system cost claim is made. Dynamic clocks, thermals and OS activity remain possible confounders; no clock or power-limit settings were changed and no second GPU research workload was intentionally run concurrently.

## Reproduction

CPU checks require Python 3.12 and its standard library. Original reference data is committed, so recreating it with NumPy/downloads is optional. Run without `-O`, because assertions are correctness gates.

```sh
python cpu_study.py --output build/cpu-check.json
python verify_composition.py
python verify_certificates.py
python verify_results.py
python summarize_results.py
```

The measured GPU stack uses Linux x86_64, CuPy 13.6.0, NumPy 2.2.6, a CUDA 12.x CuPy runtime/header environment and CUDA 13.3.73 nvcc for the CUBIN. Keep environments separate from unrelated applications.

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
python collect_build.py
python verify_certificates.py
python summarize_results.py
```

`record_environment.py` records non-identifying environment metadata in the same GPU environment. The importer scripts regenerate the separately attributed reference data; they never execute third-party research programs. CI runs CPU controls, retained certificate verification and artifact/derived-table checks. It does not rerun GPU discovery.

## Conclusion and further work

The completed campaign reproduces an exact rank-47 result on consumer hardware and shows that target attainment depends on the transition policy, not just raw transition rate. Group reductions at one probe are the strongest observed policy here, with broad uncertainty. Pooling at this interval does not help. Larger independent cohorts, better population scheduling, 64-bit factors, structured larger tensors and compiling discovered schemes into measured GF(2) kernels are distinct next studies. A GF(2) decomposition does not automatically lift to FP8/real arithmetic, a faster GEMM, or a permissionless proof-of-work protocol.

## Follow-up closure, 12 November 2026

The [follow-up report](FOLLOWUP_REPORT.md) resolves the concrete next steps with new evidence and updated conclusions. Its findings supersede the open-work wording in this historical campaign.
