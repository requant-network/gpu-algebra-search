# Larger search cohorts, compiled schemes and wide factors

Requant Network, 12 November 2026. This supplement resolves the concrete follow-ups from the [historical report](FINAL_REPORT.md). A lower tensor rank is checked independently before it is treated as a result; hardware speed is a separate question.

## Population policy: negative result

Two independent cohorts each compare five policies on 4x4x4: 32 trials per policy, 256 walks per trial and a five-second search budget. The same CUDA transition kernel is retained. Trial order alternates and policies share trial seeds; the new `diverse8` policy keeps up to eight canonical-distinct elite term sets and cycles copies into the worst quarter. Permuted copies of one tensor decomposition do not count as distinct elites. All final walk certificates are checked exactly, every retained best tensor is replayed by the CPU checker, and unsuccessful trials remain censored at their budget.

| Cohort | fixed1 | generalized1 | adaptive2 | single-elite pool | distinct-elite pool |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20261112 | 4/32 | 5/32 | 7/32 | 3/32 | 1/32 |
| 20261212 | 9/32 | 12/32 | 9/32 | 5/32 | 1/32 |

Each cell counts trials reaching rank 47. Distinct-elite copying did not improve this bounded search; it was worse in both cohorts. Retain the simpler policies rather than treating population diversity as an established improvement. Rank 47 reproduces a known bound; no new matrix-multiplication record was found. The two cohorts show substantial success-count variation, so these counts are not a universal policy ranking or an extrapolated time-to-solution theorem.

An exploratory post-hoc paired analysis uses all 64 distinct trial seeds, preserving within-trial matching. It applies an exact two-sided discordance/binomial test and Holm correction over the four new-candidate/control comparisons; censored failures remain failures at the declared budget.

| Control | Control-only successes | Diverse-only successes | Exact p | Holm-adjusted p |
| --- | ---: | ---: | ---: | ---: |
| fixed1 | 12 | 1 | 0.00341797 | 0.00683594 |
| group1 | 17 | 2 | 0.00072861 | 0.00218582 |
| adaptive2 | 15 | 1 | 0.00051880 | 0.00207520 |
| pool1 | 8 | 2 | 0.10937500 | 0.10937500 |

This exploratory table supports the bounded negative decision against preferring `diverse8` to fixed/group/adaptive controls in these trials; the single-elite-pool contrast is not established at the same conventional threshold. It is not a pre-registered confirmation or a universal ranking. [policy_statistics.py](policy_statistics.py), [raw paired analysis](results/policy-statistics.json) and [derived table](results/policy-summary.csv) preserve the exact calculation and input bindings; CI regenerates both outputs.

## Compiling rank 47: fewer multiplications is slower here

The [discovered rank-47 fixture](fixtures/discovered-rank47.json) is compiled into a straight-line GF(2) 4x4 scalar product. The independent direct comparator computes row/column parities. An exhaustive GPU check of every A bit pattern against each basis B tests 1,048,576 pairs, exercising the complete tensor identity. Each full random batch is also compared with an independent CPU bit-product oracle. Damaged tensor certificates reject.

Both cohorts have five warm-ups and thirty alternating measurements for batches of 4,096, 262,144 and 1,048,576 products. These are complete scalar products, not a blocked large matrix GEMM.

| Seed | Products | Direct resident median, ms | Rank-47 resident, ms |
| --- | ---: | ---: | ---: |
| 20261112 | 4096 | 0.00472 | 0.00531 |
| 20261112 | 262144 | 0.01106 | 0.02366 |
| 20261112 | 1048576 | 0.02858 | 0.07690 |
| 20261212 | 4096 | 0.00502 | 0.00528 |
| 20261212 | 262144 | 0.01181 | 0.02326 |
| 20261212 | 1048576 | 0.03117 | 0.07397 |

The compiled decomposition is slower at the largest batch in both cohorts, roughly 2.4–2.7 times the direct kernel. Its extra mask/parity/addition work outweighs the lower bilinear multiplication count. The evidence rejects using rank alone as a GPU performance proxy. Transfer-inclusive measurements are retained separately in [followup-summary.csv](results/followup-summary.csv).

## Wide-factor qualification

[wide_study.py](wide_study.py) makes factor masks 64-bit while preserving 32-bit warp ballots, lane masks and RNG state. It records exact generated-source transformations, verifies 32/64 transition identity on the original 4x4 case, and checks wide tensor snapshots and all final walk certificates on 6x6x6 and 8x4x8. The second shape exercises factor bit 63. The bounded search target is an improvement over naive rank, not a claim to a competitive or novel decomposition for those shapes. Both cohorts passed 800 identical 32/64 transition snapshots and 400 exact wide tensor snapshots, including bit 63. Each cohort ran eight trials per policy and shape, with 128 walks and a two-second budget. One primary adaptive 6x6x6 trial found rank 215 from naive 216; the other 47 primary cases and every replication case retained their naive best ranks (216 or 256). This qualifies the wider representation and exact transitions but provides no replicated search-quality improvement. All 128 final walk certificates in every case passed the independent tensor check; the 96 retained best schemes are replayed by CPU CI.

## Reproduction and decision

Run [compile_scheme.py](compile_scheme.py), [compiled_study.py](compiled_study.py), [diversity_study.py](diversity_study.py) and [wide_study.py](wide_study.py) in the documented CUDA/CuPy environment; each `--help` exposes cohort seed and output. The diversity protocol uses `--trials 32 --seconds 5 --variants fixed1,group1,adaptive2,pool1,diverse8 --shapes 4,4,4`; repeat with seed 20261212 and distinct replication outputs. `python verify_followups.py` checks every added source binding, exact tensor certificate, altered-certificate rejection, censor flag and fixed-input sample count.

The useful outcome is a qualified search/compilation tool with concrete negative decisions: do not prefer distinct-elite copying on this evidence, and do not claim faster GPU multiplication from rank 47. Larger searches can be designed as new bounded experiments; this campaign does not imply hardware-independent optimality, cryptographic work value, energy efficiency or heterogeneous multi-GPU scaling.

## Server timing boundary

GPU campaigns run serially on the same RTX 5060 Ti. Clocks, power limits and unrelated server services are unchanged. Measurements describe this server and retained workload; small percentage differences are not treated as universal improvements. No energy measurement or heterogeneous multi-GPU throughput is inferred.
