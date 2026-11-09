# Research protocol

A research project by [Requant Network](https://requant.network).

## Required experiments

1. Implement independent exact tensor and matrix-evaluation verifiers; verify published factorizations and reject corrupted certificates.
2. Implement CPU flip, plus and generalized shared-factor reduction with invariant self-checks, then match controlled GPU transitions with CPU results.
3. Execute complete GPU random walks for 4x4 and a structured/rectangular extension. Compare fixed probing, adaptive probing and best-pool reseeding at matched budgets.
4. Retain all seeds, trajectories, target hit times, censored runs, verified best schemes and negative findings. Optimize time-to-target and success probability, not steps per second alone.
5. Reproduce the rank-47 4x4 target or report a bounded, documented failure after a serious search budget; validate every claimed best scheme independently. Do not claim a new world record or FP8 GEMM speedup without direct evidence.

## Shared evidence requirements

Use the released RTX 5060 Ti 16 GB without concurrent GPU experiments. Retain exact executed-source hashes, pinned upstream revisions, environment, raw measurements, independent correctness evidence, timing boundaries and reproduction commands. Fixed-input latency comparisons use five warm-ups and thirty measured repetitions where practical. Stochastic search uses independent seeds and retains unsuccessful/censored trials. Energy requires sampled measurements; no TDP-derived efficiency claims.

## Completion requirements

Publish executable CPU/GPU studies, all required comparisons, independent verification, raw results, derived tables, English final conclusions and limitations. Public repository topics, homepage and attribution must identify Requant Network. CPU/reference and artifact CI must pass at the published commit. A negative result may complete a hypothesis test; missing experiments may not.
