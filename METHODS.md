# Measurement and validation methods

A research project by [Requant Network](https://requant.network).

The [final report](FINAL_REPORT.md) specifies the tensor convention, transformations, policy schedules, timing boundaries and reproduction commands. The [environment record](results/environment.json) identifies the actual device, CPU, compiler, CUDA APIs and package versions.

CPU controls check full tensor identities and independently evaluate matrix products. GPU correctness is established by intermediate tensor checks, controlled GPU/CPU transition comparisons, and exact verification of every walk-best during the search. Archived ensemble-best certificates are reverified by CPU-only CI. Verification does not rely on a floating-point tolerance or random products alone.

Each cohort has eight independent seeded ensembles per shape/policy; each ensemble contains 256 pseudo-random walks. Policies share initial seeds. Pool walks share elites and are not independent statistical samples. A nominal five-second budget can overshoot by one observed GPU chunk. Target observations are upper bounds, and unsuccessful runs remain censored. Source and raw-data hashes bind the reported experiment to its executed implementation.

Success counts and Wilson intervals are descriptive estimates for this finite campaign. Successful-only time medians are conditional and cannot represent unconditional time-to-solution. The derived restricted-time score caps target observations at the nominal budget and assigns the full budget to failures. Cohorts remain separate in the CSV.

Complete verified wall time includes final D2H and exact CPU certificate checks. Planning, allocation/initial state and a separate warm-up state are excluded. Kernel event totals are separately retained. Dynamic clocks, thermals, finite PRNG streams and OS activity limit extrapolation. No clock/power settings were changed, and no joules, mining hardness or profitability claim is made.
