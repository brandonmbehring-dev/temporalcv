# Critique of Documentation Quality Assessment & Enhancement Plan (codex)

## Scope and method
- Audited the plan in `/home/brandon_behring/.claude/plans/declarative-dancing-quiche.md` for assumptions, methodology, benchmarking, testing, notebooks, and process risk.
- I did not validate repository facts claimed in the plan (scores, counts, or module coverage). Treat those as hypotheses to verify.

## Strengths (brief)
- Clear structure and sequencing, with a sensible front-load of critical documentation gaps.
- Good attention to developer onboarding, discoverability, and user experience (toctree, model cards, troubleshooting).
- Acknowledges missing API reference coverage and suggests concrete fixes.

## Findings (ordered by severity)

### High
1) Unsubstantiated scoring and readiness claims
- Evidence: numeric scores ("Overall Documentation Quality: 8.6/10", "Sphinx Documentation: 4.5/5"), and the conclusion "v1.0 readiness: YES".
- Why this matters: without a rubric and evidence, the scores create false confidence and can mask critical gaps. It also makes progress tracking subjective.
- Fix: define a rubric aligned to a doc framework (e.g., Diataxis) with explicit criteria and acceptance thresholds, and attach evidence for each score. Include a "verification checklist" (doc build, link checks, API drift scan, notebook run status).
- References: Diataxis documentation framework, https://diataxis.fr/

2) Benchmarking plan is methodologically under-specified
- Evidence: Block 3 lists datasets and plots but lacks details on split protocol, seeds, repetitions, noise quantification, and acceptance criteria.
- Why this matters: benchmark numbers will be noisy and potentially misleading. Without a protocol, performance claims are not reproducible and comparisons may be unfair.
- Fix: define a benchmarking protocol (rolling-origin evaluation, fixed seed policy, repeated runs, and statistical summaries like median and IQR). Separate microbenchmarks (algorithmic complexity) from end-to-end evaluations (pipeline cost). Include a regression budget and a reproducibility section.
- References: Bergmeir and Benitez 2012 (time series CV), https://doi.org/10.1016/j.ins.2012.12.028; Hyndman and Athanasopoulos (forecast evaluation), https://otexts.com/fpp3/; ASV (benchmarking tool), https://asv.readthedocs.io/en/stable/

3) Testing plan does not cover correctness risks that matter most
- Evidence: the plan proposes a testing guide but does not identify edge cases, integration tests, or test strategy for stochastic gates and time-series leakage.
- Why this matters: temporal validation is fragile; missing tests for leakage, irregular sampling, short series, and stochastic behavior will lead to false safety signals.
- Fix: add a test matrix that targets known failure modes: leakage via feature engineering, nonstationarity shifts, boundary conditions (short series, constant series, missing values), and reproducibility for stochastic gates. Include a minimum set of deterministic integration tests covering end-to-end pipelines.
- References: Diebold and Mariano 1995 (predictive accuracy testing), https://doi.org/10.1080/07350015.1995.10524599; Harvey et al. 1997 (small-sample DM adjustment), https://doi.org/10.1016/S0169-2070(96)00763-4

4) Scope creep risks derail the documentation goal
- Evidence: Block 4 proposes conda-forge and Docker production infrastructure as part of a documentation quality plan.
- Why this matters: packaging and infrastructure are large, ongoing commitments. They can consume most of the schedule and dilute focus on documentation correctness and user safety.
- Fix: separate the roadmap: deliver doc fixes and testing improvements first, then evaluate packaging as an explicit, scoped follow-up with owner/timebox.

### Medium
1) "Single marathon session" plan increases error risk
- Evidence: a 12-14 hour continuous session with large, context-switching tasks.
- Why this matters: high cognitive load increases mistakes, especially on docs and benchmarks where precision matters.
- Fix: break into smaller, timeboxed milestones with validation gates. Tie each block to a measurable acceptance check (docs build, notebook run, benchmark reproducibility).
- Reference: Sandve et al. 2013 (reproducible research emphasizes incremental, verifiable steps), https://doi.org/10.1371/journal.pcbi.1003285

2) Dataset and licensing assumptions are implicit
- Evidence: use of M4, FRED-MD, Tourism without mention of dataset licenses, download size, or versioning.
- Why this matters: datasets can have usage restrictions, change over time, or require manual access, which makes benchmarks brittle.
- Fix: document dataset provenance, licensing, and version checksums; provide a small, cached subset for CI smoke tests.
- References: Pooch (data download and versioning), https://www.fatiando.org/pooch/latest/

3) Model cards are treated as a docs navigation fix, not a content requirement
- Evidence: "Model cards not in toctree" is noted, but no content standard is specified.
- Why this matters: model cards are valuable only if they follow an accepted template (intended use, limitations, evaluation data, and ethical considerations).
- Fix: add a model card template and a checklist for required fields.
- Reference: Mitchell et al. 2019 (Model Cards), https://arxiv.org/abs/1810.03993

### Low
1) Documentation validation lacks CI enforcement
- Evidence: "make html" is the only validation step mentioned.
- Why this matters: docs regress silently without CI gating; warnings become permanent.
- Fix: run Sphinx with warnings-as-errors and include link checks in CI (nightly if too heavy).

2) Notebook quality assurance is not specified
- Evidence: many notebooks exist, but no plan for executing or validating outputs.
- Why this matters: notebooks drift quickly; stale outputs mislead users.
- Fix: add notebook execution checks in CI for a small subset, and a scheduled job for full runs.
- References: nbval (pytest for notebooks), https://nbval.readthedocs.io/en/latest/; papermill (parameterized notebooks), https://papermill.readthedocs.io/

## Unstated assumptions and verification gaps
- The module coverage numbers (11/16, 69%) and counts of notebooks/guides are correct; they need a reproducible source (e.g., a script or `rg`/`cloc` output) and should be re-checked.
- "Excellent" docstring quality is assumed without sampled evidence or a rubric; consider sampling + rubric-based scoring aligned to Numpy docstring standards.
- The plan assumes all optional extras map cleanly to user tasks, but does not consider dependency conflicts or extra install time.
- The plan assumes that adding model cards and changelog to toctree is sufficient for discoverability; it does not consider search indexing or cross-links.
- The plan assumes a single hardware profile ("64-core Threadripper") is representative; it should not be the only reported result.

## Benchmarking and testing improvements with options (pros/cons)

### Option A: Use ASV for performance benchmarks
- Pros: standardized methodology, tracks regressions over versions, integrates with Git for historical baselines.
- Cons: setup time, learning curve, CI integration overhead.
- Reference: https://asv.readthedocs.io/en/stable/

### Option B: Use pytest-benchmark for microbenchmarks + custom scripts for macrobenchmarks
- Pros: simple to adopt, integrates with pytest, good for unit-scale timing.
- Cons: less ideal for long-running end-to-end benchmarks; still needs a macrobenchmark harness.
- Reference: https://pytest-benchmark.readthedocs.io/en/latest/

### Option C: Custom benchmark harness only (current plan)
- Pros: full control over protocol and output format.
- Cons: easy to get methodology wrong; harder to compare over time; higher maintenance burden.

## End-to-end test strategy (recommended)
- CI smoke tests: tiny synthetic datasets that exercise full pipelines (gates + CV + metrics). Deterministic seeds and fixed splits.
- Nightly or manual tests: real datasets with rolling-origin evaluation and multiple seeds to quantify variance.
- Edge-case suite: short series, constant series, missing values, irregular timestamps, multi-series, and nonstationary drift.
- Statistical tests: verify expected behavior on known synthetic regimes; include DM test adjustments for small samples.
- References: Bergmeir and Benitez 2012 (time-series CV), https://doi.org/10.1016/j.ins.2012.12.028; Diebold and Mariano 1995, https://doi.org/10.1080/07350015.1995.10524599

## Documentation methodology improvements (with pros/cons)

### Option A: Diataxis-based doc audit rubric
- Pros: clear separation of tutorial/how-to/reference/explanation, easier to assign owners.
- Cons: requires a one-time taxonomy effort.
- Reference: https://diataxis.fr/

### Option B: Minimal rubric based on Numpydoc + CI checks
- Pros: quicker to implement, aligns with common scientific Python standards.
- Cons: less comprehensive for pedagogy and narrative documentation.
- Reference: https://numpydoc.readthedocs.io/en/latest/format.html

## Suggested acceptance criteria (make progress measurable)
- Docs build with `-W` (warnings as errors); all toctrees are complete; no missing references.
- Notebook status: at least 80% of notebooks run end-to-end in a reproducible environment; all notebooks have deterministic seeds.
- Benchmark protocol documented (split scheme, datasets, seeds, hardware, repetitions) with a reproducibility checklist.
- End-to-end tests run in CI for a minimal pipeline, with at least one synthetic and one real dataset.
- API drift: every public symbol in docs is verified by an automated check against the codebase.

## References
- Bergmeir, C., and Benitez, J. M. 2012. "On the use of cross-validation for time series predictor evaluation." https://doi.org/10.1016/j.ins.2012.12.028
- Diebold, F. X., and Mariano, R. S. 1995. "Comparing predictive accuracy." https://doi.org/10.1080/07350015.1995.10524599
- Harvey, D., Leybourne, S., and Newbold, P. 1997. "Testing the equality of prediction mean squared errors." https://doi.org/10.1016/S0169-2070(96)00763-4
- Hyndman, R. J., and Athanasopoulos, G. "Forecasting: Principles and Practice." https://otexts.com/fpp3/
- Mitchell, M., et al. 2019. "Model Cards for Model Reporting." https://arxiv.org/abs/1810.03993
- Numpydoc format standard. https://numpydoc.readthedocs.io/en/latest/format.html
- Diataxis documentation framework. https://diataxis.fr/
- ASV (Airspeed Velocity) benchmarking tool. https://asv.readthedocs.io/en/stable/
- pytest-benchmark. https://pytest-benchmark.readthedocs.io/en/latest/
- Sandve, G. K., et al. 2013. "Ten Simple Rules for Reproducible Computational Research." https://doi.org/10.1371/journal.pcbi.1003285
- nbval. https://nbval.readthedocs.io/en/latest/
- papermill. https://papermill.readthedocs.io/
- Pooch (data download and versioning). https://www.fatiando.org/pooch/latest/
