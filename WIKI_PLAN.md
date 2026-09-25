# WIKI_PLAN — Solar-Simulator reference wiki (MkDocs Material)

> Once you approve this, the plan is first copied verbatim to `WIKI_PLAN.md` at the repo root. Implementation then runs on a new branch `docs-wiki`, cut from `observation-model-fidelity` with the working tree as it stands.

## 1. Context

**Goal.** A private, local, searchable reference wiki in the style of the PX4 guide. It is for the research team, future maintainers and AI agents. Readers are assumed to know MDPs and solar/aero basics.

**What exists today:**
- **Design docs.** Eight good design docs, about 22k words, in `SolarSimulator/docs/`:
  - Wind persistence: `wind_persistence_{math,theory,validity,briefing,threshold_benchmark}.md`, plus `theory.pdf`.
  - Solar persistence: `solar_persistence_math.md`.
  - Whale observation: `whale_observation_{literature,extensions}.md`.
  - 12 figures in `docs/figures/`.
  - The math is written in `$…$` / `$$…$$`.
- **Harness README.** `SolarSimulator/harness/README.md` is the de facto user guide: CLI, output layout, config schema, `summary.csv`.
- **Top-level README.** The current one is only on `main` (commit `71e81d3`). This branch still has a stale version (`requirements.txt`, `run_simulation.py`, `.pkl` outputs).
- **Docstrings.** Google style throughout. Excellent in `harness/*`, `whale_base`, `clearsky_models`, the `Scripts/build_*`, the chain-sweep scripts and `Tests/verify_*`. Patchy in the core `BaseClasses` solver, provider and simulation modules.
- **Recorded rationale.** From June 2026 onward, commit bodies record the reasons for decisions: `59939eb`, `32e0d4a`, `8356b5a`, `3598426`, `81e3d4d`, `76812fa`, `f281dca`. Earlier commits are one-liners.

**What is scattered or missing:**
- **No architecture page and no data-flow page.** The flow is config → `YAMLSimulationRunner` → `SimulationFactory`/`EnvironmentLoader` → `mdpAnalyticalBackwardSolver.solve()` → `SimulationRunManager` → HDF5 → `summarize`/`figures`.
- **No config key reference.** The key meanings and defaults live only in `.get()` calls in `run_sim.py` and in YAML comments.
- **Undocumented core models.** No write-up of the aircraft power model (`seaplane_base`), the transition-probability models (`transition_model_base`), the takeoff simulation, the weather-data pipeline (Open-Meteo → fitted distributions → artifacts), the GUI or the tools.
- **Environment gotchas** exist only in memory notes and a git-ignored handoff file: use the `pvlib` env, the numpy/h5py ABI issue in base, cp1252 stdout, and `conda run -c` breaking multiprocessing spawn.

**Decisions from the interview:**
- **Audience:** team plus agents. Include an agent orientation page.
- **Existing docs:** `git mv` them into `docs/`, leaving pointer stubs behind.
- **Branch state:** document the current working tree. The uncommitted encounter whale model gets an "in progress" admonition. The overview is sourced from `main`'s README.
- **mkdocs packages:** go into `environment.yml`, which is re-saved as UTF-8.
- **API pages:** all four areas: `BaseClasses`, `harness`, `Scripts` + `Tests/verify_*`, and `Figures/Scripts`.
- **Page depth:** every page is written in full in v1.
- **Known defects:** omitted from the wiki. The config reference still states facts neutrally, e.g. that `energy_increment_wh` is required.
- **Do not invent** (write a `!!! todo "Owner input needed"` admonition instead):
  - project/mission context beyond the README
  - where the aircraft parameters come from
  - roadmap and open questions beyond what the docs already state

## 2. Tooling

**Packages.** Added to `environment.yml` under `pip:`. The file is converted UTF-16 → UTF-8 and the absolute `prefix:` line is dropped.
- `mkdocs>=1.6,<2`
- `mkdocs-material>=9.5,<10`
- `mkdocstrings[python]>=0.26,<1`
- `pymdown-extensions>=10`

The same packages are also installed into the `pvlib` env with pip.

**`mkdocs.yml`** (repo root):
- **Build settings:** `docs_dir: docs`, `site_dir: site`, `strict: true`, `use_directory_urls: false`.
- **Validation:** `validation: {links: {not_found: warn, anchors: warn, absolute_links: warn, unrecognized_links: warn}, nav: {omitted_files: warn}}`. With `strict`, any broken link or anchor fails the build.
- **Plugins:**
  - `search`
  - `offline`
  - `mkdocstrings`, python handler:
    - Import root: `paths: [SolarSimulator, Figures/Scripts]`, because modules import `BaseClasses.*` directly.
    - Docstrings: `docstring_style: google`, `docstring_options: {warn_unknown_params: false, warn_missing_types: false}`, so gaps in docstrings don't fail the strict build.
    - Display: `show_source: true`, `members_order: source`, `show_if_no_docstring: true` for the core modules, `filters: ["!^_"]`, `heading_level: 2`, `show_root_heading: true`.
    - Static analysis only. Nothing is imported, so heavy dependencies never load.
- **Theme** `material`:
  - `font: false`, which stops the Google Fonts fetch that would break offline use.
  - Palette toggle between light and dark, with `scheme: default` / `slate`.
  - Features: `navigation.tabs`, `navigation.sections`, `navigation.indexes`, `navigation.top`, `content.code.copy`, `search.highlight`, `toc.follow`.
- **Markdown extensions:**
  - `admonition`, `pymdownx.details`, `tables`, `attr_list`, `md_in_html`
  - `pymdownx.superfences` with the mermaid custom fence
  - `toc: {permalink: true}`
  - `pymdownx.snippets: {base_path: [.], check_paths: true}`
  - `pymdownx.arithmatex: {generic: true}`, needed because the design docs contain LaTeX
  - `pymdownx.highlight`, `pymdownx.inlinehilite`
- **Offline assets.** Mermaid and MathJax normally load from a CDN, which fails from `file://`, so both are vendored:
  - `docs/assets/javascripts/mermaid.min.js`. Material uses `window.mermaid` when it is already present.
  - `docs/assets/javascripts/tex-svg.js`, the self-contained MathJax 3 build, plus a small `mathjax-config.js`.
  - Both are listed in `extra_javascript`.

**`.gitignore`.** Add `site/`, plus lowercase `results/` so `results/` is still ignored on case-sensitive systems.

## 3. Site tree

Each line gives a page and its one-line purpose. **[M]** marks a moved existing doc, **[G]** a page generated from source, **[H]** a hand-written page.

```
docs/
  index.md                         [H] What AHABS/Solar-Simulator is, the one-paragraph model, map of the wiki (from main README)
  getting-started/
    index.md                       [H] Install: conda env `pvlib`, environment.yml, package root, building the wiki
    first-run.md                   [H] Run iid_small.yaml end to end; what lands in results/; reading summary.csv
    agent-orientation.md           [H] For AI agents: import root = SolarSimulator/, env gotchas, ASCII stdout, conventions (Google docstrings, verify_* pattern, perf-work-on-branch, no auto-reuse of value tables, disk budget, commit style)
  architecture/
    index.md                       [H] Component map + mermaid data-flow diagram (config → solve → rollout → HDF5 → summary/figures)
    mdp.md                         [H] State (SoC × mode × exo bins), actions, reward, failure, stage indexing; AbstractMDP/stochasticMDP
    solver.md                      [H] Backward induction: _solve_iid vs _solve_chain, value-table shapes, vectorization, solver_tables/*.npy
    simulation.md                  [H] Episode rollout, batch simulation, seeding (reset(0) pairing), SimulationRunManager parallelism
    storage.md                     [H] HDF5 layout (columnar episode_scalars vs episodes/ groups), group naming, dual-format readers, run folder layout
  models/
    index.md                       [H] Feature matrix: every model switch, its default, and "default = original model" guarantee
    aircraft-power.md              [H] Seaplane power/climb model, get_mdp_power_params, maneuver times (param provenance → TODO)
    takeoff.md                     [H] TakeoffSimulation: Gudmundsson takeoff + motor models
    transition-models.md           [H] ProbabilityModelFactory: moderate/linear/nofail/realistic/optimistic/... failure-probability curves
    solar-panel.md                 [H] constant / pvwatts / variable panels, registry, PVWatts equivalence to constant
    clearsky.md                    [H] fatemi / haurwitz / ineichen, artifact suffixing, normalization
    weather-data.md                [H] Open-Meteo fetch → Beta/Weibull fits → EXPECTED_DATA pkl; build_* artifacts (windchain, solarchain, histcube)
    weather-modes.md               [H] iid vs chain vs historical bootstrap; solve-on-fitted / evaluate-on-historical
    wind-chain.md                  [H] Summary page for Markov-modulated wind: bins (quantile vs edges), within-bin Weibull, links to design records
    solar-chain.md                 [H] Summary page for clear-sky-index chain: stage-quantile bins, dusk→dawn matrix, kron joint index
    whale-observation.md           [H] Series types (JoA2026/custom/encounter/constant/sinusoidal), block series alignment, encounter p=1−exp(−D·A·a·g0·light), cruise-fraction gain ("in progress" admonition for encounter)
    policies.md                    [H] Optimal (solved) vs threshold policy rules; how each is instantiated
  experiments/
    index.md                       [M] harness/README.md → the harness guide (CLI, flags, outputs)
    config-reference.md            [H] Every YAML key: meaning, default, code location; sweep semantics (lists → cross-product)
    examples.md                    [G] Each harness/examples/*.yaml embedded via snippets with a one-line purpose
    sweeps.md                      [H] chain-vs-iid sweep pipeline: generate → run → compare(--verify) → plot; final_journal_configs
    outputs.md                     [H] summary.csv columns, run_metadata.json, figures/, _analysis/
    figures.md                     [H] Journal figure pipeline (generate_journal_paper_figures, paper_figure key, results_io)
    gui-and-tools.md               [H] gui/ (runner GUI, results viewer, run_batch), Tools/, SolarArray, plotting scripts
  verification/
    index.md                       [H] Verification philosophy (reduce-to-baseline then directional checks); how to run verify_*
    tests.md                       [H] What each verify_* and test_* checks (from their docstrings)
  design/                          (design records; moved docs, figures moved to design/figures/)
    index.md                       [H] Index of design records + decision log (date, decision, reason, commit) mined from commit bodies
    wind-persistence-math.md       [M]
    wind-persistence-theory.md     [M] (+ .pdf linked as download)
    wind-persistence-validity.md   [M]
    wind-persistence-briefing.md   [M]
    wind-threshold-benchmark.md    [M]
    solar-persistence-math.md      [M]
    whale-observation-literature.md[M]
    whale-observation-extensions.md[M]
  findings.md                      [H] Results summary sourced only from repo docs/_analysis reports (threshold benchmark, chain-vs-iid), each claim linked to its source
  reference/
    glossary.md                    [H] Symbols and terms (SoC, exo index, CMI, K clear-sky index, ψ·a·g, …)
    project-context.md             [H] TODO skeleton: mission, collaborators, roadmap (owner input)
  api/
    index.md                       [G] How the API pages are generated; import root note
    baseclasses/<module>.md        [G] one page per core module (mdp_base, backward_induction_base, transition_model_base,
                                       environment_provider_base, simulation_base, simulation_run_manager, simulation_storage,
                                       run_sim, seaplane_base, solar_panel_base, whale_base, clearsky_models, weather_processor*,
                                       plot_base, plotting_utils_base); excludes *_old.py, empty plotting_base, utc_clearsky (no docstring)
    harness.md                     [G] run_experiment, summarize, figures
    scripts.md                     [G] Scripts/* — module docstrings only (members hidden), grouped by pipeline stage
    verify.md                      [G] Tests/verify_*.py module docstrings
    figures.md                     [G] Figures/Scripts/* module docstrings + results_io members (Combined/ excluded)
```

**Moves:**
- `SolarSimulator/docs/*.md`, `figures/` and the `.pdf` go to `docs/design/`.
- `SolarSimulator/harness/README.md` goes to `docs/experiments/index.md`.
- Both old locations get a short `README.md` pointer to the new path.
- Two in-code references are updated to point at the new locations: `harness/__init__.py:8` and `run_experiment.py:418`, which currently say "See harness/README.md". So is the comment in `whale_encounter_small.yaml:28`.
- In the moved docs, relative links to source files are changed to inline code paths, or to API-page cross-refs like `[BaseClasses.whale_base][]`. This keeps the strict link check green.

**`precheck_out/`.** Its tracked PNGs are referenced only if the moved docs already embed them. In that case they are copied, not moved, because scripts write to that folder.

## 4. Generated vs hand-written

- **Generated from source:**
  - The API pages, via mkdocstrings `::: module` directives.
  - The example configs, via snippets `--8<-- "SolarSimulator/harness/examples/x.yaml"`.
  - Commands quoted in the verification pages are copied from the scripts themselves.

  The API pages are hand-listed directive stubs, one per module. That keeps them deterministic and needs no gen-files plugin.
- **Hand-written in full in v1:** every [H] page. The only exception is the owner-input TODO sections listed in §1.
- **Sources for the hand-written pages:**
  - the code itself: `run_sim.py` for the config reference, `seaplane_base` for the power model, and so on
  - the docstrings
  - the commit bodies listed in §1
  - `main:README.md`
  - `results/*/_analysis/comparison_report.md`, used only to cite findings, never to generate new numbers

## 5. Execution order

1. Create branch `docs-wiki` and write `WIKI_PLAN.md`.
2. Tooling:
   - Convert `environment.yml` to UTF-8 and add the pins.
   - Install the packages into `pvlib`.
   - Update `.gitignore`.
   - Write `mkdocs.yml` and the vendored JS.
   - Add a minimal `docs/index.md`, then build to confirm the toolchain works.
3. `git mv` the existing docs and figures, fix their links, and update the in-code references. Build strict.
4. API pages: add them one package at a time, building strict after each. Tune the griffe options until there are zero warnings.
5. Architecture and getting-started pages, including the mermaid diagram.
6. Models pages.
7. Experiments pages: config reference, examples, sweeps, outputs, figures, GUI and tools.
8. Verification, design index with the decision log, findings, glossary and project-context TODO.
9. Final strict build, the checks in §6, then a commit on `docs-wiki`. No push.

## 6. Verification

- `conda run -n pvlib mkdocs build --strict` exits with status 0 and no warnings. Because link and anchor validation are set to warn under `strict`, this also proves there are no broken internal links or anchors.
- Open `site/index.html` directly from disk with `file://` and check:
  - navigation tabs and sections
  - search returns hits (for example "cruise fraction" and "kron")
  - a mermaid diagram renders
  - MathJax renders on `design/wind-persistence-math.html`
  - the light/dark toggle
  - code copy buttons
  - the browser's network panel shows no external requests
- Spot-check the API pages for `BaseClasses.whale_base` and `harness.run_experiment`. They should render members and source without importing pvlib or h5py; running in a clean env with only mkdocs installed is optional.
- Grep `docs/` for leftover `SolarSimulator/docs/` or `harness/README.md` references, and confirm the pointer READMEs resolve.
- Run `git status` to confirm `site/` is ignored and that no source `.py` changed apart from the docstring and comment pointers in §3.
- Run one `verify_whale_model.py` as a sanity check that the moved docs and the edited comments broke nothing.
