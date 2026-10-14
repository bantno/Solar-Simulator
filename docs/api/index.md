# API reference

These pages are generated from the source code's docstrings by
[mkdocstrings](https://mkdocstrings.github.io/) with its Python handler. The prose pages
explain how the pieces fit together and why. These pages are the per-module,
per-class, and per-function reference, each with a link to its source.

| Section | Covers | Detail shown |
|---|---|---|
| [BaseClasses](baseclasses/index.md) | The core library: MDP, solver, transition models, weather providers, simulation, storage, config wiring, aircraft/panel/whale/clear-sky models, plotting helpers | All public members, source |
| [harness](harness.md) | `run_experiment`, `summarize`, `figures` | All public members, source |
| [Scripts](scripts.md) | Artifact builders, the chain-vs-iid sweep pipeline, persistence prechecks, inspection tools | Module docstring (usage) only |
| [Verification scripts](verify.md) | `Tests/verify_*.py` | Module docstring (what is checked, how to run) only |
| [Figure scripts](figures.md) | Journal-paper figure scripts in `Figures/Scripts/` | Module docstring only; `results_io` in full |

## How the pages are generated

- **Static analysis, nothing imported.** The handler uses
  [griffe](https://mkdocstrings.github.io/griffe/) to parse the source files. Modules are
  never executed, so building the wiki does not need `pvlib`, `h5py`, PyQt5, or data files.
  It also ignores import-time side effects such as `matplotlib.use(...)` and
  `sys.path.insert`.
- **Two import roots.** The code imports its packages as top-level names
  (`from BaseClasses.whale_base import ...`, `from harness import ...`), so
  **`SolarSimulator/` is the import root**, not the repository root. `Figures/Scripts/`
  uses bare sibling imports (`from results_io import ...`), so it is a second root. Both
  are listed under `plugins.mkdocstrings.handlers.python.paths` in `mkdocs.yml`.
  Identifiers are therefore written `BaseClasses.whale_base`,
  `harness.run_experiment`, `Scripts.build_windchain`, `Tests.verify_wind_chain`, and
  `results_io`.
- **Docstring style: Google.** Configured as `docstring_style: google`. Missing type or
  parameter warnings are switched off, so gaps in the docstrings don't fail the strict build.
- **Visibility.** Names starting with `_` are hidden (`filters: ["!^_"]`). Members without
  docstrings are still listed (`show_if_no_docstring: true`), so every public class and
  function is visible with its signature and source even where it is undocumented.
  Members appear in source order.
- **Page files are hand-listed.** Each page under `docs/api/` is a short Markdown stub
  containing a `:::` directive. There is no generation plugin, so the set of documented
  modules is explicit and reviewable.

## What is included and excluded

**Included:**

- every live module of `BaseClasses`
- the `harness` package
- the documented pipeline scripts in `Scripts/`
- all `Tests/verify_*.py`
- the documented journal-figure scripts in `Figures/Scripts/`

**Excluded:**

- Superseded copies kept for reference: `BaseClasses/*_old.py`.
- Modules without content or a parseable docstring.
- Undocumented one-off scripts in `Scripts/`, `Tools/`, `gui/`, `SolarArray/`,
  `TakeoffSimulation/`, and `Figures/Scripts/Combined/`. These are described in prose on
  [GUI and tools](../experiments/gui-and-tools.md), [Takeoff simulation](../models/takeoff.md),
  and [Journal figures](../experiments/figures.md).
- The `unittest` files `Tests/test_*.py`, which are catalogued in
  [Test catalogue](../verification/tests.md).

## Adding a module page

1. Create `docs/api/<area>/<module>.md` with a title, one sentence of purpose, and the
   directive:

    ```markdown
    # `BaseClasses.new_module`

    One-line purpose.

    ::: BaseClasses.new_module
        options:
          show_root_heading: false
    ```

    For a script where only the usage docstring matters, add it to the relevant grouped
    page with `members: false` and `heading_level: 3` instead.

2. Add the page to `nav:` in `mkdocs.yml`. The build is strict and warns on files missing
   from the nav.
3. Build with `conda run -n pvlib mkdocs build` and fix any warnings. The most common one
   is `Could not find cross-reference target`: text in square brackets inside a docstring,
   such as `[--verify]` or `rewards[a][b]`, is read as a Markdown reference link. Put
   command lines and indexing expressions in backticks.

## Docstring conventions that render well

- **Module docstring first.** Begin with a one-line summary, a blank line, then what the
  module is for and how to run it (for scripts). The first line is what appears in
  listings.
- **Google sections**: `Args:`, `Returns:`, `Raises:`, `Yields:`, `Attributes:`, `Note:`,
  `Example:`. Indent section bodies consistently. Name each argument exactly as it
  appears in the signature.
- **Code and commands in backticks or fenced blocks.** Docstrings are rendered as
  Markdown, so indented lines that follow a paragraph line are folded into that
  paragraph. For a multi-line usage block, leave a blank line before it and indent it by
  4 extra spaces, or use a ```` ``` ```` fence.
- **Math.** Docstrings are rendered with the site's Markdown extensions, including
  arithmatex, so a stray `$` pair gets typeset. Docstrings are also read in terminals and
  editors, so write formulas as code, for example `` `p = 1 - exp(-lambda)` ``.
- **Cross-references.** `[BaseClasses.whale_base.block_series][]` links to that object's
  API entry from any docstring or wiki page.
- **ASCII in printed output, Unicode allowed in docstrings.** The ASCII-only rule is for
  console `print` output (cp1252 consoles), not for documentation.
