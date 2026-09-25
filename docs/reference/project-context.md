# Project context

This page holds the project background that the code cannot tell you. It contains only
what the repository states. Everything else is marked for the project owner to fill in.

## What the repository states

From the top-level `README.md` on `main`:

- **AHABS** is the Autonomous Hydrofauna Aerial Biome Surveyor, a solar-powered seaplane
  designed for long-term oceanic monitoring, for example whale observation.
- The aircraft spends most of its time **moored on the water charging** from its solar
  array, and periodically **takes off to fly and observe whales**.
- The simulator models this as a finite-horizon **MDP**. Its environment is driven by
  location- and season-specific wind, solar, and whale distributions. It compares
  energy-management policies (optimal vs threshold) for maximizing observations over a
  mission.
- Development is on Windows, Python 3.11, conda env `pvlib`. The license is **TBD**.

From the design records:

- The i.i.d. model is the one **published in the Journal of Aircraft**, and every
  extension preserves it exactly as a special case
  ([wind-persistence briefing](../design/wind-persistence-briefing.md)).
- The repository tag `v1.0.0` (2025-11-19) is annotated "AIAA Journal of Aircraft
  Submission". It marks the state of the code at the journal submission.
- The whale-observation work is framed around a surface platform that can also listen
  acoustically while moored
  ([literature survey](../design/whale-observation-literature.md)).

!!! todo "Owner input needed: mission narrative"
    The operational concept, including target region(s), deployment duration, and how the
    aircraft is recovered and maintained. Also the scientific motivation (which species,
    which survey questions) beyond "long-term oceanic monitoring, e.g. whale observation",
    and how the simulator's results feed hardware or mission decisions.

!!! todo "Owner input needed: people, funding, and publications"
    Collaborators, lab or advisor, funding sources, and the citation list. That includes
    the full reference for the Journal of Aircraft paper and any thesis chapters that use
    this code.

!!! todo "Owner input needed: aircraft parameter provenance"
    Where the seaplane's mass, power draws (idle, cruise, takeoff, landing), climb and
    descent parameters, battery options, and solar-array area and efficiency come from:
    measured, from datasheets, or assumed. The values themselves are documented from code
    in [Aircraft power model](../models/aircraft-power.md) and
    [Solar panel](../models/solar-panel.md). Their sources are not recorded in the
    repository.

## Roadmap

!!! todo "Owner input needed: roadmap and priorities"
    Planned work, priorities, and timelines beyond the next steps the design records
    already state (listed below).

Next steps and open questions **already recorded** in the design records:

- **Joint wind + solar regime chain.** The two chains are currently independent
  (Kronecker product). A fitted joint chain is "the designed next step". Storms are both
  windy and cloudy. ([Solar persistence](../design/solar-persistence-math.md) §7;
  [briefing](../design/wind-persistence-briefing.md) §5)
- **Principled wind-bin selection.** Decision-aware edges or predictive-information
  criteria are "only worth pursuing if a site shows quantile bins still leaving reward
  on the table". ([Threshold benchmark](../design/wind-threshold-benchmark.md) §9)
- **Whale observation, Tier 2.** An acoustic channel while moored, `p_ac(t, wind)`,
  earned in mode 0 as reward only. Tier 3 (acoustic detection as information, MOMDP) only
  if Tier 2 changes decisions.
  ([Literature survey](../design/whale-observation-literature.md) §4)
- **Whale observation, open questions:**
    - always-on vs switchable listening
    - a luminance proxy for visual detection
    - site-specific acoustic detection radius
    - the relative worth of acoustic vs visual detections
    - diurnality of the acoustic cue rate

    ([Literature survey](../design/whale-observation-literature.md) §6)
- **Occupancy before MOMDP.** Build an occupancy + detection model first. Before that,
  decide whether the sighting data has repeat-visit structure, and how coarse the hidden
  state should be. ([Model extensions](../design/whale-observation-extensions.md) §5)
