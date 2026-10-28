# Navigation annotation algorithm overview

The pipeline uses existing map objects, rooms, and traversable cells to generate unique English references, navigation tasks, hard/soft constraints, and reference trajectories. It combines rules, templates, and path search; it does not detect objects from images or use an LLM to generate annotations.

## Five modules

| File | Responsibility |
|---|---|
| `hierarchical_descriptions.py` | Build hierarchical rules, resolve unique objects, render variants, and classify difficulty. |
| `build_train_annotations.py` | Coordinate object annotation, task generation, description allocation, and export. |
| `description_variants.py` | Assign unused descriptions, favoring different references and rules. |
| `annotate_navigation.py` | Select accessible targets and generate ordered constraints and reference paths. |
| `verify_navigation.py` | Check grounding, duplicate text, visit order, path feasibility, and distance metrics. |

## How items are annotated

Inputs include object ID, category, center, room, and footprint cells. Candidate rules are evaluated against the map and retained only when both anchor and target are unique. Ties, ambiguous rooms, and cyclic references are rejected. Records store description, rule, depth, and validation status; objects that cannot be uniquely identified are marked `unresolved`.

Descriptions build recursively, by default through four levels. Each example requires unique resolution:

- **Level 1:** `the sofa` — the only sofa in the map.
- **Level 2:** `the chair closest to the sofa`.
- **Level 3:** `the lamp in the same room as the chair closest to the sofa`.

Depth measures reference nesting; one object can have descriptions at several levels. Search retains up to 24 rules per object per level and expands the first 2 rules from the preceding level, rather than enumerating every expression.

## Supported descriptions

Supported information includes **unique object category, unique room category, largest/smallest room, same room as an anchor, and nearest/farthest/second-nearest/second-farthest to an anchor**. Relations can be nested and paraphrased, for example `in the same room as` and `that shares a room with`. Paraphrasing does not increase depth.

Distances are straight-line distances between object centers; differences within one grid cell are treated as ties by default. Largest/smallest refers only to room area. Color, material, shape, object size, orientation, left/right/front/back, above/below, and support relations are not currently supported. Between-object and relative-distance relations are now supported; see the coverage table below.

## Navigation tasks and checks

Accessible targets are selected within the largest free-space component, with up to three ordered visits per task by default. Hard constraints require reaching target approach regions in order. Soft constraints favor staying near or far from an anchor on a particular segment. Four-connected weighted Dijkstra minimizes step costs plus distance penalties; instructions use the same targets and constraints. Target and soft-reference descriptions share per-map, per-object deduplication; exhaustion raises an error.

Preference selection uses policy `endpoint_compatible_v1`. Odd legs propose near and even legs propose far. Landmarks must lie within 1.5 m of the unweighted baseline route. For far preferences, the landmark footprint must additionally be more than 1.5 m from the starting cell, every allowed target approach cell, and every cell of the target footprint. Conflicting candidates are rejected before ranking; another qualifying landmark is selected, or the preference is omitted if none remains. These distances use the map resolution and footprint geometry, not object centers. The 1.5 m rule filters landmark selection; it does not impose a hard clearance on intermediate route cells. Near/far cost functions remain unchanged. Existing saved tasks need regeneration to adopt this policy.

Outputs include item annotations, descriptions by level, difficulty-labeled tasks, and algorithmic trajectories. Validation checks grounding and path consistency, not independent English interpretation or naturalness. Existing annotation reports were removed on request, so the current verifier cannot directly rerun on those shared batches.

## Difficulty classification

`difficulty()` in `hierarchical_descriptions.py` checks the following rules in order. Maximum depth includes **both target and soft-constraint reference descriptions**.

| Label | Condition |
|---|---|
| **hard** | Maximum description depth ≥3 **or** at least 4 target visits. |
| **medium** | Not hard, and depth =2 **or** at least 2 visits **or** any soft constraint. |
| **easy** | Otherwise: one target, a level-1 description, and no soft constraints for a normal generated task. |

For example, visiting the unique sofa without preferences is easy; visiting the chair closest to it is medium; referencing the lamp in the same room as that chair is hard. Even a single simple target becomes hard if its soft constraint uses a level-3 reference. This is a priority-based heuristic, not a weighted score: route length, turns, obstacle complexity, and the number of soft constraints are not considered. Labels are not calibrated to human navigation difficulty.

## Room and preference extensions

The integrated algorithm additionally supports room selectors based on contained object categories and direct raster connectivity, including nested references. Every explicit room selector must uniquely identify a room. Connectivity requires adjacent free cells with valid distinct room IDs; shared walls and diagonal contact are insufficient. See `hierarchical_navigation.md` for the full rules and depth semantics.

Far-preference landmarks must be more than 1.5 m from the segment start, arrival region, and target footprint. This filters reference selection and is not a hard trajectory-distance threshold. `plot_navigation_trajectory.py` renders saved tasks using the companion map PNG; plotting dependencies are listed separately in `requirements-plot.txt`.


## Current instruction coverage

| Annotation mechanism | Implementation |
|---|---|
| Goal and visit order | Ordered target approach regions (hard constraints) |
| Route preference | Per-segment near/far distance costs (soft preferences) |
| Object category | Uniquely resolving category references |
| Room context | Room category, largest/smallest, category contained in room |
| Same-room relation | Target shares the uniquely resolved anchor's room |
| Distance ranking | Nearest/farthest/second-nearest/second-farthest, optionally room-scoped |
| Relative distance | Closer to one uniquely resolved object than another, with tie margin |
| Between | Open-segment projection and bounded perpendicular distance between two anchors |
| Room connectivity | Required direct connections to room references; exact connection count |
| Nested descriptions | Recursive room and object references, with unique resolution |

Exact room object counts and absence selectors are implemented but disabled in
generation by default. Enable with `--enable-counting` and `--enable-absence`.
Search is bounded: support does not guarantee every relation appears in every map.
See `hierarchical_navigation.md` for geometric definitions, evidence limitations,
search limits, JSON rules, and feature switches.
