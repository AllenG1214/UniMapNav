# Universal Navigation

Generate grounded hierarchical object descriptions, navigation instructions, ordered target constraints, segment-specific distance preferences, and algorithmic reference trajectories from Map2Route-format maps.

The algorithm uses structured rules, language templates, and graph search. It does not use an LLM or image-based object detection. Paths are algorithmic references, not human demonstrations.

## Setup

Use Python 3.12 (the locally tested version). From the project directory:

```bash
python -m venv .venv
```

Activate with `.venv/Scripts/Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on Linux, then:

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_hierarchical_navigation
```

## Generate and verify

```bash
python build_train_annotations.py --maps /path/to/maps --output generated_maps/new_run --max-depth 4 --objects-per-task 3 --seed 42
python verify_navigation.py --root generated_maps/new_run
```

`--maps` accepts one JSON map or a directory. Directory discovery selects JSON files named after their parent directory. Use a fresh output directory. `--tasks-per-map N` limits task count; 0 sets no explicit limit. Use `--help` for arguments.

Reports store absolute source/task paths and source hashes. Keep source maps and reports available. Regenerate reports after moving data to another machine. Verification uses Python assertions: run without `python -O`.

## Algorithm and modules

1. Read objects, rooms, footprints, and occupancy cells.
2. Build unique descriptions recursively; reject ambiguous or cyclic references.
3. Allocate distinct descriptions across target and soft-reference occurrences of each object within a map. Exhaustion raises an error.
4. Select targets accessible in the largest free-space component. Hard constraints are ordered approach regions; soft preferences are segment-specific near/far costs. Weighted Dijkstra searches four-connected cells.
5. Render instructions from structured tasks and assign heuristic difficulty.
6. Verify grounding, description diversity, source hashes, visit order, path continuity, traversability, and soft-reference geometry and distance metrics.

| Module | Role |
|---|---|
| `build_train_annotations.py` | Unified generation entry point and exports |
| `hierarchical_descriptions.py` | Rule catalog, resolution, depth, and difficulty |
| `description_variants.py` | Description allocation and deduplication |
| `annotate_navigation.py` | Targets, constraints, and path search |
| `verify_navigation.py` | Saved task validation |
| `test_hierarchical_navigation.py` | Regression tests |
| `generate_map_full.py` | Optional Linux simulator export |
| `generate_procthor_batch.py` | Optional random training-map export batch |

Outputs include item annotations, a description catalog, descriptions by level, difficulty-labeled task files, annotation reports, and a batch manifest. Retain reports for verification.

See [algorithm specification](hierarchical_navigation.md), [one-page overview](docs/navigation_annotation_one_page.md), and [simulator setup](full_map_generation.md).

## Limitations and external data

Depth measures semantic nesting, not sentence length. Rule search is bounded and does not enumerate all valid expressions. Difficulty is heuristic and has no human calibration. Validation reuses the rule resolver; it does not independently interpret English sentences.

Datasets, generated batches, simulator binaries, virtual environments, caches, local sync scripts, and maintenance files are excluded from Git. Supply authorized Map2Route-format inputs; third-party licenses apply separately.

Simulator export requires an external Map2Route checkout containing `scripts.make_maps.procthor`, ProcTHOR data, AI2-THOR, and a compatible Linux runtime. These are not installed by the core requirements file.

## Additional algorithms retained from object-annotation

The integrated version also supports uniquely grounded room selectors using room contents and direct raster connectivity, including recursive room references. See the room selector section in `hierarchical_navigation.md` for uniqueness, evidence, and depth rules.

Far-preference landmark selection excludes references within 1.5 m of the segment start, arrival region, or target footprint. This is a selection policy, not a mandatory route-distance threshold.

To visualize a saved task (requires `requirements-plot.txt`):

```bash
python -m pip install -r requirements-plot.txt
python plot_navigation_trajectory.py --map /path/to/map.json --instruction /path/to/instruction.json --output generated_maps/trajectory.png
```

The plotter expects a PNG beside the map JSON. Existing ten-map samples remain on the [object-annotation branch](https://github.com/AllenG1214/UniNav/tree/object-annotation/generated_maps/trajectory_test). Historical samples lack annotation reports; regenerate annotations before running the verifier.
