# Universal Navigation

Full simulator map export, hierarchical grounded object descriptions, navigation task generation, reference trajectories, and validation.

## Setup

Use Python 3.10 or newer:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m unittest test_hierarchical_navigation
```

Activate the virtual environment before installing dependencies. Map export additionally requires Linux/WSL, AI2-THOR, an external Map2Route checkout and its ProcTHOR resources. Those dependencies and resources are not bundled in this repository; supply the external checkout through `--repo`.

## Workflow

- `generate_procthor_batch.py`: sample and export a reproducible map batch.
- `generate_map_full.py`: export a full simulator-backed map.
- `build_train_annotations.py`: generate descriptions, tasks and reference trajectories.
- `verify_navigation.py`: validate newly generated annotations using their reports.
- `plot_navigation_trajectory.py`: visualize a saved map and instruction.
- `hierarchical_descriptions.py`, `description_variants.py`, `annotate_navigation.py`: core algorithms.
- `test_hierarchical_navigation.py`: algorithm regression tests.

See [map generation](full_map_generation.md), [hierarchical navigation](hierarchical_navigation.md), and [pipeline overview](docs/navigation_annotation_one_page.md).

## Retained sample data

`generated_maps/trajectory_test` contains the latest ten-map batch: 1,506 object instances and 306 navigation tasks. See its README for the recorded validation results and limitations. The old five-map training batch is excluded. Historical validation batches, caches, logs, duplicate instruction templates and intermediate raster/array exports are excluded.

Existing sample reports were previously removed. The verifier requires `annotation_report.json`, so it cannot rerun directly on the retained sample batch. Generate new annotations to obtain fresh reports. Recorded metadata retains original source paths and may require adjustment on another machine.

Reference trajectories are generated algorithmically; `human_expert_trajectory` is empty. Resources and Map2Route are external inputs and are not uploaded.
