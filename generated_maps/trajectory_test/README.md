# trajectory_test

Ten maps were sampled without replacement from the 10,000-map ProcTHOR-10k training split. Random seed: **2435713885**. Indices are zero-based; see `selection_and_results.json`.

The batch contains **1,506 object instances and 306 navigation tasks**. Map, object-reference, description-layer, and trajectory checks passed at generation time. Item statuses: 924 uniquely described, 447 structural instances excluded, and 135 unresolved. Difficulty totals: 197 medium and 109 hard tasks.

| Map | Rooms | Object instances | Navigation tasks | Difficulty counts |
|---|---:|---:|---:|---|
| train_09967 | 2 | 89 | 18 | {'medium': 11, 'hard': 7} |
| train_00647 | 10 | 282 | 59 | {'medium': 31, 'hard': 28} |
| train_06889 | 3 | 105 | 22 | {'medium': 12, 'hard': 10} |
| train_04037 | 7 | 202 | 40 | {'medium': 23, 'hard': 17} |
| train_06230 | 5 | 195 | 43 | {'medium': 28, 'hard': 15} |
| train_04647 | 4 | 87 | 18 | {'medium': 18} |
| train_02442 | 6 | 182 | 38 | {'medium': 23, 'hard': 15} |
| train_06155 | 7 | 171 | 31 | {'medium': 25, 'hard': 6} |
| train_01691 | 4 | 145 | 31 | {'medium': 20, 'hard': 11} |
| train_06131 | 2 | 48 | 6 | {'medium': 6} |

## Files

- `maps/<map>/<map>.json`: full simulator-backed map, object geometry, and collision-aware traversability; preview images are in the same folder.
- `train/<map>/item_annotations.json`: all object instances, with descriptions, rules, depths, and unresolved/structural status.
- `train/<map>/descriptions_level_1.json` through `descriptions_level_4.json`: retained rules and English variants at each level.
- `train/<map>/instruction_files/instruction_XXXXXX_medium.json` or `*_hard.json`: navigation tasks. Difficulty follows the algorithm and is not artificially balanced.
- `train/preview.txt`: the first task instruction from each map.
- `validation_summary.json`: per-map counts and the recorded validation outcome.

Tasks contain up to three ordered target visits by default and cover all eligible targets. They include ordered hard constraints, segment-specific near/far preferences, and algorithmic reference trajectories. Ambiguous or inaccessible targets are excluded.

Reference paths are stored in `generated_reference_trajectory`; `human_expert_trajectory` is empty. The original source dataset is unchanged.

## Annotation report removal

All `annotation_report.json` files were deleted at the user's request. Individual tasks and map data remain available. The current `verify_navigation.py` requires those reports and cannot directly rerun on this batch. The validation summary records checks performed before report removal. Some retained metadata contains original absolute paths; those paths may not resolve on another computer.
