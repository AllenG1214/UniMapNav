# Hierarchical navigation annotation

The general entry point is `build_train_annotations.py`. Obsolete scripts have been removed.

```powershell
.venv/Scripts/python.exe build_train_annotations.py --maps <map_directory_or_json> --output <new_output_directory> --max-depth 4 --objects-per-task 3
.venv/Scripts/python.exe verify_navigation.py --root <output_directory>
```

The verification command requires annotation reports. Reports were deleted from the existing shared batches at the user's request; the command cannot directly validate those batches. Newly generated batches still include reports.

## Input

Maps follow the existing Map2Route schema: `object_instances`, `room_instances`, `object_footprints`, `layers.occupancy`, `layer_legends.occupancy.free.value`, `metadata.map_id`, and `metadata.map_info.resolution`.
Directory discovery recursively finds map JSON files whose names match their parent directories. An explicitly supplied JSON file need not follow that naming convention. Map count, categories, and positions are not fixed. Other data formats must first be converted to this schema. Duplicate map names raise an error to prevent overwriting; use a new output directory.

## Description depth

Depth measures the semantic reference tree, not word count or the minimum information needed to identify an object. A globally unique object can have both simple and nested descriptions.

- Level 1: a globally unique category, such as `the sofa`.
- Level 2: a unique room, the largest/smallest room, or a level-1 object reference, such as `the chair closest to the sofa`.
- Level 3: a level-2 reference, such as `the lamp in the same room as the chair closest to the sofa`.
- Level 4 and beyond: continue recursively; a new object relation has depth `1 + anchor.depth`.

A room category must identify exactly one room. Room-size comparisons require complete positive areas and no tie. Area comes from room raster cell counts, falling back to `area_cells`. Object distance is Euclidean distance between grid centers, not traversable route distance. Nearest, farthest, second-nearest, and second-farthest relations exclude the anchor itself; differences within one grid cell are treated as ties by default.

Each complete rule tree must resolve to exactly one target. Reusing an object within the same reference chain is rejected. English template variants do not increase depth. At most 24 rules are retained per object per level, and at most the first 2 rules per object from the preceding level serve as recursive anchors. This bounded search does not enumerate every valid expression; limits can be configured through the `catalog` API. Objects without a unique description are marked `unresolved`, without coordinate or ID-based language fallbacks.

## Outputs and difficulty

Each map has `item_annotations.json`, `description_catalog.json`, and `descriptions_level_1.json`, `descriptions_level_2.json`, and subsequent levels. Catalog entries contain `object_id`, `depth`, `rule`, `dependencies`, and `expressions`. Empty levels are also exported.

Task names include difficulty, for example `instruction_000001_easy.json` or `instruction_000003_hard.json`. Tasks include `difficulty_level`, `difficulty_metadata`, and depth fields for target and soft-reference descriptions.

Difficulty v1 is a heuristic: **hard** if maximum reference depth is at least 3 or there are at least 4 visits; otherwise **medium** if depth is 2, there are at least 2 visits, or any soft constraint exists; otherwise **easy**. Soft-reference depths are included. Batches need not contain every difficulty, and labels are not calibrated through human experiments.

Target and soft-reference occurrences of the same object share a deduplication pool within each map. Allocation prefers less-used references and rules before wording variants. Exhaustion raises an error rather than repeating text. Instructions are rendered from the same structured references and visit order. Hard constraints are ordered accessible approach regions. Soft constraints encode segment-specific near/far distance costs, not mandatory distance thresholds. This version does not add hard forbidden-region constraints.

## Routes and validation

Routes use four-connected weighted Dijkstra within the largest free-space component. Inaccessible targets are explicitly excluded. The verifier checks source hashes, unique resolution, repeated descriptions, difficulty filenames, visit order, path continuity and traversability, soft-reference geometry and scope, and distance metrics. It reuses the rule resolver and does not independently interpret full English sentences. Reference trajectories are algorithmically generated, not human demonstrations.
