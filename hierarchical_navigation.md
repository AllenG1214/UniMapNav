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
- Level 4 and beyond: continue recursively; an object relation has depth `1 + max(reference depths)` (one or two object references).

A room category must identify exactly one room. Room-size comparisons require complete positive areas and no tie. Area comes from room raster cell counts, falling back to `area_cells`. Object distance is Euclidean distance between grid centers, not traversable route distance. Nearest, farthest, second-nearest, and second-farthest relations exclude the anchor itself; differences within one grid cell are treated as ties by default.

Each complete rule tree must resolve to exactly one target. Reusing an object within the same reference chain is rejected. English template variants do not increase depth. At most 24 rules are retained per object per level, and at most the first 2 rules per object from the preceding level serve as recursive anchors. This bounded search does not enumerate every valid expression; limits can be configured through the `catalog` API. Objects without a unique description are marked `unresolved`, without coordinate or ID-based language fallbacks.

## Explicit room relations

Catalog schema v2 introduced an object-level `room` selector. Existing `room_category`, `room_rank`, and object-anchor rules remain supported. A room selector can combine `category`, `rank` (`largest` or `smallest` over all rooms), `contains` (an object category), and recursively nested `adjacent_to` (another room selector).

Examples:

```json
{"category": "chair", "room": {"adjacent_to": {"rank": "largest"}}}
{"category": "cabinet", "room": {"adjacent_to": {"contains": "bed"}}}
{"category": "chair", "room": {"category": "bedroom", "contains": "piano", "adjacent_to": {"category": "living_room"}}}
```

These render as "the chair in the room that is directly connected to the largest room", "the cabinet in the room that is directly connected to the room that contains a bed", and "the chair in the bedroom that contains a piano and is directly connected to the living room". Four template variants express the same predicates, without changing the rule or depth. No inferred functional areas are used.

Two rooms are directly connected only when a pair of horizontally or vertically adjacent raster cells have different valid room IDs and both cells equal the declared free occupancy value. Connectivity is undirected. A shared wall or diagonal contact is insufficient. Missing or mismatched room/occupancy grids yield no connections. Connections through an unlabelled intermediate space are not inferred. The wording describes this raster-level direct-passage relation, not a guarantee about simulator door state.

Every explicit room selector, including every nested reference room, must identify exactly one room. `contains` means at least one object with the specified category and that room's `room_id`; multiple beds in the same room are allowed. If beds occur in two rooms, the unqualified "room that contains a bed" is rejected. Combined predicates may identify one room. If the largest room connects to two rooms, the unqualified "room directly connected to the largest room" is rejected even if the target object happens to be globally unique. After the room is resolved, the complete object rule must still resolve to exactly one object. Room-size ties remain rejected.

A base room selector has depth 1; each `adjacent_to` adds one room-reference level. The enclosing object adds one level, so an object in a room connected to the largest room has depth 3. Generation obeys `max_depth` and bounds room candidates per room per level using `per_object_layer`; deeper room references use `anchor_limit`. References lacking a unique expression are never replaced by room IDs or coordinates.

## Outputs and difficulty

Each map has `item_annotations.json`, `description_catalog.json`, and `descriptions_level_1.json`, `descriptions_level_2.json`, and subsequent levels. Catalog entries contain `object_id`, `depth`, `rule`, `dependencies`, and `expressions`. Empty levels are also exported.

Task names include difficulty, for example `instruction_000001_easy.json` or `instruction_000003_hard.json`. Tasks include `difficulty_level`, `difficulty_metadata`, and depth fields for target and soft-reference descriptions.

Difficulty v1 is a heuristic: **hard** if maximum reference depth is at least 3 or there are at least 4 visits; otherwise **medium** if depth is 2, there are at least 2 visits, or any soft constraint exists; otherwise **easy**. Soft-reference depths are included. Batches need not contain every difficulty, and labels are not calibrated through human experiments.

Target and soft-reference occurrences of the same object share a deduplication pool within each map. Allocation prefers less-used references and rules before wording variants. Exhaustion raises an error rather than repeating text. Instructions are rendered from the same structured references and visit order. Hard constraints are ordered accessible approach regions. Soft constraints encode segment-specific near/far distance costs, not mandatory distance thresholds. This version does not add hard forbidden-region constraints.

## Routes and validation

Routes use four-connected weighted Dijkstra within the largest free-space component. Inaccessible targets are explicitly excluded. The verifier checks source hashes, unique resolution, repeated descriptions, difficulty filenames, visit order, path continuity and traversability, soft-reference geometry and scope, and distance metrics. It reuses the rule resolver and does not independently interpret full English sentences. Reference trajectories are algorithmically generated, not human demonstrations.


## Extended relations (catalog schema v3)

The instruction pipeline covers goal/visit order, segment-specific route preferences,
object category, room context, same-room relations, distance ranking, relative
distance, between relations, room connectivity, and nested descriptions.

### Relative distance and between

```json
{"category":"chair","relation":"closer_than","anchor":{"category":"table"},"other_anchor":{"category":"sofa"},"tie_margin_cells":1.0}
{"category":"chair","relation":"between","anchor":{"category":"table"},"other_anchor":{"category":"sofa"},"between_tolerance_cells":1.0}
```

Both anchors must uniquely resolve to distinct objects with valid positions.
Targets exclude both anchors. `closer_than` means the target's Euclidean distance
to the first anchor is smaller than its distance to the second by **more than**
`tie_margin_cells` (default 1 grid cell). Ties and near ties are rejected.

`between` uses the open line segment between anchor centers: the target's projection
must have `0 < t < 1`, and perpendicular distance must be at most
`between_tolerance_cells` (default 1 grid cell). Coincident anchors are rejected.
This is a geometric center-based relation, not a visibility or reachability claim.
Every complete expression must still identify exactly one target.

Both anchors can themselves be nested expressions. Semantic depth includes the
deeper anchor; recorded dependencies include both branches. The target cannot
occur in either branch, while the two branches may share an ancestor.

### Room-scoped distance ranking

```json
{"category":"chair","room":{"category":"kitchen"},"relation":"nearest","anchor":{"category":"table"},"rank":1,"tie_margin_cells":1.0}
```

The uniquely identified room filters candidate targets **before** distance ranking.
The reference object can be outside that room. The generator combines base room
selectors with nearest/farthest and first/second ranking rules. The resolver also
accepts more deeply nested room selectors. Depth includes both the room and object
references.

### Multiple connections and exact degree

```json
{"category":"clock","room":{"adjacent_to":[{"category":"kitchen"},{"category":"bedroom"}]}}
{"category":"clock","room":{"connection_count":2}}
```

A list in `adjacent_to` requires a direct connection to **every** listed room;
references must each be unique and identify distinct rooms. These are required
connections, not the only allowed connections. `connection_count` separately
requires exactly N distinct connected rooms. Predicates can be combined.

The generator emits bounded two-reference room combinations; the resolver accepts
longer explicit lists. Missing or malformed raster evidence, or a room absent from
the raster, cannot establish even a zero connection count. Unlabelled passages
are not inferred, so degree means degree in the supplied labelled free-cell graph.

### Counting and absence (off by default)

```json
{"category":"table","room":{"object_counts":{"chair":2}}}
{"category":"table","room":{"absent_categories":["bed"]}}
```

Counts refer to exact object-instance counts grouped by category and `room_id`.
Absence means zero matching instances in that room. Both assume a complete supplied
object inventory; they do not infer undetected objects. Category names are rendered
explicitly to avoid guessing plural forms. The resolver understands these rules,
but catalog generation disables them unless requested:

```bash
python build_train_annotations.py --maps /path/to/maps --output generated_maps/extended --enable-counting --enable-absence
python verify_navigation.py --root generated_maps/extended
```

The Python API uses `catalog(data, enable_counting=True, enable_absence=True)`.
Generated absence candidates use categories observed elsewhere in the supplied map.
Catalog metadata records both switches.

### Bounded generation and validation

The default bounds remain 24 expressions per object/depth and 2 recursive anchor
candidates per object/depth. New pair searches examine at most 64 candidate pairs
per level (`pair_limit` in `catalog`). Candidate retention balances relation families
when a layer fills and preserves simple room-connection references. Room layers
reserve representation for a multi-room selector when valid. These bounds mean a
valid relation may not appear in a particular map's retained catalog; the catalog
is not an exhaustive enumeration. English paraphrases do not change the semantics.

All retained rules resolve back to their unique target before export. The same
resolver validates saved target and soft-reference descriptions. Regression tests
cover new relation generation, ambiguity, geometry boundaries, nesting, room
scoping, connectivity evidence, flags, and instruction rendering. This checks
structured semantics, not independent English understanding.
