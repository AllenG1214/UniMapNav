# Full simulator-backed map export

Run exports on Linux or WSL using a compatible Map2Route/AI2-THOR environment. Core annotation dependencies alone are insufficient for simulator export.

The external checkout passed with `--repo` must contain:

- `scripts.make_maps.procthor.transform_procthor_to_map`
- `scripts.make_maps.procthor.regenerate_procthor_maps`
- `resources/procthor/procthor-10k/train.jsonl.gz` (10,000 training scenes)

Install AI2-THOR and the external checkout's dependencies in that environment and configure its compatible Linux runtime. The default platform is Linux64; single-map export also accepts `--platform cloud`.

From this project's directory inside Linux/WSL:

```bash
python generate_procthor_batch.py --repo /path/to/Map2Route --output generated_maps/new_batch --count 10
```

Use a fresh output directory. The batch records its random seed and zero-based training indices in `selection_and_results.json`, then calls `generate_map_full.py`. It requires simulator-backed exports and does not fall back to floor-only geometry.

For one scene:

```bash
python generate_map_full.py --repo /path/to/Map2Route --split val --index 0 --output-root generated_maps
```

Then use the core Python environment:

```bash
python build_train_annotations.py --maps generated_maps/new_batch/maps --output generated_maps/new_batch/train
python verify_navigation.py --root generated_maps/new_batch/train
```

Maps retain instance identities, rooms, positions, footprints, collision-aware reachable cells, previews, and export reports. See `hierarchical_navigation.md` for annotation semantics. Keep third-party data and runtimes outside Git and follow their distribution licenses.
