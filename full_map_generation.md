# Full simulator-backed map export

Run map generation in Linux/WSL with the existing external Map2Route environment, AI2-THOR and ProcTHOR-10k dataset installed. Pass the external Map2Route checkout to `--repo`; this repository does not include that checkout or its resources.

From this repository's root:

```bash
python generate_procthor_batch.py --repo /path/to/Map2Route --output generated_maps/new_batch --count 10
python build_train_annotations.py --maps generated_maps/new_batch/maps --output generated_maps/new_batch/train
python verify_navigation.py --root generated_maps/new_batch/train
```

Use a new output directory. The exporter records the random seed and zero-based training indices and uses full simulator-backed geometry. It requires a working Linux simulator environment and does not fall back to floor-only geometry.

Maps retain instance identities, rooms, positions, footprints, collision-aware reachable cells, preview images and export reports. See `hierarchical_navigation.md` for annotation details. Existing samples lack annotation reports; newly generated batches create them.
