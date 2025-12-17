Models are generated at runtime and are not stored in git.

To train and populate `models/` locally:

```
python -m src.pipelines.aqi_flow
```

Artifacts will be written to `models/` and `models/production/`. Delete them if you need a clean checkout.
