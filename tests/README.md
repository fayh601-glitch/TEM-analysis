# Tests folder

Automatic checks that run with `pytest`. Each test builds a small synthetic TEM
image (or checks a formula) and verifies the pipeline still behaves correctly
after code changes.

| File | What it checks |
|---|---|
| `test_pipeline.py` | End-to-end analysis on fake rod + dot images |
| `test_calibrate.py` | Scale-bar pixel → nm conversion math |

Run all tests:

```bash
pytest
```
