# Publishing EquiMed-DSS to TestPyPI then PyPI

The package name `equimed_dss` is currently free on both indexes. These steps
use your own API tokens; nothing here is run automatically for you.

## 0. One-time: install tooling (already present on the HPC env)

```bash
python3 -m pip install --user --break-system-packages build twine
```

## 1. Build fresh artifacts

```bash
cd EquiMed_library/EquiMed_DSS_Library_dev
rm -rf dist build
python3 -m build --no-isolation      # plain `python -m build` fails in this env
python3 -m twine check dist/*        # must say PASSED for both files
```

## 2. Manual smoke test of the built wheel (clean, no source on path)

```bash
TMP=$(mktemp -d)
python3 -m pip install --target "$TMP" dist/equimed_dss-1.0.0-py3-none-any.whl
PYTHONPATH="$TMP" python3 -c "
import equimed_dss as e
print('version', e.__version__)
from equimed_dss.utils import SampleDataGenerator
from equimed_dss.domain2 import HierarchicalEquityRatio
df = SampleDataGenerator(random_state=42).generate_fairness_data(n=200)
print('sample data shape', df.shape)
her = HierarchicalEquityRatio()
print('HER on synthetic groups:', her.calculate_her({'White':0.85,'Black':0.78,'Hispanic':0.80}))
"
rm -rf "$TMP"
# and run the test suite:
python3 -m pytest -q
```

## 3. Upload to TestPyPI FIRST

Get a TestPyPI token at https://test.pypi.org/manage/account/token/ (scope: entire
account for the first upload).

```bash
# token-based auth; __token__ is the literal username
python3 -m twine upload --repository testpypi -u __token__ -p pypi-XXXX dist/*
```

Then install from TestPyPI to confirm it resolves (real deps come from real PyPI):

```bash
python3 -m pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ equimed_dss
python3 -c "import equimed_dss; print(equimed_dss.__version__)"
```

## 4. Upload to real PyPI (only after TestPyPI looks good)

Get a PyPI token at https://pypi.org/manage/account/token/.

```bash
python3 -m twine upload -u __token__ -p pypi-XXXX dist/*
```

## Notes
- A version can be uploaded to each index only ONCE. To re-upload, bump
  `equimed_dss/__version__.py` (e.g. 1.0.1) and rebuild.
- Prefer a `~/.pypirc` over inline `-p` so the token is not in shell history:
  ```ini
  [testpypi]
  username = __token__
  password = pypi-XXXX
  [pypi]
  username = __token__
  password = pypi-YYYY
  ```
  Then just `twine upload --repository testpypi dist/*` / `twine upload dist/*`.
- Revoke and rotate any token that has been pasted into a chat or shared.
