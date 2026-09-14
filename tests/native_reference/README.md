# Independent SAELens reference

The wheel is upstream `sae-lens==6.37.6`, distributed under the MIT license
(the wheel includes its license). Its SHA256 is checked before extraction.
The generator imports that extracted wheel in an isolated process and asserts
the import path; this repository's implementation is never the reference.

Source: https://pypi.org/project/sae-lens/6.37.6/

The committed fixtures were generated with PyTorch 2.10.0+cpu. They contain
fixed initial parameters, six global batches, dead-feature masks, and four
combinations of decoder normalization and input bias centering. Tests load the
same weights explicitly. `manifest.json` records hashes, configuration, device,
optimizer settings and clipping scope.

```bash
# Check the committed oracle without overwriting it.
python -I tests/native_reference/generate.py --device cpu --check

# Required on the target GPU environment before accepting distributed results.
python -I tests/native_reference/generate.py --device cuda:0 --check

# Regenerate deliberately (this replaces the fixtures and manifest).
python -I tests/native_reference/generate.py --device cpu
```

The binary fixtures and wheel must be included in patches (`git diff --binary`)
or commits. A text-only patch omitting them cannot reproduce the tests.
