# Contributing to hae

[Українська](docs/uk/CONTRIBUTING.md)

Thank you for helping. hae improves most from **real-world evidence**: footage from new cameras and terrains, failure reports and careful measurements. Code is welcome too.

## Ways to help

- **Report where the model fails.** Open an issue with the model name, camera, altitude, a few frames and what was missed or falsely detected. Sensitive data can be reported privately (see [SECURITY.md](SECURITY.md)).
- **Benchmarks on hardware:** run `scripts/bench.py` on your board and send the output.
- **Annotated data under an open licence** (CC0, CC BY or MIT; civilian; with the consent required in your jurisdiction). We cannot accept data whose licence forbids commercial use or derivatives.
- **Code:** bug fixes, speed-ups for weak CPUs, better tiling or tracking, documentation and translations.

## Ground rules

1. **No third-party trained weights.** Every model is trained from scratch, so its licence stays clean. Do not add pretrained backbones, teachers or checkpoints.
2. **Licence-clean data only.** Check the licence of the data itself on its primary source and record it in `DATA_CARD.md`.
3. **Measure, don't guess.** A change that claims better accuracy needs numbers on the held-out splits, produced with `scripts/evaluate.py`. If it changes training, include at least two seeds, because a single run varies by about ±0.02 AP50.
4. **Keep the runtime light.** `hae.detector` and everything it imports must work with numpy and onnxruntime only. PyTorch is for training.
5. **Civilian scope.** Contributions that serve the uses listed in [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) as out of scope will not be merged.

## Development setup

```bash
git clone https://github.com/sashapiscivca-svg/hae && cd hae
pip install -e ".[train,dev,hub]"
pytest -q                 # all tests run on CPU with synthetic data; no datasets needed
ruff check src tests scripts
```

## Pull requests

- Keep each PR focused, and add or update tests.
- Code and comments are in English. User documentation is maintained in English (`docs/en`) and Ukrainian (`docs/uk`). If you can only update one language, say so in the PR and we will sync the other.
- Describe what changed and how you verified it (commands and numbers).
- By contributing, you agree that your contribution is licensed under the MIT License of this project.

## Releasing new weights

A new or retrained model needs: the YAML recipe in `configs/`, fp32 and int8 ONNX with full metadata (from `scripts/export.py`), `safetensors`, `config.json`, `metrics.json` from `scripts/evaluate.py` on all held-out splits of its modality, an updated `models/SHA256SUMS`, and a changelog entry.
