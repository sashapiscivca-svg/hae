# Security and safety reporting

[Українська](docs/uk/SECURITY.md)

## What to report

- **Security vulnerabilities** in the code, for example unsafe deserialisation, path traversal in the CLI, or malicious model files passing integrity checks.
- **Safety issues in the models**: systematic failure modes that could cost lives (a class of victims, a sensor or a terrain where people are reliably missed), wrong presets or metadata, or published numbers you cannot reproduce.
- **Misuse** of hae that conflicts with [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md).

## How to report

Please use **GitHub private vulnerability reporting** on this repository ("Security" tab → "Report a vulnerability"). This reaches the maintainers privately. Do not open a public issue for vulnerabilities or misuse reports. Model-quality issues that are not sensitive can go to normal GitHub issues.

Please include the version (`pip show hae`), the model name, steps or data to reproduce (a few frames are enough), and what you expected.

We aim to acknowledge reports within 7 days and to publish a fix or an advisory within 90 days. If you report a safety issue, we will credit you in the changelog unless you prefer otherwise.

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x | yes |

## Verifying model files

Every released file is listed with its SHA-256 in `models/SHA256SUMS`:

```bash
cd models && sha256sum -c SHA256SUMS
```

Load models only from this repository or its official Hugging Face repository. The released formats (ONNX and safetensors) cannot execute code when loaded. Training checkpoints (`.pt`) use Python pickle, so pass to `hae.checkpoint.load_checkpoint` or `scripts/export.py` only `.pt` files that you produced yourself.
