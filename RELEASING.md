# Releasing

[Українська](docs/uk/RELEASING.md)

The steps a maintainer follows to publish a version on GitHub and the Hugging Face Hub. Nothing here runs automatically.

## 1. Before the first publication

1. Namespaces: GitHub `sashapiscivca-svg` (repository `hae`), Hugging Face `alexshynkarenk0` (model `hae-s-ir`, dataset `hae-thermal-sar-annotations`). If one changes, replace it everywhere with `grep -rl <old> . | xargs sed -i 's/<old>/<new>/g'` and check `git diff`. The runtime default for downloads lives in `src/hae/hub.py` (`HAE_HF_ORG`); GitHub redirects renamed accounts, Hugging Face does not.
2. Enable GitHub private vulnerability reporting (Settings → Code security), which [SECURITY.md](SECURITY.md) relies on.

## 2. Verify

```bash
pip install -e ".[train,dev,hub]"
pytest -q
ruff check src tests scripts
(cd models && sha256sum -c SHA256SUMS) && (cd data/annotations && sha256sum -c SHA256SUMS)
grep -rnI -e "/home/" -e "kaggle" -e "colab" --include="*.py" --include="*.md" --include="*.json" --include="*.yaml" . | grep -v "^./RELEASING" || echo "no internal references"
```

## 3. GitHub

```bash
git init -b main
git add . && git commit -m "hae v1.0.0"
git remote add origin git@github.com:<your-namespace>/hae.git
git push -u origin main
git tag -a v1.0.0 -m "hae v1.0.0" && git push origin v1.0.0
```

Create a GitHub release from the tag, use the `CHANGELOG.md` entry as notes, and attach `models/hae-s-ir/*` and `models/SHA256SUMS`.

## 4. Hugging Face Hub

Model repository (one repository per model):

```bash
huggingface-cli login
huggingface-cli repo create hae-s-ir --type model
cp huggingface/model/README.md /tmp/hae-s-ir-README.md
huggingface-cli upload <your-namespace>/hae-s-ir models/hae-s-ir . --commit-message "hae-s-ir v1.0.0"
huggingface-cli upload <your-namespace>/hae-s-ir /tmp/hae-s-ir-README.md README.md
huggingface-cli upload <your-namespace>/hae-s-ir LICENSE LICENSE
huggingface-cli upload <your-namespace>/hae-s-ir NOTICE.md NOTICE.md
huggingface-cli tag <your-namespace>/hae-s-ir v1.0.0
```

Dataset repository (annotations only):

```bash
huggingface-cli repo create hae-thermal-sar-annotations --type dataset
huggingface-cli upload <your-namespace>/hae-thermal-sar-annotations data/annotations annotations --repo-type dataset
huggingface-cli upload <your-namespace>/hae-thermal-sar-annotations huggingface/dataset/README.md README.md --repo-type dataset
huggingface-cli upload <your-namespace>/hae-thermal-sar-annotations NOTICE.md NOTICE.md --repo-type dataset
```

Afterwards, check that `pip install "hae[hub] @ git+https://github.com/<your-namespace>/hae"` followed by `hae download hae-s-ir` works on a clean machine.

## 5. New versions

Update `version` in `pyproject.toml`, `src/hae/__init__.py` and `CITATION.cff`, add a `CHANGELOG.md` entry, regenerate `SHA256SUMS` for any changed model or annotation file, and never overwrite released files in place: publish new weights under a new version tag.
