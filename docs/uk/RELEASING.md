# Публікація релізів

[English](../../RELEASING.md)

Кроки, які виконує супровідник, щоб опублікувати версію на GitHub і Hugging Face Hub. Ніщо з цього не запускається автоматично.

## 1. Перед першою публікацією

1. Простори імен: GitHub `sashapiscivca-svg` (репозиторій `hae`), Hugging Face `alexshynkarenk0` (модель `hae-s-ir`, набір `hae-thermal-sar-annotations`). Якщо якийсь зміниться, замініть його всюди командою `grep -rl <старе> . | xargs sed -i 's/<старе>/<нове>/g'` і перевірте `git diff`. Значення за замовчуванням для завантажень задано в `src/hae/hub.py` (`HAE_HF_ORG`); GitHub перенаправляє перейменовані акаунти, Hugging Face — ні.
2. Увімкніть приватне повідомлення про вразливості GitHub (Settings → Code security), на яке спирається [SECURITY.md](SECURITY.md).

## 2. Перевірка

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

Створіть реліз GitHub із тегу, використайте запис із `CHANGELOG.md` як примітки до релізу й додайте `models/hae-s-ir/*` та `models/SHA256SUMS`.

## 4. Hugging Face Hub

Репозиторій моделі (по одному репозиторію на модель):

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

Репозиторій набору даних (лише анотації):

```bash
huggingface-cli repo create hae-thermal-sar-annotations --type dataset
huggingface-cli upload <your-namespace>/hae-thermal-sar-annotations data/annotations annotations --repo-type dataset
huggingface-cli upload <your-namespace>/hae-thermal-sar-annotations huggingface/dataset/README.md README.md --repo-type dataset
huggingface-cli upload <your-namespace>/hae-thermal-sar-annotations NOTICE.md NOTICE.md --repo-type dataset
```

Після цього перевірте на чистій машині, що `pip install "hae[hub] @ git+https://github.com/<your-namespace>/hae"` з подальшим `hae download hae-s-ir` працює.

## 5. Нові версії

Оновіть `version` у `pyproject.toml`, `src/hae/__init__.py` та `CITATION.cff`, додайте запис у `CHANGELOG.md`, перегенеруйте `SHA256SUMS` для кожного зміненого файлу моделі чи анотацій і ніколи не перезаписуйте опубліковані файли на місці: публікуйте нові ваги під тегом нової версії.
