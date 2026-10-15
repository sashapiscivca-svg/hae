# Дані: точне відтворення наборів для навчання й оцінювання

[English](../en/data.md) · [Зміст документації](README.md) · [Картка даних](../../DATA_CARD.uk.md)

Репозиторій містить **анотації** кожної підмножини в `data/annotations/` (COCO JSON із сумами SHA-256). Зображення залишаються у своїх видавців: завантажте їх самостійно, а потім `scripts/prepare_data.py` перетворить їх і узгодить з опублікованими анотаціями. Нічого не завантажується з мережі автоматично.

## 1. Завантаження джерел

```bash
python scripts/prepare_data.py sources      # виводить URL, ліцензію та очікувану структуру каталогів для кожного набору даних
```

| Набір даних | Де | Ліцензія | Що має містити `--src` |
|---|---|---|---|
| HIT-UAV | https://zenodo.org/records/7633134 | CC0 1.0 | розпакований архів із `normal_json/{annotations,train,val,test}` |
| AIResQ Benchmark | https://zenodo.org/records/17405074 | CC BY 4.0 | каталог з `images/*.jpg` і мітками YOLO `labels/*.txt` |
| WiSARD (тепловізійна частина) | https://sites.google.com/uw.edu/wisard | MIT | каталоги послідовностей `<date>_<Site>_<platform>_IR_<n>/` (кадри та YOLO `.txt`) |

WiSARD потрібен лише для тесту на перенесення. Для навчання достатньо HIT-UAV і AIResQ.

## 2. Перетворення та узгодження

```bash
python scripts/prepare_data.py hituav    --src ~/dl/HIT-UAV --out data   # → hituav_trainval, hituav_test
python scripts/prepare_data.py airesq    --src ~/dl/AIResQ  --out data   # → airesq_train, airesq_val
python scripts/prepare_data.py wisard-ir --src ~/dl/WiSARD  --out data   # → wisard_ir_val
```

Кожна команда:

1. перетворює оригінальні файли в уніфікований формат (кадри 640×512 у відтінках сірого; відображення класів — як у [картці даних](../../DATA_CARD.uk.md#3-збирання-та-перетворення)) у `data/work/<dataset>/`;
2. **узгоджує** їх із `data/annotations/<set>.json` (за замовчуванням): залишає саме ті кадри, що там перелічені, використовує опубліковані рамки й повідомляє про відсутні кадри, невідповідність розмірів і відмінності в мітках. З `--no-align` натомість зберігається власне розбиття конвертера;
3. записує `data/sets/<set>/{ann.json, thermal/}` (жорсткі посилання, без копій) і навчальні кеші `data/cache/<set>_t.u8`, `_meta.json` та `_ann.json`.

Кеші uint8 для п'яти наборів займають близько 2,1 ГБ на додачу до завантажених файлів.

## 3. Перевірка

Кількість кадрів і людей, виведена наприкінці, має збігатися з карткою даних: `hituav_trainval` 2 319 / 9 701; `hituav_test` 579 / 2 611; `airesq_train` 1 486 / 1 710; `airesq_val` 502 / 933; `wisard_ir_val` 1 633 / 2 439. Якщо повідомляється про відсутні кадри, ваше завантаження, імовірно, відрізняється від версії, яку використовували ми (записи Zenodo версіонуються). Відкрийте issue з цим звітом.

## Використання власних даних

Можна додати будь-який файл COCO з рамками `person` (категорія 0) і необов'язковими ігнорованими областями (категорія 3) на 8-бітних тепловізійних кадрах: покладіть кадри в `data/sets/<name>/thermal/`, а анотації — у `data/sets/<name>/ann.json`, потім побудуйте кеш за допомогою `hae.data.cache.build_cache_thermal(<data>/sets/<name>, "<name>", <data>/cache)` і скопіюйте `ann.json` у `<data>/cache/<name>_ann.json` (саме це робить `write_set` у `scripts/prepare_data.py`). Після цього `<name>` можна використовувати в `train_sets` чи `val_sets` рецепта та в `scripts/evaluate.py --sets`.
