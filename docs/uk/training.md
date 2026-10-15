# Навчання та донавчання

[English](../en/training.md) · [Зміст документації](README.md)

## Налаштування

```bash
pip install -e ".[train]"      # PyTorch, pycocotools, tensorboard, pyyaml, safetensors, rich, matplotlib
```

Рекомендовано GPU з CUDA. Опубліковану модель навчено за 42 хвилини на одній NVIDIA T4 (90 епох × 4 900 випадкових вирізок 256×320). Той самий рецепт працює й на CPU, але в багато разів повільніше. Спершу підготуйте дані ([data.md](data.md)).

## Відтворення hae-s-ir

```bash
python scripts/train.py --config configs/hae-s-ir.yaml --data data --out runs/hae-s-ir --device cuda
python scripts/export.py --checkpoint runs/hae-s-ir/last.pt --config configs/hae-s-ir.yaml --data data --out models-local/hae-s-ir --conf 0.5 --sensitive-conf 0.45
python scripts/evaluate.py --model models-local/hae-s-ir/hae-s-ir_lwir_int8.onnx --data data --sets hituav_test,airesq_val,wisard_ir_val
```

`train.py` записує в `--out`: `last.pt`, `best.pt` і `resume.pt` (після переривання продовжуйте з `--resume`); `log.jsonl` і `results.csv` для кожної епохи; готові графіки (`dashboard.png`, `results.png`, `curves.png`, `val_pred.png`, `errors.png`, `labels.png`, `train_batch*.png`); а також `final.json` із метриками фінальних EMA-ваг на кожному наборі з `val_sets` рецепта.

**Очікуваний розкид.** Рецепт — той, що збережений у `models/hae-s-ir/config.json`. Порядок вибірки навчальних вирізок трохи відрізняється від оригінального запуску, а ядра GPU не детерміновані, тож очікуйте AP50 у межах приблизно ±0,02 від опублікованих результатів на HIT-UAV test і ±0,04–0,08 на AIResQ val, а не ідентичних ваг.

## Файл рецепта

```yaml
name: hae-s-ir
train_sets:                      # перший = основний набір; weight = частка навчальних вирізок (сума 1)
  - {name: hituav_trainval, weight: 0.6}
  - {name: airesq_train, weight: 0.4}
val_sets: [hituav_test, airesq_val, wisard_ir_val]   # перший = валідація під час навчання; усі = фінальне оцінювання
eval_stride: 4                   # валідація під час навчання на кожному 4-му кадрі
train:                           # будь-яке поле hae.train.TrainConfig
  epochs: 90
  size: s                        # n | t | s | m (пресет архітектури)
  scale_range: [0.9, 2.4]        # випадкова зміна масштабу вирізок: привчає до вигляду ×1.5 під час інференсу
  eval_scale: 1.5
  copy_paste: 0.3
  tnorm: true                    # z-нормалізація входу для кожного кадру
  ctx: true                      # контекстний рівень на кроці 32
  neg_decay: 1.0                 # м'які негативи
  ...
```

`hae.train.TrainConfig` у `src/hae/train.py` документує кожне поле. Прапорці командного рядка (`--epochs`, `--crops-per-epoch`, `--seed`, `--device`, `--threads`, `--init`) мають пріоритет над рецептом.

## Донавчання під власну камеру

Донавчання найбільше допомагає, коли ваша камера, висота чи місцевість відрізняються від HIT-UAV і AIResQ (див. результати перенесення в [картці моделі](../../MODEL_CARD.uk.md#7-кількісний-аналіз)).

1. Розмітьте кількасот кадрів зі свого сенсора у форматі COCO: `person` як категорія 0 і, за потреби, області для ігнорування як категорія 3. І CVAT, і Label Studio експортують у COCO. Для валідації відкладайте цілі польоти.
2. Додайте їх як набори `my_train` і `my_val` ([data.md](data.md#використання-власних-даних)).
3. Скопіюйте `configs/hae-s-ir.yaml` у `configs/my.yaml`. Поставте `my_train` першим у `train_sets` і залиште частку HIT-UAV/AIResQ (наприклад, 0,5 / 0,3 / 0,2), щоб модель не забувала вивчене. Задайте `val_sets: [my_val, hituav_test]`, менше епох (20–30) і менший `lr` (наприклад, 0,001).
4. Почніть з опублікованих ваг: `python scripts/train.py --config configs/my.yaml --data data --out runs/my --init models/hae-s-ir/hae-s-ir.safetensors` (або `init_weights:` у блоці `train` рецепта). Рецепт має зберігати архітектуру hae-s-ir (`size: s`, `ctx: true`, `tnorm: true`); у разі невідповідності навчання зупиняється з помилкою. Опублікована модель також може слугувати вчителем для дистиляції (`distill` у `TrainConfig`, див. `src/hae/distill.py`).
5. Експортуйте за допомогою `scripts/export.py` (калібруйте int8 на `my_train`) і оберіть пороги за результатами перебору `scripts/evaluate.py --conf 0.3,0.35,0.4,0.45,0.5` на `my_val`.

Не оцінюйте донавчену модель на кадрах, на яких її навчали, і тримайте `wisard_ir_val` або інший незайманий набір, щоб помічати регресії.

## Правила, яких ми дотримувалися (і рекомендуємо)

- **Жодних сторонніх попередньо натренованих ваг.** `build_model` відхиляє `backbone`/`pretrained`. Це зберігає чистоту ліцензії MIT для ваг.
- **Розбивайте за сценами** (політ, локація чи група записів), а не за кадрами. Сусідні кадри майже дублюють один одного.
- **Два запуски з різними seed, перш ніж заявляти про покращення.** Окремі запуски коливаються приблизно на 0,02 AP50.
