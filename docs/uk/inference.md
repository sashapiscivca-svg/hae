# Інференс

[English](../en/inference.md) · [Зміст документації](README.md)

## Встановлення

```bash
pip install -e .                 # з клону репозиторію: numpy, onnxruntime, opencv-python-headless, onnx, pillow
pip install -e ".[hub]"          # + huggingface_hub, щоб завантажувати моделі за назвою
```

## Пошук моделей

`Detector("hae-s-ir")` шукає модель за назвою в такому порядку:

1. явний шлях до файлу `.onnx`;
2. аргумент `root=`, потім каталог `$HAE_MODELS`;
3. `./models` у поточному каталозі, потім каталог `models/` репозиторію;
4. Hugging Face Hub (`sashapiscivca-svg/hae-s-ir`), якщо встановлено `huggingface_hub`. Файли кешуються локально.

Каталог моделі має вигляд `<root>/<name>/<name>_<mode>[_int8].onnx`, де `mode` = `lwir` або `rgb`. За замовчуванням використовується файл int8.

```bash
hae models                         # вбудовані або локальні моделі з їхніми режимами й пресетами
hae info hae-s-ir                  # метадані, збережені у файлі ONNX
hae download hae-s-ir              # завантажити з Hub у локальний кеш
```

## Python API

```python
from hae.detector import Detector

det = Detector("hae-s-ir")                     # збалансований пресет, рекомендований масштаб
det = Detector("hae-s-ir", preset="sensitive") # вища повнота, більше хибних тривог
det = Detector("hae-s-ir", conf=0.3, scale=1.0, threads=4)
det = Detector("hae-s-ir", tile=(256, 320), overlap=0.2)    # розбиття на тайли для великих кадрів

res = det.predict("frame.png")       # файл → Results; каталог або список → list[Results]
res.boxes    # (n, 4) x1, y1, x2, y2 в оригінальних пікселях
res.conf     # (n,) оцінки
res.cls      # (n,) ідентифікатори класів (0 = person)
res.speed    # {"preprocess": ms, "inference": ms, "postprocess": ms}
res.save("out.png"); res.to_json(); res.summary()

dets = det.detect(numpy_image)       # низькорівневий виклик: (n, 6) [x1, y1, x2, y2, score, class], відсортовано за оцінкою
```

Вхідне зображення: двовимірний масив `uint8` у відтінках сірого (тепловізійне зображення в палітрі white-hot). Кольорові зображення перетворюються у відтінки сірого, що коректно лише для тепловізійних кадрів у палітрі відтінків сірого, збережених як RGB.

## Командний рядок

```bash
hae predict MODEL SOURCE [--conf C] [--preset balanced|sensitive] [--scale S] [--tile H,W] [--overlap O]
                         [--project runs] [--json] [--no-save]
```

Команда записує анотовані зображення (і `results.json` із `--json`) у `runs/predict/exp`, `exp2` тощо та виводить час обробки кожного зображення.

## Контракт ONNX (для C++, Rust, Java, мобільних платформ …)

Модель — це один граф ONNX (opset 17) з динамічними розміром пакета, висотою та шириною. Висота й ширина мають бути кратними 32: доповнюйте кадр нулями знизу та справа.

| | `hae-s-ir_lwir.onnx`, `hae-s-ir_lwir_int8.onnx` |
|---|---|
| Назва / форма входу | `t`: float32 `(B,1,H,W)` |
| Передобробка | z-нормалізація кожного кадру в шкалі 0..255: `(t − mean(t)) / max(std(t), 1)`, обчислена на всьому кадрі **до** масштабування чи розбиття на тайли |
| Рекомендований масштаб | збільшити нормалізований кадр у 1,5 раза (білінійна інтерполяція) |
| Виходи | `cls`: логіти `(B,1,H/4,W/4)`; `box`: сирі відстані `(B,4,H/4,W/4)` |

(Код також підтримує RGB-моделі з входом `rgb`: float32 `(B,3,H,W)`, `pixel / 255`; у v1.0.0 жодної такої моделі не опубліковано.)

Декодування (див. `src/hae/decode_np.py`, близько 60 рядків numpy):

1. `score = sigmoid(cls)`;
2. залишити комірки, які є локальними максимумами 3×3 для `score` і мають `score ≥ conf`, не більше 300 найкращих;
3. для залишеної комірки `(i, j)` із центром `cx = (j + 0.5)·4`, `cy = (i + 0.5)·4`: `l, t, r, b = softplus(box[:, i, j]) · 4`, рамка = `(cx − l, cy − t, cx + r, cy + b)`;
4. поділити координати на масштаб входу, щоб повернутися до оригінальних пікселів.

Для одного зображення NMS не потрібне. Під час розбиття на тайли об'єднуйте дублікати на межах тайлів (`hae.sahi`).

**Метадані** (`onnx.load(path).metadata_props`, значення закодовано в JSON): `name`, `hae_version`, `mode`, `tnorm`, `stride`, `ncls`, `class_names`, `input_multiple_of`, `recommended_scale`, `recommended_conf`, `sensitive_conf`, `precision`, `preprocess`, `outputs`, `recipe`, `training_data`, `license`, `copyright`, `intended_use`.

## Вибір порогів

Пресети обрано на валідаційних даних: `balanced` максимізує F1, а `sensitive` — F2. На ваших власних зйомках правильний поріг залежить від того, скільки хибних тривог за хвилину здатен переглянути ваш оператор. `metrics.json` містить повноту (recall), точність (precision) і кількість хибних тривог на кадр для 12 порогів (0,05–0,60) на кожному оцінювальному наборі. Для пошуку радимо починати з `sensitive` і підтверджувати виявлення в часі.
