# NOTICE — сторонні матеріали та зазначення авторства

[English](NOTICE.md)

Вихідний код і натреновані ваги hae опубліковано під [ліцензією MIT](LICENSE), Copyright (c) 2026 The hae authors. У цьому файлі перелічено сторонні матеріали, на які спирається hae, їхні ліцензії та вимоги щодо зазначення авторства. Умови ліцензій ми перевіряли на первинній сторінці кожного набору даних (записи Zenodo — через Zenodo API, сайти або репозиторії авторів). Перш ніж будь-що поширювати, перевірте чинні умови на цих сторінках.

## 1. Набори даних

**У цьому репозиторії не поширюється жодне зображення з цих наборів даних.** `data/annotations/` містить наші *перетворені анотації* (COCO JSON: рамки, імена файлів, належність до підмножин) для тих самих підмножин, на яких навчено й оцінено опубліковану модель. Це адаптації оригінальних анотацій, які поширюються під ліцензією відповідного набору даних, як зазначено нижче. Внесені зміни: перетворення у формат COCO, відображення класу «людина» на id 0, а областей «don't care»/перекритих областей — на клас ignore (id 3), відкидання всіх інших класів (вони стають тлом), зміна розміру кадрів до 640×512 і розбиття на підмножини. Подробиці див. у [DATA_CARD.uk.md](DATA_CARD.uk.md).

| Набір даних | Ліцензія | Використання в hae | Зазначення авторства |
|---|---|---|---|
| **HIT-UAV** | CC0 1.0 | тепловізійне навчання (trainval) і тест | Suo, J. et al. *HIT-UAV: A high-altitude infrared thermal dataset for Unmanned Aerial Vehicle-based object detection.* Scientific Data 10, 227 (2023), doi:[10.1038/s41597-023-02066-6](https://doi.org/10.1038/s41597-023-02066-6). Data: Zenodo, doi:[10.5281/zenodo.7633134](https://doi.org/10.5281/zenodo.7633134). CC0 не вимагає зазначення авторства; ми цитуємо набір як добру практику. |
| **AIResQ Benchmark** | CC BY 4.0 | тепловізійне навчання й валідація | Büttner, J. *AIResQ Benchmark.* Zenodo (2025), doi:[10.5281/zenodo.17405074](https://doi.org/10.5281/zenodo.17405074). Ліцензовано під [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Зміни: див. вище. |
| **WiSARD** (тепловізійна частина) | MIT | лише оцінювання (тест на перенесення, `wisard_ir_val`) | Broyles, D., Hayner, C. R., Leung, K. *WiSARD: A Labeled Visual and Thermal Image Dataset for Wilderness Search and Rescue.* IEEE/RSJ IROS 2022, pp. 9467–9474, doi:[10.1109/IROS47612.2022.9981298](https://doi.org/10.1109/IROS47612.2022.9981298). https://sites.google.com/uw.edu/wisard. Набір даних поширюється під ліцензією MIT, copyright (c) 2022 Daniel Broyles, Christopher Hayner, Karen Leung. Його дозвільне повідомлення: *"Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files … THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND …"* [Дозвіл безоплатно надається будь-якій особі, яка отримала копію цього програмного забезпечення та супровідної документації … ПРОГРАМНЕ ЗАБЕЗПЕЧЕННЯ НАДАЄТЬСЯ «ЯК Є», БЕЗ БУДЬ-ЯКИХ ГАРАНТІЙ …] (повний текст — як у [LICENSE](LICENSE), з наведеним вище рядком про авторські права). |

Ілюстрації в README (`docs/assets/hae-*.jpg`) — тестові кадри HIT-UAV (CC0 1.0), показані в кольоровій палітрі й накладені на виявлення нашої моделі.

Чи поширюються обов'язки щодо зазначення авторства на ваги моделі, натреновані на цих наборах даних, юридично не врегульовано. Ми все одно зазначаємо авторство, а метадані ONNX моделі називають набори даних, на яких її навчено (ключ `training_data`).

### Набори даних, які розглядали, але не використали

Ми відхиляли набори даних з умовами non-commercial (NC) чи no-derivatives (ND), без зазначеної ліцензії, доступні лише на запит чи за передплатою, а також військового походження. Приклади: LLVIP, VisDrone, ForestPersons, POP, NII-CU (некомерційні); SARD, VTUAV-det, TinyPerson, NOMAD (ліцензія неясна). BIRDSAI виключено рішенням проєкту, хоча його ліцензія дозволяла б використання. Повний перелік наведено в [DATA_CARD.uk.md](DATA_CARD.uk.md#7-відхилені-набори-даних).

## 2. Попередньо натреновані ваги

Жодних. Опубліковану модель навчено з випадкової ініціалізації. Жодних контрольних точок ImageNet чи інших сторонніх контрольних точок, моделей-вчителів або backbone не використано.

## 3. Опубліковані ідеї, на яких ґрунтується архітектура

Увесь код у цьому репозиторії написали автори hae. Жодного стороннього вихідного коду не скопійовано. Архітектура спирається на такі опубліковані ідеї:

- Objects as Points / CenterNet — Zhou, Wang, Krähenbühl (2019)
- FCOS: Fully Convolutional One-Stage Object Detection — Tian, Shen, Chen, He (2019)
- Generalized Focal Loss (Quality Focal Loss) — Li et al. (2020)
- Generalized IoU — Rezatofighi et al. (2019)
- Normalized Wasserstein Distance для крихітних об'єктів — Wang, Xu, Yang, Yu (2021)
- GhostNet — Han et al. (2020); RepVGG — Ding et al. (2021)
- Субпіксельна згортка / space-to-depth — Shi et al. (2016)
- Slicing Aided Hyper Inference (SAHI) — Akyon, Altinuc, Temizel (2022)
- AdamW і SGDR — Loshchilov and Hutter (2017, 2019); EMA ваг

## 4. Залежності середовища виконання (встановлює користувач, не входять до складу)

| Пакет | Ліцензія |
|---|---|
| numpy | BSD-3-Clause (і вбудовані компоненти) |
| onnxruntime | MIT (див. його ThirdPartyNotices) |
| onnx | Apache-2.0 |
| opencv-python-headless | Apache-2.0. Wheel-пакет містить FFmpeg під LGPL-2.1; це важливо, якщо ви поширюєте сам wheel-пакет. |
| pillow | MIT-CMU (HPND) |
| huggingface_hub (необов'язковий) | Apache-2.0 |
| torch, tensorboard, pyyaml, pycocotools (лише для навчання) | BSD-подібна / Apache-2.0 / MIT / BSD |

hae не постачає ці пакети. `pip` встановлює їх від їхніх видавців під їхніми власними ліцензіями.
