# FAQ

[Українська](../uk/faq.md) · [Docs index](README.md)

**Can I use hae commercially, for example in a drone I sell?** Yes. Code and weights are MIT-licensed. Keep the copyright notice and licence text. The training datasets allow commercial use too. We ask you to give attribution to the CC BY dataset (AIResQ) as listed in [NOTICE.md](../../NOTICE.md), and to respect [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md).

**Is there an RGB model?** Not in this release. We trained an RGB line, but it does not yet meet our bar for publication. The code supports RGB, so you can train one with your own data.

**Will smaller or larger models be released?** We trained `n`, `t` and `m` sizes with the same recipe. `s` was the most accurate on HIT-UAV and is the largest size that fits a Pi Zero 2 W, so it is the one we release. Other sizes will be published only if they clearly add value.

**How high can I fly?** What matters is the person's size in pixels. The model was trained on people with median heights of 8–19 px in 640×512 frames, and they work best in roughly that range at the default ×1.5. For people larger than about 25 px, use `--scale 1.0`. Compute your ground sampling distance from the altitude, the field of view and the resolution. A standing adult is about 1.7 m tall, and someone lying down is about 0.5 m wide in a nadir view.

**Why are scores so low (0.3–0.6) for obvious people?** The score is trained to predict box quality (IoU), and IoU is inherently low for tiny boxes. Use the stored presets rather than 0.5 as a default.

**My thermal camera outputs 14-bit radiometric data.** Convert it to 8-bit grayscale (white-hot) with an automatic gain or linear stretch first. The models were trained on 8-bit frames.

**The model misses people in my footage.** Check the scale (people too large or too small for ×1.5), the palette (must be white-hot grayscale) and the threshold (try `--preset sensitive`). Then consider fine-tuning: label a few hundred frames, convert them to COCO, and train from the released `safetensors` weights ([training.md](training.md)). Please also report the failure ([CONTRIBUTING.md](../../CONTRIBUTING.md)).

**Why didn't you use a pretrained backbone (ImageNet, YOLO …)?** To keep the licence chain clean, so that everything can be MIT and usable by anyone, and because pretrained backbones of this size bring little for 8-pixel thermal blobs. The cost is accuracy: published detectors with ImageNet backbones and 100–1000× more compute score higher on standard benchmarks.

**Does hae track people across frames or recognise who they are?** No. It detects presence and location in a single image. Tracking within a mission is left to the application. Recognising identity is out of scope by design (see [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md)).

**Can I run it in a browser or on a phone?** Yes. The ONNX files work with ONNX Runtime Web and Mobile. Implement the preprocessing and the decoding described in [inference.md](inference.md#the-onnx-contract-for-c-rust-java-mobile-).
