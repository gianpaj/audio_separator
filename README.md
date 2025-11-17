# Audio🔹Separator

---
title: Audio🔹Separator
emoji: 🏃
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 5.43.1
app_file: app.py
pinned: true
license: mit
short_description: Vocal and background audio separator
---

Original repository: <https://huggingface.co/spaces/r3gm/Audio_separator/tree/main>

HF Space: <https://huggingface.co/spaces/r3gm/Audio_separator>

Adapted into a Replicate cog using Claude and this prompt

```text
based on this hugging face space repository, create a Cog (replicate) predict.py file like
this one

https://raw.githubusercontent.com/gianpaj/cog-orpheus-3b-0.1-ft/refs/heads/main/predict.py

the inputs should only be:
- file URL
- option to extract voice (stem vocal)
- output format (wav or mp3)

use the same defaults as the app.py e.g. 'vocal_reverb_room_size_conf' 0.15,
vocal_reverb_damping_conf 0.7
```
