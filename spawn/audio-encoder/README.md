---
thumbnail: "https://deej-ai.online/favicon-96x96.png"
tags:
- audio
- music
license: "gpl-3.0"
---

This model encodes audio files into vectors of 100 dimensions. It was trained on a million Spotify playlists and tracks. The details can be found [here](https://github.com/teticio/Deej-AI).

To encode an audio first install the package with
```
pip install audiodiffusion
```

and then run

```python
from audiodiffusion.audio_encoder import AudioEncoder

audio_encoder = AudioEncoder.from_pretrained("teticio/audio-encoder")
audio_encoder.encode(<list of audio files>)
```
