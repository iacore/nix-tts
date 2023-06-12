# Usage: python example.py | play -r 22050 -t f32 -

import os, sys
from nix.models.TTS import NixTTSInference
from IPython.display import Audio

# Initiate Nix-TTS
nix = NixTTSInference(model_dir = "nix-ljspeech-deterministic-v0.1")
# Tokenize input text
c, c_length, phoneme = nix.tokenize("Born to multiply, born to gaze into night skies.")
# Convert text to raw speech
xw = nix.vocalize(c, c_length)

with os.fdopen(sys.stdout.fileno(), "wb", closefd=False) as stdout:
    stdout.write(xw)
