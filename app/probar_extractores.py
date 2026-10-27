"""
Prueba rápida de los extractores (ejecutar ANTES de regenerar bases).
    python probar_extractores.py              -> prueba los 4 modelos nuevos
    python probar_extractores.py repvit       -> prueba uno solo
"""
import sys
import time
import numpy as np
import torch
from feature_extractors import build_extractor, TIMM_MODELS

device = 'cuda' if torch.cuda.is_available() else 'cpu'
modelos = sys.argv[1:] or list(TIMM_MODELS)
img = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)  # alto=256, ancho=128

for nombre in modelos:
    try:
        ext = build_extractor(device=device, model_name=nombre)
        ext(img)  # calentamiento
        t = time.time()
        for _ in range(20):
            out = ext(img)
        ms = (time.time() - t) / 20 * 1000
        print(f"✅ {nombre:20s} dim={out.shape[1]:5d}  {ms:6.1f} ms/crop  ({device})")
    except Exception as e:
        print(f"❌ {nombre:20s} {type(e).__name__}: {e}")