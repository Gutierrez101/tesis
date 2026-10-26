"""
Fábrica de extractores de características para Re-ID.

Mantiene la MISMA interfaz que torchreid.utils.FeatureExtractor:
    extractor(imagen_bgr_numpy)  ->  tensor de forma (1, D)
por lo que el resto del código (extractor(crop)[0].cpu().numpy()) no cambia.

Modelo por defecto: osnet_x1_0 (comportamiento actual, sin cambios).
Para cambiar de modelo:   REID_MODEL=repvit python deteccion_reid_dron_contador.py
Opciones: osnet_x1_0 | repvit | efficientformerv2 | fastvit | swiftformer
"""
import os
import cv2
import numpy as np
import torch

from config import DATA_DIR, BASE_EMBEDDINGS, MODEL_PATH

DEFAULT_MODEL = 'osnet_x1_0'


TIMM_MODELS = {
    'repvit':            'repvit_m1_0',
    'efficientformerv2': 'efficientformerv2_s0',
    'fastvit':           'fastvit_t8',
    'swiftformer':       'swiftformer_xs',
}
# Se aceptan también los nombres completos de timm
_PREFIXES = ('repvit', 'efficientformerv2', 'fastvit', 'swiftformer')


def get_model_name():
    return os.environ.get('REID_MODEL', DEFAULT_MODEL).strip().lower()


def _is_timm(name):
    return name in TIMM_MODELS or name.startswith(_PREFIXES)


def get_base_path(model_name=None):
    """OSNet usa el .pkl actual; los demás modelos usan su propio .pkl
    (los embeddings de modelos distintos NO son compatibles entre sí)."""
    name = model_name or get_model_name()
    if not _is_timm(name):
        return BASE_EMBEDDINGS
    root, ext = os.path.splitext(BASE_EMBEDDINGS)
    return f"{root}_{name}{ext}"


def letterbox(img, size_hw, pad_bgr=(104, 116, 124)):
    """Redimensiona conservando proporciones y rellena hasta size_hw.
    El relleno usa el color medio de ImageNet (queda neutro tras normalizar)."""
    th, tw = size_hw
    h, w = img.shape[:2]
    scale = min(th / h, tw / w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.empty((th, tw, 3), dtype=np.uint8)
    canvas[:] = pad_bgr
    top, left = (th - nh) // 2, (tw - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


class TimmExtractor:
    """Extractor para RepViT / EfficientFormerV2 / FastViT / SwiftFormer."""
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_name, device='cpu', input_hw=(256, 128)):
        import timm
        self.device = device
        timm_name = TIMM_MODELS.get(model_name, model_name)
        self.input_hw = input_hw
        self.letterbox = False
        kwargs = {}
        if timm_name.startswith('efficientformerv2'):
            # Sus pesos preentrenados tienen tablas de atención ligadas a 224x224;
            # con otro tamaño da "size mismatch". Se usa 224x224 con relleno
            # (letterbox) para no deformar a la persona.
            self.input_hw = (224, 224)
            self.letterbox = True
        elif timm_name.startswith('repvit'):
            kwargs['img_size'] = input_hw
        # FastViT y SwiftFormer aceptan 256x128 sin parámetros extra.
        try:
            self.model = timm.create_model(
                timm_name, pretrained=True, num_classes=0, **kwargs)
        except TypeError:
            self.model = timm.create_model(
                timm_name, pretrained=True, num_classes=0)
            self.input_hw = (224, 224)
            self.letterbox = True
        self.model.eval().to(device)
        self.feature_dim = self.model.num_features

    def _prep(self, img):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        h, w = self.input_hw
        if self.letterbox:
            img = letterbox(img, (h, w))
        elif img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.MEAN) / self.STD
        return torch.from_numpy(img.transpose(2, 0, 1))

    @torch.no_grad()
    def __call__(self, images):
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]
        batch = torch.stack([self._prep(i) for i in images]).to(self.device)
        return self.model(batch)


def build_extractor(device='cpu', model_name=None):
    name = model_name or get_model_name()
    if _is_timm(name):
        return TimmExtractor(name, device=device)
    # --- comportamiento original: OSNet con torchreid ---
    from torchreid.utils import FeatureExtractor
    return FeatureExtractor(model_name='osnet_x1_0',
                            model_path=MODEL_PATH, device=device)