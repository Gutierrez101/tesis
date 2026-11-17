"""
SUITE AUTOMÁTICA DE PRUEBAS UAV Re-ID
=====================================
Implementa las pruebas sobre el pipeline
de deteccion_reid_dron_contador.py.

Solo LEE de la arquitectura actual (config.py, feature_extractors.py,
deteccion_reid_dron_contador.py) y escribe en:
    app/pruebas/datos/                      -> datos de entrada (se crea con --init)
    app/data/pruebas_resultados/<fecha>/    -> CSV, JSON, PNG y resumen.md

Uso rápido (desde la carpeta app/):
    python pruebas/pruebas_uav_reid.py --autotest          # verifica la suite con datos sintéticos
    python pruebas/pruebas_uav_reid.py --init              # crea las carpetas de datos
    python pruebas/pruebas_uav_reid.py --listar            # qué bloques hay y qué datos faltan
    python pruebas/pruebas_uav_reid.py --bloques todo      # ejecuta todo lo que tenga datos
    python pruebas/pruebas_uav_reid.py --bloques arquitecturas,fusion --modelos sistema
"""
import os
import sys
import csv
import json
import time
import math
import random
import argparse
import tempfile
import traceback
from datetime import datetime
from collections import OrderedDict, defaultdict

import numpy as np
import cv2

PRUEBAS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(PRUEBAS_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
DATOS_DIR = os.path.join(PRUEBAS_DIR, 'datos')

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
CROP_HW = (256, 128) 
SEED = 42


def _config():
    """config.py del proyecto (solo lectura)."""
    try:
        import config
        return config
    except Exception:
        return None


def log(msg=''):
    print(msg, flush=True)


def titulo(txt):
    log('\n' + '=' * 78 + f'\n{txt}\n' + '=' * 78)


# ============================================================================
# CONTEXTO COMPARTIDO ENTRE BLOQUES
# ============================================================================
class Ctx:
    def __init__(self, args, salida):
        self.args = args
        self.salida = salida
        os.makedirs(salida, exist_ok=True)
        self.tablas = OrderedDict()      # nombre -> (titulo, filas)
        self.notas = []                  # (bloque, texto)
        self.estado = OrderedDict()      # bloque -> (estado, motivo)
        self.clave = {}                  # métricas para el semáforo (Tabla 34 / 23)
        self.cache = {}
        self.perq = {}                   # configuración -> {'ap','hit1'} alineado a la misma lista de queries
        self.figuras = []
        self._device = None

    # ---- dispositivo ----
    @property
    def device(self):
        if self._device is None:
            d = self.args.dispositivo
            if d == 'auto':
                try:
                    import torch
                    d = 'cuda' if torch.cuda.is_available() else 'cpu'
                except Exception:
                    d = 'cpu'
            self._device = d
        return self._device

    # ---- salidas ----
    def tabla(self, nombre, titulo_t, filas):
        if not filas:
            return
        self.tablas[nombre] = (titulo_t, filas)
        cols = list(filas[0].keys())
        for f in filas:
            for c in f:
                if c not in cols:
                    cols.append(c)
        with open(os.path.join(self.salida, f'{nombre}.csv'), 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            w.writerows(filas)
        log(f'\n📋 {titulo_t}')
        imprimir_tabla(filas, cols)

    def nota(self, bloque, texto):
        self.notas.append((bloque, texto))
        log(f'   ℹ️  {texto}')

    def fig(self, nombre):
        ruta = os.path.join(self.salida, nombre)
        self.figuras.append(nombre)
        return ruta


def _fmt(v):
    if v is None:
        return '-'
    if isinstance(v, float):
        if math.isnan(v):
            return '-'
        return f'{v:.3f}' if abs(v) < 10 else f'{v:.1f}'
    return str(v)


def imprimir_tabla(filas, cols=None):
    cols = cols or list(filas[0].keys())
    ancho = {c: max(len(str(c)), *(len(_fmt(f.get(c))) for f in filas)) for c in cols}
    ancho = {c: min(w, 34) for c, w in ancho.items()}
    log('   ' + ' | '.join(str(c)[:ancho[c]].ljust(ancho[c]) for c in cols))
    log('   ' + '-+-'.join('-' * ancho[c] for c in cols))
    for f in filas:
        log('   ' + ' | '.join(_fmt(f.get(c))[:ancho[c]].ljust(ancho[c]) for c in cols))


# ============================================================================
# DATASET DE RE-ID (galería / consultas)
# ============================================================================
def _listar_imgs(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


class DatasetReID:
    """gal / qry: listas de (ruta, id_persona). Los recortes se redimensionan a 128x256
    exactamente como hace el sistema (cv2.resize(crop, (128, 256)))."""

    def __init__(self, nombre, gal, qry, meta=None, origen=''):
        self.nombre = nombre
        self.gal = gal
        self.qry = qry
        self.meta = meta or {}
        self.origen = origen
        self._img = {}

    @property
    def g_ids(self):
        return np.array([i for _, i in self.gal])

    @property
    def q_ids(self):
        return np.array([i for _, i in self.qry])

    def imagen(self, ruta):
        if ruta not in self._img:
            im = cv2.imread(ruta)
            if im is None:
                raise IOError(f'No se pudo leer {ruta}')
            self._img[ruta] = cv2.resize(im, (CROP_HW[1], CROP_HW[0]))
        return self._img[ruta]

    def imgs_gal(self):
        return [self.imagen(r) for r, _ in self.gal]

    def imgs_qry(self):
        return [self.imagen(r) for r, _ in self.qry]

    def n_ids(self):
        return len(set(self.g_ids) | set(self.q_ids))

    def info(self):
        return (f"{self.nombre}: {len(self.gal)} galería / {len(self.qry)} consultas / "
                f"{len(set(self.g_ids))} identidades (origen: {self.origen})")


def _leer_metadata(raiz):
    ruta = os.path.join(raiz, 'metadata.csv')
    meta = {}
    if os.path.exists(ruta):
        with open(ruta, newline='', encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                meta[os.path.normpath(r['archivo'])] = r
    return meta


def dataset_desde_carpetas(raiz, nombre, max_por_id=None):
    """raiz/gallery/<id>/*.jpg  y  raiz/query/<id>/*.jpg"""
    def leer(sub):
        out = []
        base = os.path.join(raiz, sub)
        for pid in sorted(os.listdir(base)):
            d = os.path.join(base, pid)
            if not os.path.isdir(d):
                continue
            fs = _listar_imgs(d)[:max_por_id] if max_por_id else _listar_imgs(d)
            out += [(os.path.join(d, f), pid) for f in fs]
        return out
    gal, qry = leer('gallery'), leer('query')
    meta_raw = _leer_metadata(raiz)
    meta = {}
    for ruta, _ in qry + gal:
        rel = os.path.normpath(os.path.relpath(ruta, raiz))
        if rel in meta_raw:
            meta[ruta] = meta_raw[rel]
    return DatasetReID(nombre, gal, qry, meta, origen=raiz)


def dataset_desde_personas_base(carpeta, nombre='personas_base', frac_gallery=0.5, max_por_id=None):
    """Sin datos dedicados: se parte cada persona en bloques temporales
    (primera mitad -> galería, segunda mitad -> consultas). Es MÁS estricto que un
    reparto aleatorio, porque fotogramas consecutivos casi idénticos no se mezclan."""
    gal, qry = [], []
    for pid in sorted(os.listdir(carpeta)):
        d = os.path.join(carpeta, pid)
        if not os.path.isdir(d):
            continue
        fs = _listar_imgs(d)
        if max_por_id:
            fs = fs[:max_por_id]
        if len(fs) < 2:
            continue
        k = max(1, int(math.ceil(len(fs) * frac_gallery)))
        gal += [(os.path.join(d, f), pid) for f in fs[:k]]
        qry += [(os.path.join(d, f), pid) for f in fs[k:]]
    return DatasetReID(nombre, gal, qry, origen=carpeta)


def cargar_dataset(ctx, nombre=None):
    clave = f'dataset:{nombre or "principal"}'
    if clave in ctx.cache:
        return ctx.cache[clave]
    mx = ctx.args.max_por_id
    ds = None
    if ctx.args.autotest_dir and nombre is None:
        ds = dataset_desde_carpetas(ctx.args.autotest_dir, 'sintetico', mx)
    elif nombre is not None:
        raiz = os.path.join(DATOS_DIR, 'datasets', nombre)
        if os.path.isdir(os.path.join(raiz, 'gallery')) and os.path.isdir(os.path.join(raiz, 'query')):
            ds = dataset_desde_carpetas(raiz, nombre, mx)
    else:
        raiz = os.path.join(DATOS_DIR, 'reid')
        if os.path.isdir(os.path.join(raiz, 'gallery')) and os.path.isdir(os.path.join(raiz, 'query')):
            ds = dataset_desde_carpetas(raiz, 'propio', mx)
            if not ds.gal or not ds.qry:
                ds = None
        if ds is None:
            cfg = _config()
            pb = getattr(cfg, 'PERSONAS_BASE', None) if cfg else None
            if pb and os.path.isdir(pb):
                ds = dataset_desde_personas_base(pb, 'personas_base', max_por_id=mx)
                if not ds.gal or not ds.qry:
                    ds = None
    ctx.cache[clave] = ds
    if ds is not None:
        log(f'📂 Dataset Re-ID -> {ds.info()}')
    return ds


# ============================================================================
# MÉTRICAS DE RANKING
# ============================================================================
def metricas_ranking(S, q_ids, g_ids):
    """S: (n_query, n_gallery), mayor = más parecido. Devuelve arrays por consulta
    (NaN si la consulta no tiene ninguna imagen correcta en la galería)."""
    S = np.nan_to_num(np.asarray(S, dtype=np.float64), nan=-1e9)
    nq, ng = S.shape
    order = np.argsort(-S, axis=1, kind='stable')
    rel = (np.asarray(g_ids)[order] == np.asarray(q_ids)[:, None])
    valido = rel.any(1)
    cum = np.cumsum(rel, axis=1)
    prec = cum / np.arange(1, ng + 1)[None, :]
    ap = (prec * rel).sum(1) / np.maximum(rel.sum(1), 1)
    out = {'valido': valido, 'ap': np.where(valido, ap, np.nan)}
    for k in (1, 5, 10):
        out[f'hit{k}'] = np.where(valido, rel[:, :k].any(1).astype(float), np.nan)
    out['cmc'] = np.array([np.nanmean(np.where(valido, rel[:, :k].any(1).astype(float), np.nan))
                           for k in range(1, min(ng, 20) + 1)])
    out['top1_idx'] = order[:, 0]
    out['top1_sim'] = S[np.arange(nq), order[:, 0]]
    out['order'] = order
    return out


def resumen_metricas(m):
    return {
        'Rank-1 (%)': 100 * float(np.nanmean(m['hit1'])),
        'Rank-5 (%)': 100 * float(np.nanmean(m['hit5'])),
        'Rank-10 (%)': 100 * float(np.nanmean(m['hit10'])),
        'mAP (%)': 100 * float(np.nanmean(m['ap'])),
        'consultas': int(m['valido'].sum()),
    }


def coseno(A, B):
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    A = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-9, None)
    B = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-9, None)
    return A @ B.T


def a_numpy(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


# ============================================================================
# DESCRIPTORES SOFT-BIOMÉTRICOS (color, textura, silueta) Y ROSTRO
# ============================================================================
REGIONES = ((0.0, 0.20), (0.20, 0.55), (0.55, 1.0))      # cabeza / torso / piernas


def _l2(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _hell(h):
    h = np.asarray(h, dtype=np.float32).ravel()
    s = h.sum()
    return np.sqrt(h / s) if s > 0 else h


def _hist(canal, bins, rango):
    return cv2.calcHist([canal], [0], None, [bins], rango).ravel()


def _por_regiones(img, fn):
    h = img.shape[0]
    return np.concatenate([fn(img[int(a * h):max(int(b * h), int(a * h) + 1)]) for a, b in REGIONES])


def _hsv_hist(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return np.concatenate([_hell(_hist(hsv[:, :, 0], 16, [0, 180])),
                           _hell(_hist(hsv[:, :, 1], 8, [0, 256])),
                           _hell(_hist(hsv[:, :, 2], 8, [0, 256]))])


def d_rgb_global(img):
    return _l2(np.concatenate([_hell(_hist(img[:, :, c], 16, [0, 256])) for c in range(3)]))


def d_hsv_global(img):
    return _l2(_hsv_hist(img))


def d_hsv_regiones(img):
    return _l2(_por_regiones(img, _hsv_hist))


def d_hsv_normalizado(img):
    """Gray-world (balance de blancos) + normalización del brillo (V) + histogramas por región."""
    f = img.astype(np.float32)
    medias = f.reshape(-1, 3).mean(0)
    f = np.clip(f * (medias.mean() / np.maximum(medias, 1e-6)), 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    hsv[:, :, 2] = np.clip(v * (128.0 / max(float(np.median(v)), 1.0)), 0, 255).astype(np.uint8)
    return _l2(_por_regiones(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), _hsv_hist))


def d_lbp(img):
    from skimage.feature import local_binary_pattern
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def h(region):
        lbp = local_binary_pattern(region, 8, 1, 'uniform')
        return _hell(np.bincount(lbp.astype(int).ravel(), minlength=10)[:10])
    return _l2(_por_regiones(g, h))


_GABOR = None


def _banco_gabor():
    global _GABOR
    if _GABOR is None:
        _GABOR = [cv2.getGaborKernel((15, 15), 4.0, th, lam, 0.5, 0)
                  for lam in (10.0, 4.0) for th in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)]
    return _GABOR


def d_gabor(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    resp = [np.abs(cv2.filter2D(g, cv2.CV_32F, k)) for k in _banco_gabor()]
    feats = []
    for a, b in REGIONES:
        i0, i1 = int(a * g.shape[0]), max(int(b * g.shape[0]), int(a * g.shape[0]) + 1)
        for r in resp:
            reg = r[i0:i1]
            feats += [reg.mean(), reg.std()]
    return _l2(np.log1p(np.asarray(feats, dtype=np.float32) * 10))


DESC_BASE = OrderedDict([
    ('rgb_global', d_rgb_global), ('hsv_global', d_hsv_global),
    ('hsv_regiones', d_hsv_regiones), ('hsv_normalizado', d_hsv_normalizado),
    ('lbp', d_lbp), ('gabor', d_gabor),
])
# Qué descriptor representa cada "pista" en las pruebas de fusión
COMP_COLOR, COMP_TEXTURA = 'hsv_normalizado', 'lbp_gabor'


class Segmentador:
    """Silueta corporal con YOLOv8n-seg (se descarga solo la primera vez)."""

    def __init__(self, device):
        from ultralytics import YOLO
        self.m = YOLO('yolov8n-seg.pt')
        self.m.to(device)

    def mascara(self, img):
        r = self.m(img, verbose=False, imgsz=256, conf=0.25)[0]
        if r.masks is None:
            return None
        mejor, a_max = None, 0
        for poly, cls in zip(r.masks.xy, r.boxes.cls):
            if int(cls) != 0 or len(poly) < 3:
                continue
            a = cv2.contourArea(poly.astype(np.float32))
            if a > a_max:
                mejor, a_max = poly, a
        if mejor is None:
            return None
        m = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(m, [mejor.astype(np.int32)], 1)
        return m

    def descriptor(self, img):
        m = self.mascara(img)
        if m is None:
            return None
        frac = m.mean()
        if not (0.10 <= frac <= 0.95):
            return None
        return _l2(cv2.resize(m.astype(np.float32), (32, 64), interpolation=cv2.INTER_AREA).ravel())


class Rostros:
    """EXPERIMENTAL (no probado en la suite): embeddings faciales con insightface.
    pip install insightface onnxruntime   (opcional). Si no está, las variantes con rostro se omiten."""

    def __init__(self, device):
        from insightface.app import FaceAnalysis
        prov = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'], providers=prov)
        self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(320, 320))

    def descriptor(self, img):
        zona = img[:int(0.45 * img.shape[0])]
        zona = cv2.resize(zona, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        caras = self.app.get(zona)
        if not caras:
            return None
        c = max(caras, key=lambda f: f.det_score)
        return np.asarray(c.normed_embedding, dtype=np.float32)


_HAAR = {}


def detectar_rostro_haar(img):
    """Detector de rostros clásico (frontal + perfil) sobre la mitad superior del recorte.
    Devuelve True/False, o None si OpenCV no trae los clasificadores."""
    if not _HAAR:
        base = getattr(getattr(cv2, 'data', None), 'haarcascades', None)
        for k, f in (('frontal', 'haarcascade_frontalface_default.xml'),
                     ('perfil', 'haarcascade_profileface.xml')):
            c = cv2.CascadeClassifier(os.path.join(base, f)) if base else None
            _HAAR[k] = c if (c is not None and not c.empty()) else None
    if _HAAR['frontal'] is None:
        return None
    zona = cv2.cvtColor(img[:max(1, int(0.5 * img.shape[0]))], cv2.COLOR_BGR2GRAY)
    esc = max(1.0, 200.0 / max(zona.shape[1], 1))
    zona = cv2.resize(zona, None, fx=esc, fy=esc, interpolation=cv2.INTER_CUBIC)
    for k in ('frontal', 'perfil'):
        c = _HAAR[k]
        if c is None:
            continue
        for z in (zona, cv2.flip(zona, 1)):
            if len(c.detectMultiScale(z, 1.1, 4, minSize=(24, 24))) > 0:
                return True
    return False


def get_segmentador(ctx):
    if 'seg' not in ctx.cache:
        ctx.cache['seg'] = None
        if not ctx.args.sin_silueta and not ctx.args.autotest_dir:
            try:
                ctx.cache['seg'] = Segmentador(ctx.device)
            except Exception as e:
                ctx.nota('softbio', f'Silueta omitida: no se pudo cargar yolov8n-seg ({type(e).__name__}: {e})')
    return ctx.cache['seg']


def get_rostros(ctx):
    if 'rostros' not in ctx.cache:
        ctx.cache['rostros'] = None
        if not ctx.args.sin_rostro and not ctx.args.autotest_dir:
            try:
                ctx.cache['rostros'] = Rostros(ctx.device)
            except Exception as e:
                ctx.nota('fusion', f'Rostro omitido: insightface no disponible ({type(e).__name__}). '
                                   f'Instala "insightface onnxruntime" si quieres las variantes con rostro.')
    return ctx.cache['rostros']


def _apilar(vectores):
    """Lista de vectores/None -> matriz con NaN en las filas sin descriptor."""
    dim = next((len(v) for v in vectores if v is not None), 0)
    if dim == 0:
        return None
    return np.stack([v if v is not None else np.full(dim, np.nan, dtype=np.float32) for v in vectores]).astype(np.float32)


def calcular_descriptores(ctx, imgs, con_silueta=True, con_rostro=True):
    out = OrderedDict()
    for nombre, fn in DESC_BASE.items():
        out[nombre] = np.stack([fn(im) for im in imgs]).astype(np.float32)
    out['lbp_gabor'] = np.concatenate([out['lbp'], out['gabor']], axis=1) / np.sqrt(2)
    if con_silueta and get_segmentador(ctx) is not None:
        m = _apilar([get_segmentador(ctx).descriptor(im) for im in imgs])
        if m is not None:
            out['silueta'] = m
    if con_rostro and get_rostros(ctx) is not None:
        m = _apilar([get_rostros(ctx).descriptor(im) for im in imgs])
        if m is not None:
            out['rostro'] = m
    return out


# ============================================================================
# EXTRACTORES DE EMBEDDINGS (mismos que usa el sistema) Y VELOCIDAD
# ============================================================================
PERFILES_MODELOS = OrderedDict([
    ('sistema', ['osnet_x1_0', 'repvit', 'efficientformerv2', 'fastvit', 'swiftformer']),
    ('documento', ['osnet_x1_0', 'tf_efficientnetv2_b0', 'edgenext_xx_small', 'fastvit_t8']),
    ('sota', ['osnet_x1_0', 'resnet50', 'mobilenetv2_100', 'tf_efficientnetv2_b0',
              'edgenext_xx_small', 'fastvit_t8']),
])


def resolver_modelos(texto):
    out = []
    for p in [t.strip() for t in texto.split(',') if t.strip()]:
        out += PERFILES_MODELOS.get(p.lower(), [p.lower()])
    vistos = []
    for m in out:
        if m not in vistos:
            vistos.append(m)
    return vistos


class ExtractorSintetico:
    """Solo para --autotest (no requiere torch)."""
    feature_dim = 32

    def __call__(self, imgs):
        if isinstance(imgs, np.ndarray) and imgs.ndim == 3:
            imgs = [imgs]
        out = []
        for im in imgs:
            peq = cv2.resize(im, (4, 8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0
            ruido = np.random.RandomState(int(im.sum()) % (2 ** 31)).randn(peq.size) * 0.20
            out.append(peq + ruido)
        return np.stack(out)


def construir_extractor(nombre, device):
    if nombre == '_sintetico':
        return ExtractorSintetico()
    from feature_extractors import build_extractor, TimmExtractor, _is_timm
    if nombre == 'osnet_x1_0' or _is_timm(nombre):
        return build_extractor(device=device, model_name=nombre)
    return TimmExtractor(nombre, device=device)          # EfficientNetV2, EdgeNeXt, ResNet50...


def extraer_embeddings(ext, imgs, batch=32):
    out = []
    for i in range(0, len(imgs), batch):
        chunk = imgs[i:i + batch]
        try:
            f = a_numpy(ext(chunk))
            if f.ndim != 2 or f.shape[0] != len(chunk):
                raise ValueError('forma inesperada')
        except Exception:
            f = np.stack([a_numpy(ext(c))[0] for c in chunk]) 
        out.append(f)
    return np.concatenate(out)


def medir_velocidad(ext, img, device, n=30):
    """Latencia de UN recorte (como en el sistema), en ms."""
    for _ in range(3):
        a_numpy(ext(img))
    pico = None
    try:
        import torch
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        torch = None
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        a_numpy(ext(img))
        ts.append((time.perf_counter() - t) * 1000)
    mem_mb = None
    try:
        import torch
        if device == 'cuda' and torch.cuda.is_available():
            mem_mb = torch.cuda.max_memory_allocated() / 1e6
    except Exception:
        pass
    if mem_mb is None:
        try:
            import psutil
            mem_mb = psutil.Process().memory_info().rss / 1e6
        except Exception:
            mem_mb = float('nan')
    return float(np.mean(ts)), float(np.percentile(ts, 95)), float(mem_mb)


def n_parametros(ext):
    m = getattr(ext, 'model', None)
    try:
        return sum(p.numel() for p in m.parameters()) / 1e6
    except Exception:
        return float('nan')


# ============================================================================
# FEATURES DEL DATASET (se calculan una sola vez y se comparten entre bloques)
# ============================================================================
def modelo_principal(ctx):
    if ctx.args.modelo_principal:
        return ctx.args.modelo_principal
    if ctx.args.autotest_dir:
        return '_sintetico'
    try:
        from feature_extractors import get_model_name
        return get_model_name()
    except Exception:
        return 'osnet_x1_0'


def deep_embeddings(ctx, ds, modelo, tag='gt', imgs_g=None, imgs_q=None):
    clave = f'deep:{ds.nombre}:{modelo}:{tag}'
    if clave not in ctx.cache:
        ext = construir_extractor(modelo, ctx.device)
        ig = imgs_g if imgs_g is not None else ds.imgs_gal()
        iq = imgs_q if imgs_q is not None else ds.imgs_qry()
        ctx.cache[clave] = (extraer_embeddings(ext, ig), extraer_embeddings(ext, iq))
        ctx.cache[f'ext:{modelo}'] = ext
    return ctx.cache[clave]


def feats_soft(ctx, ds, tag='gt', imgs_g=None, imgs_q=None):
    clave = f'soft:{ds.nombre}:{tag}'
    if clave not in ctx.cache:
        ig = imgs_g if imgs_g is not None else ds.imgs_gal()
        iq = imgs_q if imgs_q is not None else ds.imgs_qry()
        log(f'   ⚙️  calculando descriptores soft-biométricos ({len(ig) + len(iq)} imágenes)...')
        ctx.cache[clave] = (calcular_descriptores(ctx, ig), calcular_descriptores(ctx, iq))
    return ctx.cache[clave]


def bloque_features(ctx, ds, modelo, tag='gt', imgs_g=None, imgs_q=None):
    """Devuelve (feats_gal, feats_qry): dict componente -> matriz."""
    dg, dq = deep_embeddings(ctx, ds, modelo, tag, imgs_g, imgs_q)
    sg, sq = feats_soft(ctx, ds, tag, imgs_g, imgs_q)
    fg, fq = OrderedDict(deep=dg), OrderedDict(deep=dq)
    fg.update(sg)
    fq.update(sq)
    return fg, fq


def sims_de(fg, fq):
    out = OrderedDict()
    for k in fg:
        if k in fq:
            A, B = fq[k], fg[k]
            S = np.full((len(A), len(B)), np.nan, dtype=np.float32)
            va, vb = ~np.isnan(A).any(1), ~np.isnan(B).any(1)
            if va.any() and vb.any():
                S[np.ix_(va, vb)] = coseno(A[va], B[vb])
            out[k] = S
    return out


# ============================================================================
# MOTOR DE FUSIÓN (early / late / weighted / híbrida)
# ============================================================================
def calib_params(S):
    v = S[~np.isnan(S)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.percentile(v, 1)), float(np.percentile(v, 99))
    return lo, (hi if hi - lo > 1e-6 else lo + 1e-6)


def aplicar_calib(S, p):
    return np.clip((S - p[0]) / (p[1] - p[0]), 0.0, 1.0)


def fusion_cal(sims_cal, pesos):
    """Media ponderada de similitudes calibradas; ignora las parejas sin dato (p. ej. sin rostro)."""
    num, den = 0.0, 0.0
    for k, w in pesos.items():
        if w <= 0 or k not in sims_cal:
            continue
        Sk = sims_cal[k]
        m = ~np.isnan(Sk)
        num = num + np.where(m, w * Sk, 0.0)
        den = den + np.where(m, w, 0.0)
    return np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)


def simplex(n, paso):
    pasos = int(round(1 / paso))

    def rec(k, resto):
        if k == 1:
            yield (resto,)
            return
        for i in range(resto + 1):
            for t in rec(k - 1, resto - i):
                yield (i,) + t
    for c in rec(n, pasos):
        yield [x / pasos for x in c]


def _completar(m, hit5, hit10):
    m['hit5'], m['hit10'] = hit5, hit10
    return m


def buscar_pesos(sims_cal, claves, q_ids, g_ids, filas, paso):
    mejor, mejor_w = -1.0, None
    sub = {k: sims_cal[k][filas] for k in claves}
    for w in simplex(len(claves), paso):
        S = fusion_cal(sub, dict(zip(claves, w)))
        ap = float(np.nanmean(metricas_ranking(S, q_ids[filas], g_ids)['ap']))
        if ap > mejor:
            mejor, mejor_w = ap, w
    return dict(zip(claves, mejor_w))


def folds_por_identidad(q_ids, seed=SEED):
    rng = np.random.RandomState(seed)
    fold = np.zeros(len(q_ids), dtype=int)
    for pid in np.unique(q_ids):
        idx = np.where(q_ids == pid)[0]
        rng.shuffle(idx)
        fold[idx[1::2]] = 1
    return fold


def fusion_ponderada_cv(sims, claves, calib, q_ids, g_ids):
    """Pesos aprendidos en una mitad de las consultas y evaluados en la otra (2-fold):
    evita optimizar los pesos sobre los mismos datos con los que se reporta."""
    cal = {k: aplicar_calib(sims[k], calib[k]) for k in claves}
    fold = folds_por_identidad(q_ids)
    paso = 0.1 if len(claves) <= 4 else 0.2
    nq = len(q_ids)
    res = {k: np.full(nq, np.nan) for k in ('ap', 'hit1', 'hit5', 'hit10')}
    pesos_f = []
    for f in (0, 1):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        if len(tr) == 0 or len(te) == 0:
            continue
        w = buscar_pesos(cal, claves, q_ids, g_ids, tr, paso)
        pesos_f.append(w)
        m = metricas_ranking(fusion_cal({k: cal[k][te] for k in claves}, w), q_ids[te], g_ids)
        for k in res:
            res[k][te] = m[k]
    if not pesos_f:                       # muy pocas consultas: pesos iguales
        w = {k: 1.0 / len(claves) for k in claves}
        m = metricas_ranking(fusion_cal(cal, w), q_ids, g_ids)
        pesos_f = [w]
        for k in res:
            res[k] = m[k]
    res['valido'] = ~np.isnan(res['ap'])
    pesos = {k: float(np.mean([w[k] for w in pesos_f])) for k in claves}
    res['pesos'] = pesos
    res['S'] = fusion_cal(cal, pesos)
    return res


def sim_early(fg, fq, claves):
    def apilar(f):
        bl = []
        for k in claves:
            X = np.nan_to_num(f[k], nan=0.0)
            bl.append(X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None))
        return np.concatenate(bl, axis=1)
    return coseno(apilar(fq), apilar(fg))


def f1_umbral_optimo(S, q_ids, g_ids):
    """F1 de identificación con decisión de aceptación (P = aciertos/aceptadas, R = aciertos/consultas)."""
    S = np.nan_to_num(S, nan=-1e9)
    top = S.argmax(1)
    score = S[np.arange(len(S)), top]
    ok = (np.asarray(g_ids)[top] == np.asarray(q_ids))
    mejor = 0.0
    for t in np.unique(np.quantile(score, np.linspace(0, 1, 101))):
        acc = score >= t
        tp = int((acc & ok).sum())
        f1 = 2 * tp / max(int(acc.sum()) + len(S), 1)
        mejor = max(mejor, f1)
    return mejor


class BaseFusion:
    """Todo lo necesario para fusionar: sims por componente, calibración y pesos (se calcula 1 vez)."""
    pass


def obtener_fusion_base(ctx, ds, modelo):
    clave = f'fusion_base:{ds.nombre}:{modelo}'
    if clave in ctx.cache:
        return ctx.cache[clave]
    fg, fq = bloque_features(ctx, ds, modelo)
    sims = sims_de(fg, fq)
    comps = ['deep'] + [c for c in ('silueta', COMP_COLOR, COMP_TEXTURA, 'rostro') if c in sims]
    calib = {k: calib_params(sims[k]) for k in sims}
    res = fusion_ponderada_cv(sims, comps, calib, ds.q_ids, ds.g_ids)
    b = BaseFusion()
    b.fg, b.fq, b.sims, b.comps, b.calib, b.res = fg, fq, sims, comps, calib, res
    ctx.cache[clave] = b
    return b


def _resumen_perq(m):
    return {'ap': np.asarray(m['ap']), 'hit1': np.asarray(m['hit1'])}


# ============================================================================
# SOFT-BIOMETRÍA
# ============================================================================
ESPERADO_SOFT = {
    'rgb_global': 'Rank-1 55-68 / mAP 45-58', 'hsv_global': 'Rank-1 62-72 / mAP 50-62',
    'hsv_regiones': 'Rank-1 66-76 / mAP 55-66', 'hsv_normalizado': 'Rank-1 68-78 / mAP 58-70',
    'lbp': 'Rank-1 60-72 / mAP 50-62', 'gabor': 'Rank-1 62-74 / mAP 52-65',
    'lbp_gabor': 'Rank-1 66-78 / mAP 56-68', 'silueta': 'Rank-1 55-75',
}


def bloque_softbio(ctx):
    ds = cargar_dataset(ctx)
    if ds is None:
        return 'OMITIDO', 'Sin dataset Re-ID (datos/reid/gallery+query o personas_base).'
    sg, sq = feats_soft(ctx, ds)
    sims = sims_de(sg, sq)
    filas = []
    tipos = {'rgb_global': 'Color', 'hsv_global': 'Color', 'hsv_regiones': 'Color', 'hsv_normalizado': 'Color',
             'lbp': 'Textura', 'gabor': 'Textura', 'lbp_gabor': 'Textura', 'silueta': 'Silueta'}
    for k in ESPERADO_SOFT:
        if k not in sims:
            filas.append({'Tipo': tipos[k], 'Descriptor': k, 'Rank-1 (%)': None, 'mAP (%)': None,
                          'Esperado (doc)': ESPERADO_SOFT[k], 'Nota': 'no disponible (sin modelo de segmentación)'})
            continue
        m = metricas_ranking(sims[k], ds.q_ids, ds.g_ids)
        r = resumen_metricas(m)
        filas.append({'Tipo': tipos[k], 'Descriptor': k, 'Rank-1 (%)': r['Rank-1 (%)'], 'mAP (%)': r['mAP (%)'],
                      'Esperado (doc)': ESPERADO_SOFT[k], 'Nota': ''})
        ctx.perq[f'soft:{k}'] = _resumen_perq(m)
    ctx.tabla('t07_09_softbio', 'Tablas 7-9: silueta, color y textura (individuales)', filas)
    ctx.nota('softbio', 'IoU/Dice de segmentación corporal y "CNN liviana de textura" requieren máscaras de referencia / '
                        'entrenamiento y no se automatizan aquí.')
    return 'OK', ''


# ============================================================================
# ARQUITECTURAS LIVIANAS
# ============================================================================
TIPO_MODELO = {
    'osnet_x1_0': 'Omni-scale (Re-ID)', 'repvit': 'CNN reparametrizada', 'efficientformerv2': 'CNN-Transformer',
    'fastvit': 'ViT híbrido rápido', 'fastvit_t8': 'ViT híbrido rápido', 'swiftformer': 'Atención aditiva',
    'tf_efficientnetv2_b0': 'CNN eficiente', 'edgenext_xx_small': 'CNN-Transformer edge',
    'resnet50': 'CNN baseline', 'mobilenetv2_100': 'CNN liviana', '_sintetico': 'sintético',
}


def bloque_arquitecturas(ctx):
    ds = cargar_dataset(ctx)
    if ds is None:
        return 'OMITIDO', 'Sin dataset Re-ID.'
    modelos = resolver_modelos(ctx.args.modelos)
    filas, cmc = [], {}
    for mod in modelos:
        try:
            log(f'\n🧠 Modelo: {mod}')
            dg, dq = deep_embeddings(ctx, ds, mod)
            m = metricas_ranking(coseno(dq, dg), ds.q_ids, ds.g_ids)
            r = resumen_metricas(m)
            ext = ctx.cache[f'ext:{mod}']
            ms, p95, mem = medir_velocidad(ext, ds.imagen(ds.qry[0][0]), ctx.device)
            filas.append({'Arquitectura': mod, 'Tipo': TIPO_MODELO.get(mod, ''),
                          'Rank-1 (%)': r['Rank-1 (%)'], 'Rank-5 (%)': r['Rank-5 (%)'],
                          'Rank-10 (%)': r['Rank-10 (%)'], 'mAP (%)': r['mAP (%)'],
                          'ms/recorte': ms, 'ms p95': p95, 'FPS (solo Re-ID)': 1000.0 / ms,
                          'Params (M)': n_parametros(ext), 'Memoria pico (MB)': mem, 'dim': int(dg.shape[1])})
            ctx.perq[f'deep:{mod}'] = _resumen_perq(m)
            cmc[mod] = m['cmc']
        except Exception as e:
            filas.append({'Arquitectura': mod, 'Tipo': TIPO_MODELO.get(mod, ''), 'Rank-1 (%)': None,
                          'Nota': f'ERROR {type(e).__name__}: {str(e)[:80]}'})
            log(f'   ❌ {mod}: {type(e).__name__}: {e}')
    ctx.tabla('t10_11_arquitecturas', 'Tablas 10-11 y 15: comparación de arquitecturas (preentrenadas, sin fine-tuning '
                                      'en Re-ID salvo OSNet)', filas)
    ctx.cache['cmc'] = cmc
    pm = modelo_principal(ctx)
    for f in filas:
        if f['Arquitectura'] == pm and f.get('Rank-1 (%)') is not None:
            ctx.clave['reid_rank1_deep'], ctx.clave['reid_rank5_deep'] = f['Rank-1 (%)'], f['Rank-5 (%)']
            ctx.clave['reid_map_deep'] = f['mAP (%)']
    ctx.nota('arquitecturas', 'Los rangos "esperados" del documento suponen modelos entrenados para Re-ID. Los timm '
                              '(RepViT, FastViT...) vienen con pesos ImageNet: sin fine-tuning el Rank-1 será menor.')
    ctx.nota('arquitecturas', 'PCB, MGN y TransReID (Tabla 15) requieren repositorios/entrenamiento externos: no se automatizan.')
    return 'OK', ''


# ============================================================================
# FUSIÓN BIOMÉTRICA + ABLACIÓN
# ============================================================================
VARIANTES_V = [  # nombre, rostro, silueta, color, textura
    ('V1', 0, 0, 0, 0), ('V2', 1, 0, 0, 0), ('V3', 0, 1, 0, 0), ('V4', 0, 0, 1, 0),
    ('V5', 0, 0, 0, 1), ('V6', 0, 1, 1, 0), ('V7', 0, 1, 1, 1), ('V8', 1, 1, 1, 1)]
SI = lambda b: 'Sí' if b else 'No'


def _comps_de(rostro, silueta, color, textura):
    c = ['deep']
    if rostro:
        c.append('rostro')
    if silueta:
        c.append('silueta')
    if color:
        c.append(COMP_COLOR)
    if textura:
        c.append(COMP_TEXTURA)
    return c


def evaluar_estrategia(b, comps, estrategia, q_ids, g_ids, calib=None):
    """Devuelve dict de métricas por consulta (+ 'S' y 'pesos')."""
    sims, calib = b.sims, (calib or b.calib)
    if len(comps) == 1:
        S = sims[comps[0]]
        m = metricas_ranking(S, q_ids, g_ids)
        m['S'], m['pesos'] = S, {comps[0]: 1.0}
        return m
    if estrategia == 'early':
        S = sim_early(b.fg, b.fq, comps)
        m = metricas_ranking(S, q_ids, g_ids)
        m['S'], m['pesos'] = S, {k: 1.0 / len(comps) for k in comps}
        return m
    if estrategia == 'late':
        cal = {k: aplicar_calib(sims[k], calib[k]) for k in comps}
        w = {k: 1.0 for k in comps}
        S = fusion_cal(cal, w)
        m = metricas_ranking(S, q_ids, g_ids)
        m['S'], m['pesos'] = S, w
        return m
    if estrategia == 'hibrida':
        soft = [c for c in comps if c not in ('deep', 'rostro')]
        if len(soft) < 2:
            return evaluar_estrategia(b, comps, 'weighted', q_ids, g_ids, calib)
        S_soft = sim_early(b.fg, b.fq, soft)
        sims2 = {'deep': sims['deep'], 'soft': S_soft}
        calib2 = {'deep': calib['deep'], 'soft': calib_params(S_soft)}
        claves = ['deep', 'soft']
        if 'rostro' in comps:
            sims2['rostro'], calib2['rostro'] = sims['rostro'], calib['rostro']
            claves.append('rostro')
        return fusion_ponderada_cv(sims2, claves, calib2, q_ids, g_ids)
    return fusion_ponderada_cv(sims, comps, calib, q_ids, g_ids)          # weighted


def recortar_con_yolo(ctx, imgs):
    """Re-detecta a la persona dentro de cada recorte con el mismo YOLO del sistema (proxy del
    impacto del detector automático: A1 = recorte de referencia vs A2 = recorte de YOLO)."""
    det = get_detector(ctx)
    out, fallos = [], 0
    conf_min = _sist('MIN_CONFIDENCE', 0.4)
    for im in imgs:
        nueva = None
        try:
            r = det(im, verbose=False, imgsz=320, conf=conf_min)[0]
            best, a_max = None, 0
            for box in r.boxes:
                if int(box.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                a = max(0, x2 - x1) * max(0, y2 - y1)
                if a > a_max:
                    best, a_max = (x1, y1, x2, y2), a
            if best is not None:
                x1, y1, x2, y2 = best
                c = im[max(0, y1):y2, max(0, x1):x2]
                if c.size > 0:
                    nueva = cv2.resize(c, (CROP_HW[1], CROP_HW[0]))
        except Exception:
            nueva = None
        if nueva is None:
            fallos += 1
            nueva = im
        out.append(nueva)
    return out, fallos


def bloque_fusion(ctx):
    ds = cargar_dataset(ctx)
    if ds is None:
        return 'OMITIDO', 'Sin dataset Re-ID.'
    mod = modelo_principal(ctx)
    log(f'   modelo principal para fusión: {mod}')
    b = obtener_fusion_base(ctx, ds, mod)
    q, g = ds.q_ids, ds.g_ids
    disp = set(b.comps)
    falta_rostro, falta_sil = 'rostro' not in disp, 'silueta' not in disp
    if falta_rostro:
        ctx.nota('fusion', 'SIN módulo facial: las variantes con rostro (V2, V8, A3, A9-A12) se omiten. '
                           'Instala insightface+onnxruntime para evaluarlas (experimental).')
    if falta_sil:
        ctx.nota('fusion', 'SIN silueta (no cargó yolov8n-seg): las variantes con silueta se omiten.')
    # ---- Variantes internas para las pruebas estadísticas ----
    sin_tex = ['deep'] + [c for c in ('silueta', COMP_COLOR) if c in b.sims]
    internas = OrderedDict([('X:deep', ['deep']), ('X:completo', list(b.comps)),
                            ('X:sin_textura', sin_tex), ('X:con_textura', sin_tex + [COMP_TEXTURA])])
    if 'rostro' in b.sims:
        internas['X:sin_rostro'] = [c for c in b.comps if c != 'rostro']
        internas['X:con_rostro'] = list(b.comps)
    for k, comps in internas.items():
        ctx.perq[k] = _resumen_perq(evaluar_estrategia(b, comps, ctx.args.fusion, q, g))
    if len(b.comps) >= 3:
        ctx.perq['X:early'] = _resumen_perq(evaluar_estrategia(b, list(b.comps), 'early', q, g))
        ctx.perq['X:hibrida'] = _resumen_perq(evaluar_estrategia(b, list(b.comps), 'hibrida', q, g))

    filas, base_deep = [], None
    for nombre, r, s, c, t in VARIANTES_V:
        fila = {'Variante': nombre, 'Rostro': SI(r), 'Silueta': SI(s), 'Color': SI(c), 'Textura': SI(t), 'Deep': 'Sí'}
        if (r and falta_rostro) or (s and falta_sil):
            fila.update({'Rank-1 (%)': None, 'mAP (%)': None, 'Nota': 'omitida (falta módulo)'})
            filas.append(fila)
            continue
        m = evaluar_estrategia(b, _comps_de(r, s, c, t), ctx.args.fusion, q, g)
        res = resumen_metricas(m)
        fila.update({'Rank-1 (%)': res['Rank-1 (%)'], 'mAP (%)': res['mAP (%)'],
                     'F1 (umbral óptimo)': f1_umbral_optimo(m['S'], q, g), 'Nota': ''})
        ctx.perq[f'V:{nombre}'] = _resumen_perq(m)
        if nombre == 'V1':
            base_deep = res
        filas.append(fila)
    ctx.tabla('t12_fusion_variantes', f'Tabla 12: variantes de fusión (estrategia: {ctx.args.fusion})', filas)

    validas = [f for f in filas if f.get('Rank-1 (%)') is not None and f['Variante'] != 'V1']
    if base_deep and validas:
        mejor = max(validas, key=lambda f: f['Rank-1 (%)'])
        t13 = [{'Métrica': 'Rank-1', 'V1 (deep solo)': base_deep['Rank-1 (%)'], f"Mejor ({mejor['Variante']})": mejor['Rank-1 (%)'],
                'Mejora (pp)': mejor['Rank-1 (%)'] - base_deep['Rank-1 (%)'], 'Esperado (doc)': '+5 a +8'},
               {'Métrica': 'mAP', 'V1 (deep solo)': base_deep['mAP (%)'], f"Mejor ({mejor['Variante']})": mejor['mAP (%)'],
                'Mejora (pp)': mejor['mAP (%)'] - base_deep['mAP (%)'], 'Esperado (doc)': '+7 a +11'}]
        ctx.tabla('t13_mejora_fusion', 'Tabla 13: mejora de la fusión frente a deep solo', t13)
        ctx.clave['reid_rank1_propuesto'] = mejor['Rank-1 (%)']
        ctx.clave['reid_map_propuesto'] = mejor['mAP (%)']
        ctx.clave['mejora_vs_deep'] = mejor['Rank-1 (%)'] - base_deep['Rank-1 (%)']
        ctx.clave['reid_f1_propuesto'] = mejor.get('F1 (umbral óptimo)')
        ctx.cache['mejor_variante'] = mejor['Variante']

    if ctx.args.sin_yolo_ablacion:
        ctx.nota('fusion', 'Ablación A1-A12 omitida (--sin-yolo-ablacion).')
        return 'OK', ''
    try:
        yg, f1 = recortar_con_yolo(ctx, ds.imgs_gal())
        yq, f2 = recortar_con_yolo(ctx, ds.imgs_qry())
        ctx.nota('fusion', f'A2-A12: recortes re-detectados con YOLO; fallos de detección: {f1 + f2}/{len(yg) + len(yq)} '
                           f'(se usó el recorte original en esos casos).')
        fgy, fqy = bloque_features(ctx, ds, mod, tag='yolo', imgs_g=yg, imgs_q=yq)
        by = BaseFusion()
        by.fg, by.fq, by.sims = fgy, fqy, sims_de(fgy, fqy)
        by.calib = {k: calib_params(by.sims[k]) for k in by.sims}
    except Exception as e:
        ctx.nota('fusion', f'Ablación A1-A12 omitida: detector YOLO no disponible ({type(e).__name__}).')
        return 'OK', ''
    plan = [('A1', 'Ground truth', b, 0, 0, 0, 0, 'No'), ('A2', 'YOLOv8n', by, 0, 0, 0, 0, 'No'),
            ('A3', 'YOLOv8n', by, 1, 0, 0, 0, 'early'), ('A4', 'YOLOv8n', by, 0, 1, 0, 0, 'early'),
            ('A5', 'YOLOv8n', by, 0, 0, 1, 0, 'early'), ('A6', 'YOLOv8n', by, 0, 0, 0, 1, 'early'),
            ('A7', 'YOLOv8n', by, 0, 1, 1, 0, 'early'), ('A8', 'YOLOv8n', by, 0, 1, 1, 1, 'early'),
            ('A9', 'YOLOv8n', by, 1, 1, 1, 1, 'early'), ('A10', 'YOLOv8n', by, 1, 1, 1, 1, 'late'),
            ('A11', 'YOLOv8n', by, 1, 1, 1, 1, 'weighted'), ('A12', 'YOLOv8n', by, 1, 1, 1, 1, 'hibrida')]
    filas = []
    for nombre, det, bb, r, s, c, t, est in plan:
        fila = {'Variante': nombre, 'Detector': det, 'Backbone': mod, 'Rostro': SI(r), 'Silueta': SI(s),
                'Color': SI(c), 'Textura': SI(t), 'Fusión': est}
        comps = _comps_de(r, s, c, t)
        if any(k not in bb.sims for k in comps):
            # En A9-A12 sin rostro/silueta se usa lo disponible (se indica en la fila)
            if nombre in ('A9', 'A10', 'A11', 'A12'):
                comps = [k for k in comps if k in bb.sims]
                fila['Nota'] = 'sin: ' + ','.join(k for k in _comps_de(1, 1, 1, 1) if k not in bb.sims)
            else:
                fila.update({'Rank-1 (%)': None, 'Nota': 'omitida (falta módulo)'})
                filas.append(fila)
                continue
        m = evaluar_estrategia(bb, comps, est if est != 'No' else 'early', q, g)
        res = resumen_metricas(m)
        fila.update({'Rank-1 (%)': res['Rank-1 (%)'], 'Rank-5 (%)': res['Rank-5 (%)'], 'mAP (%)': res['mAP (%)'],
                     'F1 (umbral óptimo)': f1_umbral_optimo(m['S'], q, g)})
        ctx.perq[f'A:{nombre}'] = _resumen_perq(m)
        filas.append(fila)
    ctx.tabla('t14_ablacion', 'Tabla 14: ablation study (A1 = recorte de referencia; A2-A12 = recorte de YOLO)', filas)
    return 'OK', ''


# ============================================================================
# ROBUSTEZ UAV
# ============================================================================
def ocluir(img, frac, rng):
    H, W = img.shape[:2]
    w = max(1, int(W * rng.uniform(0.5, 1.0)))
    h = max(1, int(min(H, frac * H * W / w)))
    x, y = rng.randint(0, W - w + 1), rng.randint(0, H - h + 1)
    out = img.copy()
    out[y:y + h, x:x + w] = 127
    return out


def degradar_resolucion(img, h, w):
    p = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.resize(p, (CROP_HW[1], CROP_HW[0]), interpolation=cv2.INTER_LINEAR)


def _propuesto(b, sims):
    pesos = b.res['pesos']
    cal = {k: aplicar_calib(sims[k], b.calib[k]) for k in pesos if k in sims}
    return fusion_cal(cal, pesos)


def _eval_consultas(ctx, ds, mod, b, imgs_q):
    """Deep solo vs propuesto para un conjunto (perturbado) de imágenes de consulta."""
    ext = ctx.cache.get(f'ext:{mod}') or construir_extractor(mod, ctx.device)
    fq = OrderedDict(deep=extraer_embeddings(ext, imgs_q))
    desc = calcular_descriptores(ctx, imgs_q)
    for k in b.comps:
        if k != 'deep' and k in desc:
            fq[k] = desc[k]
    sims = sims_de(b.fg, fq)
    md = metricas_ranking(sims['deep'], ds.q_ids, ds.g_ids)
    mp = metricas_ranking(_propuesto(b, sims), ds.q_ids, ds.g_ids)
    return md, mp


GRUPOS_ALTURA = [('5-10 m', 0, 10), ('10-20 m', 10, 20), ('20-40 m', 20, 40), ('40-60 m', 40, 60), ('>60 m', 60, 1e9)]


def bloque_robustez(ctx):
    ds = cargar_dataset(ctx)
    if ds is None:
        return 'OMITIDO', 'Sin dataset Re-ID.'
    mod = modelo_principal(ctx)
    b = obtener_fusion_base(ctx, ds, mod)
    q, g = ds.q_ids, ds.g_ids
    S_deep, S_prop = b.sims['deep'], _propuesto(b, b.sims)
    # ---- por altura y ángulo (requiere metadata.csv) ----
    qmeta = [ds.meta.get(r) for r, _ in ds.qry]
    if any(m is not None for m in qmeta):
        for campo, nombre_t, tit in (('altura_m', 't18_robustez_altura', 'Tabla 18: robustez por altura UAV'),
                                     ('angulo', 't19_robustez_angulo', 'Tabla 19: robustez por ángulo de cámara')):
            grupos = OrderedDict()
            for i, m in enumerate(qmeta):
                if not m or not m.get(campo):
                    continue
                if campo == 'altura_m':
                    try:
                        h = float(m[campo])
                    except ValueError:
                        continue
                    et = next((n for n, lo, hi in GRUPOS_ALTURA if lo <= h < hi), None)
                else:
                    et = m[campo].strip()
                if et:
                    grupos.setdefault(et, []).append(i)
            filas = []
            orden = [n for n, _, _ in GRUPOS_ALTURA] if campo == 'altura_m' else sorted(grupos)
            for et in orden:
                idx = np.array(grupos.get(et, []), dtype=int)
                if len(idx) == 0:
                    continue
                md = metricas_ranking(S_deep[idx], q[idx], g)
                mp = metricas_ranking(S_prop[idx], q[idx], g)
                filas.append({'Grupo': et, 'Consultas': len(idx), 'Rank-1 deep (%)': 100 * np.nanmean(md['hit1']),
                              'Rank-1 propuesto (%)': 100 * np.nanmean(mp['hit1']),
                              'mAP propuesto (%)': 100 * np.nanmean(mp['ap'])})
            ctx.tabla(nombre_t, tit, filas)
            ctx.cache[f'rob_{campo}'] = filas
    else:
        ctx.nota('robustez', 'Altura/ángulo omitidos: falta datos/reid/metadata.csv '
                             '(columnas: archivo,altura_m,angulo). Oclusión y resolución sí se evalúan (sintéticas).')
    # ---- oclusión sintética ----
    rng = np.random.RandomState(SEED)
    base_q = ds.imgs_qry()
    filas = []
    for nombre, lo, hi in (('Sin oclusión', 0, 0), ('10-25%', .10, .25), ('25-50%', .25, .50), ('>50%', .50, .75)):
        imgs = base_q if hi == 0 else [ocluir(im, rng.uniform(lo, hi), rng) for im in base_q]
        md, mp = _eval_consultas(ctx, ds, mod, b, imgs)
        d1, p1 = 100 * np.nanmean(md['hit1']), 100 * np.nanmean(mp['hit1'])
        filas.append({'Oclusión (aprox.)': nombre, 'Rank-1 deep solo (%)': d1, 'Rank-1 propuesto (%)': p1,
                      'Mejora (pp)': p1 - d1, 'mAP propuesto (%)': 100 * np.nanmean(mp['ap'])})
    ctx.tabla('t20_robustez_oclusion', 'Tabla 20: robustez ante oclusión (oclusión sintética por bloque gris)', filas)
    ctx.cache['rob_oclusion'] = filas
    # ---- baja resolución ----
    filas = []
    for h, w in ((256, 128), (192, 96), (128, 64), (96, 48), (64, 32)):
        imgs = base_q if h == 256 else [degradar_resolucion(im, h, w) for im in base_q]
        md, mp = _eval_consultas(ctx, ds, mod, b, imgs)
        filas.append({'Resolución (alto x ancho)': f'{h}x{w}', 'Rank-1 deep solo (%)': 100 * np.nanmean(md['hit1']),
                      'Rank-1 propuesto (%)': 100 * np.nanmean(mp['hit1']),
                      'mAP propuesto (%)': 100 * np.nanmean(mp['ap'])})
    ctx.tabla('t21_robustez_resolucion', 'Tabla 21: robustez ante baja resolución del recorte', filas)
    ctx.cache['rob_resolucion'] = filas
    ctx.nota('robustez', '"Propuesto" = fusión ponderada con los pesos aprendidos en datos limpios (sin reajustar).')
    return 'OK', ''


# ============================================================================
# OPEN-SET Y UMBRAL
# ============================================================================
def particion_openset(ds, frac=0.3, seed=SEED):
    ids = sorted(set(ds.g_ids))
    if len(ids) < 3:
        return None
    rng = np.random.RandomState(seed)
    unk = set(rng.choice(ids, max(1, int(round(len(ids) * frac))), replace=False))
    g_mask = np.array([i not in unk for i in ds.g_ids])
    q_known = np.array([i not in unk for i in ds.q_ids])
    if q_known.all() or not q_known.any():
        return None
    return g_mask, q_known


def _decision(S, q_ids, g_ids, g_mask, q_known):
    Sg = np.nan_to_num(S[:, g_mask], nan=-1e9)
    gi = np.asarray(g_ids)[g_mask]
    top = Sg.argmax(1)
    score = Sg[np.arange(len(Sg)), top]
    pred_ok = (gi[top] == np.asarray(q_ids)) & q_known
    return score, pred_ok


def _curva(score, pred_ok, q_known, umbrales):
    filas = []
    for t in umbrales:
        acc = score >= t
        tp = int((acc & pred_ok).sum())
        p = tp / max(int(acc.sum()), 1)
        r = tp / max(int(q_known.sum()), 1)
        far = int((acc & ~q_known).sum()) / max(int((~q_known).sum()), 1) if (~q_known).any() else float('nan')
        filas.append({'umbral': float(t), 'precision': p, 'recall': r,
                      'f1': 2 * p * r / (p + r) if p + r > 0 else 0.0, 'far': far})
    return filas


def bloque_openset(ctx):
    from sklearn.metrics import roc_auc_score, roc_curve
    ds = cargar_dataset(ctx)
    if ds is None:
        return 'OMITIDO', 'Sin dataset Re-ID.'
    part = particion_openset(ds)
    if part is None:
        return 'OMITIDO', 'Se necesitan ≥3 identidades para separar conocidas/desconocidas.'
    g_mask, q_known = part
    mod = modelo_principal(ctx)
    b = obtener_fusion_base(ctx, ds, mod)
    cal = lambda comps: {k: aplicar_calib(b.sims[k], b.calib[k]) for k in comps if k in b.sims}
    soft = [k for k in ('deep', COMP_COLOR, COMP_TEXTURA) if k in b.sims]
    configs = OrderedDict([
        ('Solo deep embedding', b.sims['deep']),
        ('Deep + color/textura', fusion_cal(cal(soft), {k: 1.0 for k in soft})),
        ('Deep + bio/soft-bio completo', fusion_cal(cal(b.comps), b.res['pesos']))])
    filas, curvas = [], {}
    for nombre, S in configs.items():
        score, ok = _decision(S, ds.q_ids, ds.g_ids, g_mask, q_known)
        auroc = float(roc_auc_score(q_known.astype(int), score))
        fpr, tpr, _ = roc_curve(q_known.astype(int), score)
        qs = np.unique(np.quantile(score, np.linspace(0, 1, 201)))
        cur = _curva(score, ok, q_known, qs)
        mejor = max(cur, key=lambda r: r['f1'])
        filas.append({'Configuración': nombre, 'AUROC': auroc, 'F1': mejor['f1'], 'Falsos aceptados (FAR)': mejor['far'],
                      'Umbral óptimo': mejor['umbral']})
        curvas[nombre] = {'fpr': fpr, 'tpr': tpr, 'auroc': auroc}
    ctx.tabla('t27_openset', f'Tabla 27: open-set ({int((~q_known).sum())} consultas desconocidas, '
                             f'{int(q_known.sum())} conocidas)', filas)
    ctx.cache['roc'] = curvas
    ctx.clave['openset_auroc'] = filas[-1]['AUROC']
    return 'OK', ''


def bloque_umbral(ctx):
    ds = cargar_dataset(ctx)
    if ds is None:
        return 'OMITIDO', 'Sin dataset Re-ID.'
    part = particion_openset(ds)
    g_mask = part[0] if part else np.ones(len(ds.gal), bool)
    q_known = part[1] if part else np.ones(len(ds.qry), bool)
    modelos = resolver_modelos(ctx.args.modelos)
    pm = modelo_principal(ctx)
    if pm not in modelos:
        modelos = [pm] + modelos
    recom, resumen = {}, []
    for mod in modelos:
        try:
            dg, dq = deep_embeddings(ctx, ds, mod)
        except Exception as e:
            log(f'   ❌ {mod}: {type(e).__name__}: {e}')
            continue
        score, ok = _decision(coseno(dq, dg), ds.q_ids, ds.g_ids, g_mask, q_known)
        fino = _curva(score, ok, q_known, np.arange(0.20, 0.981, 0.01))
        mejor = max(fino, key=lambda r: r['f1'])
        recom[mod] = {'umbral_recomendado': round(mejor['umbral'], 2), 'f1': mejor['f1'],
                      'precision': mejor['precision'], 'recall': mejor['recall']}
        resumen.append({'Modelo': mod, 'Umbral óptimo (F1)': mejor['umbral'], 'Precision': mejor['precision'],
                        'Recall': mejor['recall'], 'F1': mejor['f1']})
        if mod == pm:
            doc = [r for r in _curva(score, ok, q_known, [0.5, 0.6, 0.7, 0.8, 0.9])]
            ctx.tabla('t28_umbral', f'Tabla 28: umbral de similitud ({mod})',
                      [{'Umbral': r['umbral'], 'Precision': r['precision'], 'Recall': r['recall'], 'F1': r['f1']} for r in doc])
            ctx.cache['pr_curve'] = (fino, mod)
            ctx.clave['reid_f1_deep'] = mejor['f1']
    ctx.tabla('t28b_umbral_por_modelo', 'Umbral óptimo por modelo (úsalo en config.py -> IDENTIFICATION_THRESHOLDS)', resumen)
    with open(os.path.join(ctx.salida, 'umbrales_recomendados.json'), 'w', encoding='utf-8') as fh:
        json.dump(recom, fh, indent=2)
    log('\n   📌 Sugerencia para app/config.py (NO se modifica automáticamente):')
    log('   IDENTIFICATION_THRESHOLDS = {' + ', '.join(f"'{m}': {v['umbral_recomendado']}" for m, v in recom.items()) + '}')
    return 'OK', ''


# ============================================================================
# ESCALABILIDAD DE GALERÍA
# ============================================================================
def bloque_escalabilidad(ctx):
    ds = cargar_dataset(ctx)
    if ds is None:
        return 'OMITIDO', 'Sin dataset Re-ID.'
    mod = modelo_principal(ctx)
    dg, dq = deep_embeddings(ctx, ds, mod)
    ids = sorted(set(ds.g_ids))
    n_base = len(ids)
    # Distractores opcionales: identidades extra solo en la galería
    dist_emb, dist_ids = None, []
    ddir = os.path.join(DATOS_DIR, 'distractores')
    if os.path.isdir(ddir):
        rutas, etq = [], []
        for pid in sorted(os.listdir(ddir)):
            d = os.path.join(ddir, pid)
            if os.path.isdir(d):
                for f in _listar_imgs(d):
                    rutas.append(os.path.join(d, f))
                    etq.append('DIST_' + pid)
        if rutas:
            ext = ctx.cache[f'ext:{mod}']
            imgs = [cv2.resize(cv2.imread(r), (CROP_HW[1], CROP_HW[0])) for r in rutas]
            dist_emb, dist_ids = extraer_embeddings(ext, imgs), np.array(etq)
    n_dist = len(set(dist_ids)) if len(dist_ids) else 0
    total = n_base + n_dist
    tam = sorted(set([s for s in (50, 100, 500, 1000, 1500) if s <= total] + [n_base]
                     + ([max(2, n_base // 4), max(2, n_base // 2)] if n_base >= 8 else [])))
    rng = np.random.RandomState(SEED)
    filas = []
    for N in tam:
        r1, mp = [], []
        for rep in range(3):
            k = min(N, n_base)
            base_ids = list(rng.choice(ids, k, replace=False))
            extra = []
            if N > k and n_dist:
                extra = list(rng.choice(sorted(set(dist_ids)), min(N - k, n_dist), replace=False))
            gm = np.isin(ds.g_ids, base_ids)
            G = dg[gm]
            gi = ds.g_ids[gm]
            if extra:
                dm = np.isin(dist_ids, extra)
                G, gi = np.concatenate([G, dist_emb[dm]]), np.concatenate([gi, dist_ids[dm]])
            qm = np.isin(ds.q_ids, base_ids)
            m = metricas_ranking(coseno(dq[qm], G), ds.q_ids[qm], gi)
            r1.append(100 * np.nanmean(m['hit1']))
            mp.append(100 * np.nanmean(m['ap']))
            if len(base_ids) == n_base and not extra:
                break
        filas.append({'Identidades en galería': k + len(extra), 'Rank-1 (%)': float(np.mean(r1)),
                      'mAP (%)': float(np.mean(mp)), 'Rank-1 desv.': float(np.std(r1)), 'repeticiones': len(r1)})
    ctx.tabla('t29_escalabilidad', f'Tabla 29: escalabilidad de la galería ({mod})', filas)
    ctx.cache['escala'] = filas
    if total < 500:
        ctx.nota('escalabilidad', f'Solo hay {total} identidades disponibles. Para 500-1500+ añade identidades en '
                                  f'datos/distractores/<id>/*.jpg (se agregan solo a la galería).')
    return 'OK', ''


# ============================================================================
# IMAGEN ÚNICA vs VIDEO-BASE
# ============================================================================
def bloque_tracking(ctx):
    ds = cargar_dataset(ctx)
    if ds is None:
        return 'OMITIDO', 'Sin dataset Re-ID.'
    mod = modelo_principal(ctx)
    dg, dq = deep_embeddings(ctx, ds, mod)
    L = ctx.args.tracklet_len
    por_id = OrderedDict()
    for i, (_, pid) in enumerate(ds.qry):
        por_id.setdefault(pid, []).append(i)
    tracklets = []
    for pid, idx in por_id.items():
        for s in range(0, len(idx), L):
            ch = idx[s:s + L]
            if len(ch) >= 3:
                tracklets.append((pid, ch))
    if len(tracklets) < 2:
        return 'OMITIDO', f'Se necesitan tracklets de ≥3 fotogramas consecutivos por persona (tracklet-len={L}).'
    t_ids = np.array([p for p, _ in tracklets])
    filas = []
    for nombre, k in (('Imagen única', 1), ('Promedio de 3 frames', 3), ('Promedio de 5 frames', 5),
                      ('Tracklet completo', None)):
        Q = np.stack([dq[c[:k] if k else c].mean(0) for _, c in tracklets])
        m = metricas_ranking(coseno(Q, dg), t_ids, ds.g_ids)
        r = resumen_metricas(m)
        filas.append({'Entrada': nombre, 'Rank-1 (%)': r['Rank-1 (%)'], 'mAP (%)': r['mAP (%)'], 'tracklets': len(tracklets)})
        ctx.perq[f'track:{k or "all"}'] = _resumen_perq(m)
    ctx.tabla('t25_image_vs_video', 'Tabla 25: imagen única vs video-based (tracklets = fotogramas consecutivos '
                                    'de cada persona en orden de archivo)', filas)
    ctx.nota('tracking', 'Los tracklets se arman con imágenes consecutivas de la consulta. Para IDF1/MOTA/ID-switches '
                         'reales usa el bloque "mot" con un video + anotación MOT.')
    return 'OK', ''


# ============================================================================
# GENERALIZACIÓN (evaluación zero-shot por dataset)
# ============================================================================
def bloque_generalizacion(ctx):
    nombres = []
    droot = os.path.join(DATOS_DIR, 'datasets')
    if os.path.isdir(droot):
        nombres = [d for d in sorted(os.listdir(droot)) if os.path.isdir(os.path.join(droot, d, 'gallery'))]
    if not nombres:
        return 'OMITIDO', ('Sin datasets adicionales. Convierte PRAI-1581 / UAV-Human / AG-VPReID / Market-1501 a '
                           'datos/datasets/<nombre>/{gallery,query}/<id>/*.jpg')
    modelos = resolver_modelos(ctx.args.modelos)
    pm = modelo_principal(ctx)
    filas, cross = [], []
    todos = [(None, 'propio')] + [(n, n) for n in nombres]
    for clave, nombre in todos:
        ds = cargar_dataset(ctx, clave)
        if ds is None:
            continue
        for mod in modelos:
            try:
                dg, dq = deep_embeddings(ctx, ds, mod)
                r = resumen_metricas(metricas_ranking(coseno(dq, dg), ds.q_ids, ds.g_ids))
                filas.append({'Dataset': nombre, 'Modelo': mod, 'Rank-1 (%)': r['Rank-1 (%)'], 'mAP (%)': r['mAP (%)'],
                              'identidades': len(set(ds.g_ids)), 'Entrenamiento': 'preentrenado (zero-shot)'})
                if mod == pm and nombre != 'propio':
                    cross.append(r['Rank-1 (%)'])
            except Exception as e:
                filas.append({'Dataset': nombre, 'Modelo': mod, 'Rank-1 (%)': None,
                              'Entrenamiento': f'ERROR {type(e).__name__}'})
    ctx.tabla('t26_generalizacion', 'Tabla 26: generalización entre datasets (zero-shot, sin reentrenar)', filas)
    if cross:
        ctx.clave['cross_rank1'] = float(np.mean(cross))
    ctx.nota('generalizacion', 'Las filas "PRAI→UAV-Human", "Mixto + fine-tuning" del documento requieren ENTRENAR; '
                               'aquí solo se mide el desempeño de los modelos preentrenados en cada dataset.')
    return 'OK', ''


# ============================================================================
# ACCESO AL SISTEMA REAL (deteccion_reid_dron_contador.py) - solo lectura
# ============================================================================
_D = {'mod': None, 'intentado': False, 'desactivado': False}


def sistema():
    """Importa el módulo real para reutilizar PersonTrack, match_detections_to_tracks, identificar y
    sus constantes (SCALE_PERCENT, MIN_CONFIDENCE...). No ejecuta main() ni abre ventanas."""
    if _D['desactivado']:
        return None
    if not _D['intentado']:
        _D['intentado'] = True
        try:
            import deteccion_reid_dron_contador as D
            _D['mod'] = D
        except Exception as e:
            log(f'⚠️  No se pudo importar deteccion_reid_dron_contador ({type(e).__name__}: {e}). '
                f'Se usan valores por defecto del sistema.')
    return _D['mod']


def _sist(nombre, default):
    D = sistema()
    return getattr(D, nombre, default) if D is not None else default


def get_detector(ctx):
    if 'yolo' not in ctx.cache:
        from ultralytics import YOLO
        peso = ctx.args.yolo or (os.path.join(APP_DIR, 'yolov8n.pt') if os.path.exists(os.path.join(APP_DIR, 'yolov8n.pt'))
                                 else 'yolov8n.pt')
        m = YOLO(peso)
        m.to(ctx.device)
        ctx.cache['yolo'] = m
        ctx.cache['yolo_peso'] = peso
    return ctx.cache['yolo']


def detectar_como_sistema(ctx, frame, conf=0.05):
    """Misma preparación que main(): reescala a SCALE_PERCENT, imgsz=640, clase persona;
    devuelve cajas en coordenadas del frame original."""
    det = get_detector(ctx)
    esc = _sist('SCALE_PERCENT', 60)
    if esc != 100:
        small = cv2.resize(frame, (int(frame.shape[1] * esc / 100), int(frame.shape[0] * esc / 100)))
        f = 100.0 / esc
    else:
        small, f = frame, 1.0
    r = det(small, verbose=False, imgsz=640, conf=conf)[0]
    return [(*[float(v) * f for v in b.xyxy[0]], float(b.conf[0])) for b in r.boxes if int(b.cls[0]) == 0]


# ============================================================================
# BLOQUES DE DETECCIÓN (Tablas 1-6)
# ============================================================================
def leer_etiquetas_yolo(ruta, W, H):
    cajas = []
    if os.path.exists(ruta):
        with open(ruta, encoding='utf-8') as fh:
            for linea in fh:
                p = linea.split()
                if len(p) >= 5 and int(float(p[0])) == 0:
                    cx, cy, w, h = map(float, p[1:5])
                    cajas.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
    return np.array(cajas, dtype=np.float64).reshape(-1, 4)


def iou_matriz(A, B):
    A, B = np.asarray(A, dtype=np.float64).reshape(-1, 4), np.asarray(B, dtype=np.float64).reshape(-1, 4)
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    x1 = np.maximum(A[:, None, 0], B[None, :, 0])
    y1 = np.maximum(A[:, None, 1], B[None, :, 1])
    x2 = np.minimum(A[:, None, 2], B[None, :, 2])
    y2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    ab = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def emparejar(gt, preds, iou_t=0.5):
    """Asignación voraz por confianza. Devuelve (flags_tp por predicción, índice GT emparejado o -1, IoU)."""
    orden = sorted(range(len(preds)), key=lambda i: -preds[i][4])
    usado = set()
    flags, par, ious = {}, {}, {}
    M = iou_matriz(gt, [p[:4] for p in preds]) if len(gt) and len(preds) else None
    for i in orden:
        mejor, jm = iou_t, -1
        if M is not None:
            for j in range(len(gt)):
                if j not in usado and M[j, i] >= mejor:
                    mejor, jm = M[j, i], j
        flags[i], par[i], ious[i] = jm >= 0, jm, (float(M[jm, i]) if jm >= 0 else 0.0)
        if jm >= 0:
            usado.add(jm)
    return flags, par, ious


def evaluar_detecciones(gts, preds, conf_op, iou_t=0.5):
    tp = fp = fn = 0
    reg = []
    n_gt = int(sum(len(g) for g in gts))
    for g, p in zip(gts, preds):
        flags, _, _ = emparejar(g, p, iou_t)
        for i, d in enumerate(p):
            reg.append((d[4], flags[i]))
        p_op = [d for d in p if d[4] >= conf_op]
        f_op, _, _ = emparejar(g, p_op, iou_t)
        t = sum(f_op.values())
        tp += t
        fp += len(p_op) - t
        fn += len(g) - t
    prec, rec = tp / max(tp + fp, 1), tp / max(n_gt, 1)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    ap = 0.0
    if reg and n_gt:
        reg.sort(key=lambda x: -x[0])
        tpc = np.cumsum([r[1] for r in reg])
        fpc = np.cumsum([not r[1] for r in reg])
        recall, precision = tpc / n_gt, tpc / np.maximum(tpc + fpc, 1)
        ap = float(np.mean([precision[recall >= r].max() if (recall >= r).any() else 0.0
                            for r in np.linspace(0, 1, 101)]))
    return {'TP': tp, 'FP': fp, 'FN': fn, 'Precision': prec, 'Recall': rec, 'F1': f1, 'mAP@0.5': ap, 'GT': n_gt}


CONDICIONES_DET = OrderedDict([
    ('baja_altura', 'P 0.94-0.97 / R 0.92-0.96 / mAP 0.93-0.97'),
    ('altura_media', 'P 0.90-0.95 / R 0.86-0.92 / mAP 0.89-0.94'),
    ('altura_alta', 'P 0.80-0.88 / R 0.72-0.84 / mAP 0.78-0.88'),
    ('oclusion', 'P 0.82-0.90 / R 0.70-0.82 / mAP 0.76-0.86'),
    ('grupos', '(FP, FN, mAP)'), ('fondo_complejo', 'P 0.85-0.92 / R 0.78-0.88 / mAP 0.84-0.91')])


def _condiciones_con_datos(sub, requiere_labels=True):
    raiz = os.path.join(DATOS_DIR, sub)
    out = []
    if os.path.isdir(raiz):
        for c in sorted(os.listdir(raiz)):
            im = os.path.join(raiz, c, 'images')
            lb = os.path.join(raiz, c, 'labels')
            if os.path.isdir(im) and _listar_imgs(im) and (not requiere_labels or os.path.isdir(lb)):
                out.append(c)
    return out


def _correr_deteccion(ctx, cond):
    clave = f'det:{cond}'
    if clave in ctx.cache:
        return ctx.cache[clave]
    im_dir = os.path.join(DATOS_DIR, 'deteccion', cond, 'images')
    lb_dir = os.path.join(DATOS_DIR, 'deteccion', cond, 'labels')
    muestras = []
    archivos = _listar_imgs(im_dir)
    if ctx.args.rapido:
        archivos = archivos[:20]
    for f in archivos:
        img = cv2.imread(os.path.join(im_dir, f))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = leer_etiquetas_yolo(os.path.join(lb_dir, os.path.splitext(f)[0] + '.txt'), W, H)
        muestras.append((img, gt, detectar_como_sistema(ctx, img)))
    ctx.cache[clave] = muestras
    return muestras


def bloque_deteccion(ctx):
    conds = _condiciones_con_datos('deteccion')
    if not conds:
        return 'OMITIDO', 'Sin datos: datos/deteccion/<condicion>/{images,labels} (etiquetas YOLO, clase 0 = persona).'
    conf_op = _sist('MIN_CONFIDENCE', 0.4)
    filas, aps = [], []
    for c in conds:
        mu = _correr_deteccion(ctx, c)
        r = evaluar_detecciones([m[1] for m in mu], [m[2] for m in mu], conf_op)
        filas.append({'Condición': c, 'Imágenes': len(mu), 'GT': r['GT'], 'TP': r['TP'], 'FP': r['FP'], 'FN': r['FN'],
                      'Precision': r['Precision'], 'Recall': r['Recall'], 'F1': r['F1'], 'mAP@0.5': r['mAP@0.5'],
                      'Esperado (doc)': CONDICIONES_DET.get(c, '')})
        aps.append(r['mAP@0.5'])
    ctx.tabla('t01_02_deteccion', f'Tablas 1-2: detección de personas (YOLO {os.path.basename(ctx.cache.get("yolo_peso", ""))}, '
                                  f'conf ≥ {conf_op}, mismo preprocesado que el sistema)', filas)
    ctx.clave['det_map50'] = 100 * float(np.mean(aps))
    return 'OK', ''


class Pose:
    def __init__(self, device):
        from ultralytics import YOLO
        self.m = YOLO('yolov8n-pose.pt')
        self.m.to(device)

    def visibilidad(self, img, umbral=0.5):
        r = self.m(img, verbose=False, imgsz=256, conf=0.25)[0]
        if r.keypoints is None or r.keypoints.conf is None or len(r.keypoints.conf) == 0:
            return None
        k = r.keypoints.conf[int(r.boxes.conf.argmax())].cpu().numpy()
        torso = all(k[i] > umbral for i in (5, 6, 11, 12))
        cuerpo = torso and all(k[i] > umbral for i in (13, 14, 15, 16))
        return {'torso': bool(torso), 'cuerpo': bool(cuerpo)}


def bloque_recorte(ctx):
    conds = _condiciones_con_datos('deteccion')
    if not conds:
        return 'OMITIDO', 'Sin datos: datos/deteccion/<condicion>/{images,labels}.'
    conf_op = _sist('MIN_CONFIDENCE', 0.4)
    pose = None
    if not ctx.args.sin_pose:
        try:
            pose = Pose(ctx.device)
        except Exception as e:
            ctx.nota('recorte', f'Torso/cuerpo por pose omitidos ({type(e).__name__}); se usa solo el criterio de borde.')
    seg = get_segmentador(ctx)
    iou_l, borde, rost, torso, sil, n = [], [], [], [], [], 0
    for c in conds:
        for img, gt, pr in _correr_deteccion(ctx, c):
            H, W = img.shape[:2]
            p_op = [d for d in pr if d[4] >= conf_op]
            flags, par, ious = emparejar(gt, p_op)
            for i, d in enumerate(p_op):
                if not flags[i]:
                    continue
                n += 1
                iou_l.append(ious[i])
                x1, y1, x2, y2 = [int(v) for v in d[:4]]
                borde.append(x1 > 2 and y1 > 2 and x2 < W - 2 and y2 < H - 2)
                crop = img[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, (CROP_HW[1], CROP_HW[0]))
                rr = detectar_rostro_haar(crop)
                if rr is not None:
                    rost.append(rr)
                if pose is not None:
                    v = pose.visibilidad(crop)
                    if v is not None:
                        torso.append(v['torso'])
                if seg is not None:
                    sil.append(seg.descriptor(crop) is not None)
    if n == 0:
        return 'OMITIDO', 'Ninguna detección coincidió con las etiquetas (IoU ≥ 0.5).'
    pct = lambda l: 100 * float(np.mean(l)) if l else None
    filas = [
        {'Elemento evaluado': 'IoU promedio (caja automática vs real)', 'Resultado': float(np.mean(iou_l)), 'Esperado (doc)': '0.78-0.88'},
        {'Elemento evaluado': 'Recortes con cuerpo completo (no cortado por el borde) %', 'Resultado': pct(borde), 'Esperado (doc)': '80-92%'},
        {'Elemento evaluado': 'Recortes con rostro visible (Haar frontal/perfil) %', 'Resultado': pct(rost), 'Esperado (doc)': '40-70%'},
        {'Elemento evaluado': 'Recortes con torso visible (pose) %', 'Resultado': pct(torso), 'Esperado (doc)': '85-95%'},
        {'Elemento evaluado': 'Siluetas segmentables (yolov8n-seg) %', 'Resultado': pct(sil), 'Esperado (doc)': '75-90%'}]
    ctx.tabla('t03_04_recorte', f'Tablas 3-4: calidad del recorte ({n} detecciones emparejadas con GT)', filas)
    ctx.nota('recorte', 'Rostro/torso/silueta son estimaciones automáticas (Haar, pose y segmentación), no anotaciones humanas.')
    return 'OK', ''


def bloque_rostro(ctx):
    conds = []
    raiz = os.path.join(DATOS_DIR, 'rostro')
    if os.path.isdir(raiz):
        conds = [c for c in sorted(os.listdir(raiz)) if os.path.isdir(os.path.join(raiz, c)) and
                 any(f.lower().endswith(IMG_EXT) for _, _, fs in os.walk(os.path.join(raiz, c)) for f in fs)]
    filas = []
    for c in conds:
        det = tot = 0
        for d, _, fs in os.walk(os.path.join(raiz, c)):
            for f in fs:
                if f.lower().endswith(IMG_EXT):
                    im = cv2.imread(os.path.join(d, f))
                    if im is None:
                        continue
                    r = detectar_rostro_haar(cv2.resize(im, (CROP_HW[1], CROP_HW[0])))
                    if r is None:
                        return 'OMITIDO', 'Esta instalación de OpenCV no incluye los clasificadores Haar.'
                    det += int(r)
                    tot += 1
        filas.append({'Condición facial': c, 'Imágenes': tot, 'Tasa de detección facial (%)': 100 * det / max(tot, 1)})
    if filas:
        ctx.tabla('t05_06_rostro_deteccion', 'Tablas 5-6: detección facial por condición de vista', filas)
    # verificación facial (solo si hay módulo facial y dataset Re-ID)
    ds = cargar_dataset(ctx)
    if ds is not None and get_rostros(ctx) is not None:
        from sklearn.metrics import roc_auc_score
        b = obtener_fusion_base(ctx, ds, modelo_principal(ctx))
        S = b.sims.get('rostro')
        if S is not None:
            ok = ~np.isnan(S)
            etq = (ds.q_ids[:, None] == ds.g_ids[None, :])[ok]
            sc = S[ok]
            if etq.any() and (~etq).any():
                cur = _curva_pares(sc, etq)
                mj = max(cur, key=lambda r: r['f1'])
                ctx.tabla('t05_verificacion_facial', 'Tabla 5: verificación facial (rostro consulta vs galería)',
                          [{'Pares con rostro en ambos': int(ok.sum()), 'ROC-AUC': float(roc_auc_score(etq, sc)),
                            'Precision': mj['precision'], 'Recall': mj['recall'], 'F1': mj['f1'], 'Umbral': mj['umbral']}])
    elif not filas:
        return 'OMITIDO', 'Sin datos: datos/rostro/<condicion>/*.jpg y/o módulo facial (insightface).'
    ctx.nota('rostro', 'F1 de reconocimiento por condición de vista requiere imágenes con identidad por condición '
                       '(datos/rostro/<condicion>/<id>/*.jpg) + insightface: no incluido en esta versión.')
    return 'OK', ''


def _curva_pares(score, etq):
    filas = []
    for t in np.unique(np.quantile(score, np.linspace(0, 1, 101))):
        acc = score >= t
        tp = int((acc & etq).sum())
        p, r = tp / max(int(acc.sum()), 1), tp / max(int(etq.sum()), 1)
        filas.append({'umbral': float(t), 'precision': p, 'recall': r, 'f1': 2 * p * r / (p + r) if p + r else 0.0})
    return filas


# ============================================================================
# PIPELINE ESPEJO (headless) DE deteccion_reid_dron_contador.py
# ============================================================================
def _cargar_base(ctx, modelo, ext):
    import pickle
    try:
        from feature_extractors import get_base_path
        p = get_base_path(modelo)
        if os.path.exists(p):
            with open(p, 'rb') as fh:
                return pickle.load(fh)
    except Exception:
        pass
    ds = cargar_dataset(ctx)
    if ds is not None:
        dg, _ = deep_embeddings(ctx, ds, modelo)
        return [{'nombre': pid, 'embedding': dg[ds.g_ids == pid].mean(0)} for pid in sorted(set(ds.g_ids))]
    return []


def _es_archivo(fuente):
    return os.path.exists(str(fuente))


def correr_pipeline(ctx, fuente, modelo, max_frames=300):
    """Repite el bucle de main() sin ventana: detección -> asociación IoU -> Re-ID cada
    REID_UPDATE_INTERVAL frames, usando las funciones REALES del sistema."""
    D = sistema()
    if D is None:
        raise RuntimeError('No se pudo importar deteccion_reid_dron_contador.py')
    import torch
    det = get_detector(ctx)
    ext = construir_extractor(modelo, ctx.device)
    base = _cargar_base(ctx, modelo, ext)
    try:
        from feature_extractors import get_threshold
        umbral = get_threshold(modelo)
    except Exception:
        umbral = D.IDENTIFICATION_THRESHOLD
    es_arch = _es_archivo(fuente)
    cap = cv2.VideoCapture(str(fuente)) if es_arch else cv2.VideoCapture(str(fuente), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise IOError(f'No se pudo abrir la fuente de video: {fuente}')
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if ctx.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    D.PersonTrack._next_id = 1
    tracks = []
    t_frame, t_det, t_reid, n_personas = [], [], [], []
    salidas, excepciones, frames, fallos_lectura, frames_reid = [], 0, 0, 0, 0
    # calentamiento (no se mide)
    ok, fr0 = cap.read()
    if ok:
        detectar_como_sistema(ctx, fr0, conf=D.MIN_CONFIDENCE)
        a_numpy(ext(cv2.resize(fr0, (128, 256))))
    t_ini = time.perf_counter()
    while frames < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            if es_arch:
                break
            fallos_lectura += 1
            if fallos_lectura >= D.MAX_CONSECUTIVE_FRAME_FAILURES * 3:
                break
            continue
        frames += 1
        try:
            t0 = time.perf_counter()
            esc = D.SCALE_PERCENT
            if esc != 100:
                small = cv2.resize(frame, (int(frame.shape[1] * esc / 100), int(frame.shape[0] * esc / 100)))
                sf = 100 / esc
            else:
                small, sf = frame, 1.0
            td = time.perf_counter()
            res = det(small, verbose=False, imgsz=640)
            detecciones, crops = [], []
            for r in res:
                for box in r.boxes:
                    if int(box.cls[0]) == 0 and float(box.conf[0]) >= D.MIN_CONFIDENCE:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x1, y1, x2, y2 = int(x1 * sf), int(y1 * sf), int(x2 * sf), int(y2 * sf)
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
                        if x2 > x1 and y2 > y1:
                            detecciones.append((x1, y1, x2, y2))
                            crops.append(frame[y1:y2, x1:x2])
            t_det.append((time.perf_counter() - td) * 1000)
            matched, un_d, un_t = D.match_detections_to_tracks(detecciones, tracks)
            for di, ti in matched:
                tracks[ti].update(detecciones[di], frames)
            for di in un_d:
                tracks.append(D.PersonTrack(detecciones[di], frames))
            for ti in un_t:
                tracks[ti].increment_age()
            for i, tr in enumerate(tracks):
                if tr.is_confirmed() and tr.age == 0 and (frames - tr.last_reid_frame) >= D.REID_UPDATE_INTERVAL:
                    ci = next((d for d, t in matched if t == i), None)
                    if ci is not None and ci < len(crops) and crops[ci].size > 0:
                        tr_ = time.perf_counter()
                        emb = ext(cv2.resize(crops[ci], (128, 256)))[0]
                        ident, conf = D.identificar(emb, base, threshold=umbral)
                        t_reid.append((time.perf_counter() - tr_) * 1000)
                        tr.update_identity(ident, conf, frames)
                        frames_reid += 1
            tracks = [t for t in tracks if t.is_alive()]
            if ctx.device == 'cuda':
                torch.cuda.synchronize()
            t_frame.append((time.perf_counter() - t0) * 1000)
            n_personas.append(len(detecciones))
            salidas.append([(t.track_id, *t.bbox) for t in tracks if t.is_confirmed() and t.age == 0])
        except Exception:
            excepciones += 1
            salidas.append([])
    dur = max(time.perf_counter() - t_ini, 1e-9)
    cap.release()
    mem_gpu = torch.cuda.max_memory_allocated() / 1e9 if ctx.device == 'cuda' else float('nan')
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / 1e9
    except Exception:
        rss = float('nan')
    fps = frames / dur
    return {'frames': frames, 'fps': fps, 'fps_video': fps_video,
            'lat_p50': float(np.percentile(t_frame, 50)) if t_frame else float('nan'),
            'lat_p95': float(np.percentile(t_frame, 95)) if t_frame else float('nan'),
            'det_ms': float(np.mean(t_det)) if t_det else float('nan'),
            'reid_ms': float(np.mean(t_reid)) if t_reid else float('nan'),
            'personas_frame': float(np.mean(n_personas)) if n_personas else 0.0,
            'mem_gpu_gb': mem_gpu, 'rss_gb': rss, 'excepciones': excepciones,
            'perdida_pct': 100 * max(0.0, 1 - fps / fps_video), 'estabilidad_pct': 100 * (1 - excepciones / max(frames, 1)),
            'fallos_lectura': fallos_lectura, 'salidas': salidas, 'identificaciones': frames_reid, 'es_archivo': es_arch}


def _fuente_video(ctx):
    if ctx.args.video:
        return ctx.args.video
    vd = os.path.join(DATOS_DIR, 'videos')
    if os.path.isdir(vd):
        for f in sorted(os.listdir(vd)):
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                return os.path.join(vd, f)
    return None


def _info_hw(ctx):
    try:
        import torch
        if ctx.device == 'cuda':
            return f'GPU {torch.cuda.get_device_name(0)}'
    except Exception:
        pass
    import platform
    return f'CPU {platform.processor() or platform.machine()}'


def bloque_eficiencia(ctx):
    fuente = _fuente_video(ctx)
    if not fuente:
        return 'OMITIDO', 'Sin video: pasa --video ruta.mp4 / rtmp://... o coloca uno en datos/videos/.'
    modelos = resolver_modelos(ctx.args.modelos)
    hw = _info_hw(ctx)
    filas = []
    for mod in modelos:
        try:
            log(f'\n⏱️  Pipeline completo con {mod} sobre {os.path.basename(str(fuente))} '
                f'(máx {ctx.args.max_frames} frames, {hw})')
            r = correr_pipeline(ctx, fuente, mod, ctx.args.max_frames)
            filas.append({'Configuración': f'{mod} @ {hw}', 'FPS': r['fps'], 'Latencia p50 (ms)': r['lat_p50'],
                          'Latencia p95 (ms)': r['lat_p95'], 'YOLO (ms)': r['det_ms'], 'Re-ID (ms)': r['reid_ms'],
                          'Memoria GPU pico (GB)': r['mem_gpu_gb'], 'RAM proceso (GB)': r['rss_gb'],
                          'Pérdida estimada (%)': r['perdida_pct'], 'Estabilidad (%)': r['estabilidad_pct'],
                          'Personas/frame': r['personas_frame'], 'frames': r['frames']})
            if mod == modelo_principal(ctx):
                ctx.clave.update({'fps': r['fps'], 'latencia_ms': r['lat_p50'], 'mem_gb': r['mem_gpu_gb'],
                                  'perdida_pct': r['perdida_pct'], 'estabilidad_pct': r['estabilidad_pct']})
        except Exception as e:
            filas.append({'Configuración': f'{mod} @ {hw}', 'FPS': None, 'Nota': f'ERROR {type(e).__name__}: {str(e)[:80]}'})
            log(f'   ❌ {mod}: {type(e).__name__}: {e}')
    ctx.tabla('t22_23_eficiencia', 'Tablas 22-23: eficiencia del pipeline completo (YOLO + tracking + Re-ID)', filas)
    ctx.nota('eficiencia', 'Pérdida estimada = fracción de frames que se perderían en vivo si el procesamiento es más '
                           'lento que el video. Para Jetson Nano/Xavier/Orin ejecuta esta misma suite en cada equipo.')
    return 'OK', ''


# ============================================================================
# BLOQUE MOT: ID switches / IDF1 / MOTA
# ============================================================================
def leer_gt_mot(ruta):
    gt = defaultdict(list)
    with open(ruta, encoding='utf-8') as fh:
        for linea in fh:
            p = linea.strip().split(',')
            if len(p) < 6:
                continue
            fr, pid, x, y, w, h = int(float(p[0])), int(float(p[1])), *map(float, p[2:6])
            gt[fr].append((pid, x, y, x + w, y + h))
    return gt


def metricas_mot(gt, pred, iou_t=0.5):
    from scipy.optimize import linear_sum_assignment
    co = defaultdict(int)
    fn = fp = idsw = n_gt = n_pred = 0
    last = {}
    for fr in sorted(set(gt) | set(pred)):
        G, P = gt.get(fr, []), pred.get(fr, [])
        n_gt += len(G)
        n_pred += len(P)
        if not G or not P:
            fn += len(G)
            fp += len(P)
            continue
        M = iou_matriz([g[1:] for g in G], [p[1:] for p in P])
        r, c = linear_sum_assignment(-M)
        ok = [(i, j) for i, j in zip(r, c) if M[i, j] >= iou_t]
        for i, j in ok:
            gid, pid = G[i][0], P[j][0]
            co[(gid, pid)] += 1
            if gid in last and last[gid] != pid:
                idsw += 1
            last[gid] = pid
        fn += len(G) - len(ok)
        fp += len(P) - len(ok)
    mota = 1 - (fn + fp + idsw) / max(n_gt, 1)
    gids = sorted({k[0] for k in co})
    pids = sorted({k[1] for k in co})
    idtp = 0
    if gids and pids:
        M = np.zeros((len(gids), len(pids)))
        for (g, p), v in co.items():
            M[gids.index(g), pids.index(p)] = v
        r, c = linear_sum_assignment(-M)
        idtp = int(M[r, c].sum())
    idf1 = 2 * idtp / max(2 * idtp + (n_pred - idtp) + (n_gt - idtp), 1)
    return {'ID switches': idsw, 'IDF1 (%)': 100 * idf1, 'MOTA (%)': 100 * mota, 'FN': fn, 'FP': fp}


def correr_tracker_ultralytics(ctx, fuente, tracker_yaml, max_frames):
    from ultralytics import YOLO
    m = YOLO(ctx.cache.get('yolo_peso') or (ctx.args.yolo or 'yolov8n.pt'))
    m.to(ctx.device)
    esc = _sist('SCALE_PERCENT', 60)
    conf = _sist('MIN_CONFIDENCE', 0.4)
    cap = cv2.VideoCapture(str(fuente))
    salidas, n = [], 0
    while n < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        small = cv2.resize(fr, (int(fr.shape[1] * esc / 100), int(fr.shape[0] * esc / 100))) if esc != 100 else fr
        f = 100.0 / esc if esc != 100 else 1.0
        r = m.track(small, persist=True, tracker=tracker_yaml, classes=[0], conf=conf, imgsz=640, verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            salidas.append([])
        else:
            salidas.append([(int(i), *[float(v) * f for v in b]) for i, b in zip(r.boxes.id.int().tolist(),
                                                                                  r.boxes.xyxy.tolist())])
    cap.release()
    return salidas


def bloque_mot(ctx):
    vd = os.path.join(DATOS_DIR, 'videos')
    pares = []
    if ctx.args.video and os.path.exists(os.path.splitext(ctx.args.video)[0] + '_gt.txt'):
        pares.append((ctx.args.video, os.path.splitext(ctx.args.video)[0] + '_gt.txt'))
    elif os.path.isdir(vd):
        for f in sorted(os.listdir(vd)):
            b, e = os.path.splitext(f)
            if e.lower() in ('.mp4', '.avi', '.mov', '.mkv') and os.path.exists(os.path.join(vd, b + '_gt.txt')):
                pares.append((os.path.join(vd, f), os.path.join(vd, b + '_gt.txt')))
    if not pares:
        return 'OMITIDO', 'Sin video con anotación: datos/videos/<nombre>.mp4 + <nombre>_gt.txt (formato MOT: frame,id,x,y,w,h,...).'
    mod = modelo_principal(ctx)
    filas = []
    for video, gtp in pares:
        gt = leer_gt_mot(gtp)
        for nombre, fn in (('IoU del sistema (PersonTrack)', None), ('ByteTrack', 'bytetrack.yaml'), ('BoT-SORT', 'botsort.yaml')):
            try:
                t = time.perf_counter()
                sal = (correr_pipeline(ctx, video, mod, ctx.args.max_frames)['salidas'] if fn is None
                       else correr_tracker_ultralytics(ctx, video, fn, ctx.args.max_frames))
                seg = time.perf_counter() - t
                pred = {i + 1: s for i, s in enumerate(sal)}
                gt_r = {k: v for k, v in gt.items() if k <= len(sal)}
                r = metricas_mot(gt_r, pred)
                filas.append({'Video': os.path.basename(video), 'Método': nombre, 'ID switches': r['ID switches'],
                              'IDF1 (%)': r['IDF1 (%)'], 'MOTA (%)': r['MOTA (%)'], 'FN': r['FN'], 'FP': r['FP'],
                              'FPS': len(sal) / max(seg, 1e-9)})
            except Exception as e:
                filas.append({'Video': os.path.basename(video), 'Método': nombre, 'ID switches': None,
                              'Nota': f'ERROR {type(e).__name__}: {str(e)[:70]}'})
    ctx.tabla('t24_tracking_mot', 'Tabla 24: tracking temporal (ID switches, IDF1, MOTA)', filas)
    ctx.nota('mot', 'SORT y DeepSORT no vienen en ultralytics: no se incluyen. Rank-1 por método no aplica a este bloque.')
    return 'OK', ''


# ============================================================================
# SIGNIFICANCIA ESTADÍSTICA
# ============================================================================
def _pareado(a, b, escala=100.0):
    from scipy import stats
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return None
    m = ~(np.isnan(a) | np.isnan(b))
    d = (b - a)[m] * escala
    n = len(d)
    if n < 3:
        return None
    sd = d.std(ddof=1)
    t_p = float(stats.ttest_rel(b[m], a[m]).pvalue) if sd > 0 else 1.0
    try:
        w_p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        w_p = 1.0
    rng = np.random.RandomState(SEED)
    boots = [d[rng.randint(0, n, n)].mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {'n': n, 'Δ medio (pp)': float(d.mean()), 't-test p': t_p, 'Wilcoxon p': w_p,
            'IC95% inf': float(lo), 'IC95% sup': float(hi), "Cohen's d": float(d.mean() / sd) if sd > 0 else float('nan')}


def _interpretar(r):
    p = min(x for x in (r['t-test p'], r['Wilcoxon p']) if not math.isnan(x)) if r else 1
    if r is None:
        return ''
    sig = 'significativa (p<0.01)' if max(r['t-test p'], r['Wilcoxon p']) < 0.01 else \
          'significativa (p<0.05)' if max(r['t-test p'], r['Wilcoxon p']) < 0.05 else 'no significativa'
    return sig + (' [empeora]' if r['Δ medio (pp)'] < 0 and 'no' not in sig else '')


def bloque_estadistica(ctx):
    from scipy import stats
    perq = ctx.perq
    comparaciones = [
        ('Deep solo vs sistema completo', 'X:deep', 'X:completo'),
        ('Sin rostro vs con rostro', 'X:sin_rostro', 'X:con_rostro'),
        ('Sin textura vs con textura', 'X:sin_textura', 'X:con_textura'),
        ('Early fusion vs fusión híbrida', 'X:early', 'X:hibrida'),
        ('Imagen única vs promedio de 5 frames (video-based)', 'track:1', 'track:5'),
        ('Imagen única vs tracklet completo', 'track:1', 'track:all')]
    filas = []
    for nombre, ka, kb in comparaciones:
        if ka not in perq or kb not in perq:
            filas.append({'Comparación': nombre, 'Métrica': '-', 'Nota': f'omitida (falta {ka if ka not in perq else kb})'})
            continue
        for met, clave in (('mAP (AP por consulta)', 'ap'), ('Rank-1 (acierto por consulta)', 'hit1')):
            r = _pareado(perq[ka][clave], perq[kb][clave])
            if r is None:
                continue
            filas.append({'Comparación': f'{nombre} [{ka} → {kb}]', 'Métrica': met, **r, 'Interpretación': _interpretar(r)})
            if nombre.startswith('Deep solo') and clave == 'ap':
                ctx.clave['p_baseline'] = max(r['t-test p'], r['Wilcoxon p'])
    # arquitecturas: comparación contra la principal + ANOVA
    mk = [k for k in perq if k.startswith('deep:')]
    pm = 'deep:' + modelo_principal(ctx)
    if pm in perq:
        for k in mk:
            if k != pm:
                r = _pareado(perq[k]['ap'], perq[pm]['ap'])
                if r:
                    filas.append({'Comparación': f'{k[5:]} → {pm[5:]} (arquitecturas)', 'Métrica': 'mAP (AP por consulta)',
                                  **r, 'Interpretación': _interpretar(r)})
    if not any('n' in f for f in filas):
        return 'OMITIDO', 'Sin resultados previos por consulta (ejecuta antes arquitecturas, fusion y tracking con datos).'
    ctx.tabla('t31_significancia', 'Tabla 31: significancia estadística (pruebas pareadas sobre resultados por consulta)', filas)
    if len(mk) >= 3:
        aps = [np.asarray(perq[k]['ap'], dtype=float) for k in mk]
        validos = ~np.any([np.isnan(a) for a in aps], axis=0)
        if validos.sum() >= 3:
            f = stats.f_oneway(*[a[validos] for a in aps])
            fr = stats.friedmanchisquare(*[a[validos] for a in aps])
            ctx.tabla('t32_anova', 'Tabla 32: comparación global de arquitecturas',
                      [{'Prueba': 'ANOVA (f_oneway)', 'Estadístico': float(f.statistic), 'p-value': float(f.pvalue),
                        'Modelos': len(mk), 'Consultas': int(validos.sum())},
                       {'Prueba': 'Friedman (medidas repetidas, no paramétrica)', 'Estadístico': float(fr.statistic),
                        'p-value': float(fr.pvalue), 'Modelos': len(mk), 'Consultas': int(validos.sum())}])
    ctx.nota('estadistica', 'p-valores sobre consultas individuales de UN dataset: validan la mejora en esos datos; '
                            'con pocas identidades generalizan poco. Cohen\'s d pareado = Δ medio / desv. de Δ.')
    return 'OK', ''

def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _rgb(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def bloque_cualitativa(ctx):
    plt = _plt()
    hechas = 0
    cmc = ctx.cache.get('cmc')
    if cmc:
        fig, ax = plt.subplots(figsize=(6, 4))
        for m, c in cmc.items():
            ax.plot(range(1, len(c) + 1), 100 * np.asarray(c), marker='o', ms=3, label=m)
        ax.set_xlabel('Rango k'); ax.set_ylabel('Identificación (%)'); ax.set_title('Curva CMC'); ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(ctx.fig('fig_cmc.png'), dpi=150); plt.close(fig); hechas += 1
    roc = ctx.cache.get('roc')
    if roc:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        for n, d in roc.items():
            ax.plot(d['fpr'], d['tpr'], label=f"{n} (AUC {d['auroc']:.2f})")
        ax.plot([0, 1], [0, 1], 'k--', lw=.7); ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('Curva ROC (open-set)')
        ax.legend(fontsize=7); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(ctx.fig('fig_roc_openset.png'), dpi=150); plt.close(fig); hechas += 1
    pr = ctx.cache.get('pr_curve')
    if pr:
        cur, mod = pr
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot([c['recall'] for c in cur], [c['precision'] for c in cur], marker='.', ms=3)
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title(f'Curva Precision-Recall ({mod})'); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(ctx.fig('fig_precision_recall.png'), dpi=150); plt.close(fig); hechas += 1
    for clave, xk, titulo_f, nombre in (('rob_altura_m', 'Grupo', 'Rendimiento vs altura UAV', 'fig_vs_altura.png'),
                                         ('rob_oclusion', 'Oclusión (aprox.)', 'Rendimiento vs oclusión', 'fig_vs_oclusion.png'),
                                         ('rob_resolucion', 'Resolución (alto x ancho)', 'Rendimiento vs resolución del recorte', 'fig_vs_resolucion.png')):
        filas = ctx.cache.get(clave)
        if not filas:
            continue
        k_deep = next(k for k in filas[0] if 'deep' in k and 'Rank-1' in k)
        k_prop = next(k for k in filas[0] if 'propuesto' in k and 'Rank-1' in k)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([str(f[xk]) for f in filas], [f[k_deep] for f in filas], marker='o', label='Deep solo')
        ax.plot([str(f[xk]) for f in filas], [f[k_prop] for f in filas], marker='s', label='Propuesto')
        ax.set_ylabel('Rank-1 (%)'); ax.set_title(titulo_f); ax.grid(alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig(ctx.fig(nombre), dpi=150); plt.close(fig); hechas += 1
    esc = ctx.cache.get('escala')
    if esc:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot([f['Identidades en galería'] for f in esc], [f['Rank-1 (%)'] for f in esc], marker='o', label='Rank-1')
        ax.plot([f['Identidades en galería'] for f in esc], [f['mAP (%)'] for f in esc], marker='s', label='mAP')
        ax.set_xlabel('Identidades en la galería'); ax.set_title('Escalabilidad de la galería'); ax.grid(alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig(ctx.fig('fig_escalabilidad.png'), dpi=150); plt.close(fig); hechas += 1
    # heatmap + top-5 (requieren dataset y features)
    ds = cargar_dataset(ctx)
    if ds is not None:
        try:
            b = obtener_fusion_base(ctx, ds, modelo_principal(ctx))
            ids = sorted(set(ds.g_ids))
            S = b.sims['deep']
            M = np.array([[np.nanmean(S[np.ix_(ds.q_ids == qi, ds.g_ids == gj)]) for gj in ids] for qi in ids])
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            im = ax.imshow(M, cmap='viridis')
            if len(ids) <= 30:
                ax.set_xticks(range(len(ids))); ax.set_yticks(range(len(ids)))
                ax.set_xticklabels([str(i)[:10] for i in ids], rotation=90, fontsize=6)
                ax.set_yticklabels([str(i)[:10] for i in ids], fontsize=6)
            ax.set_xlabel('Galería'); ax.set_ylabel('Consulta'); ax.set_title('Heatmap de similitud (deep, media por identidad)')
            fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(ctx.fig('fig_heatmap_similitud.png'), dpi=150); plt.close(fig); hechas += 1
            for nombre, Sx in (('deep solo', S), ('propuesto (fusión)', _propuesto(b, b.sims))):
                m = metricas_ranking(Sx, ds.q_ids, ds.g_ids)
                bien = list(np.where(m['hit1'] == 1)[0][:3])
                mal = list(np.where(m['hit1'] == 0)[0][:3])
                pick = bien + mal
                if not pick:
                    continue
                fig, axes = plt.subplots(len(pick), 6, figsize=(9, 2.6 * len(pick)), squeeze=False)
                for r, qi in enumerate(pick):
                    axes[r, 0].imshow(_rgb(ds.imagen(ds.qry[qi][0])))
                    axes[r, 0].set_title(f'Query {str(ds.q_ids[qi])[:12]}', fontsize=7)
                    for c, gi in enumerate(m['order'][qi][:5]):
                        axes[r, c + 1].imshow(_rgb(ds.imagen(ds.gal[gi][0])))
                        acierto = ds.g_ids[gi] == ds.q_ids[qi]
                        for sp in axes[r, c + 1].spines.values():
                            sp.set_edgecolor('limegreen' if acierto else 'red'); sp.set_linewidth(3)
                        axes[r, c + 1].set_title(f'#{c + 1}', fontsize=7)
                    for a in axes[r]:
                        a.set_xticks([]); a.set_yticks([])
                fig.suptitle(f'Top-5 recuperados: {nombre} (verde = misma persona)', fontsize=9)
                fig.tight_layout()
                fig.savefig(ctx.fig(f'fig_top5_{"deep" if "deep" in nombre else "propuesto"}.png'), dpi=130)
                plt.close(fig); hechas += 1
        except Exception as e:
            ctx.nota('cualitativa', f'Heatmap/top-5 omitidos: {type(e).__name__}: {e}')
    if hechas == 0:
        return 'OMITIDO', 'No hay resultados previos para graficar (ejecuta antes arquitecturas/robustez/openset...).'
    ctx.nota('cualitativa', f'{hechas} figuras guardadas en {ctx.salida}. Los diagramas de arquitectura y pipeline '
                            f'(Tabla 30) deben dibujarse aparte.')
    return 'OK', ''


CRITERIOS = [  # etiqueta, clave(s), sentido, mínimo, fuerte
    ('Detección mAP@0.5 (%)', ['det_map50'], '>', 85, 90),
    ('Re-ID Rank-1 (%)', ['reid_rank1_propuesto', 'reid_rank1_deep'], '>', 82, 88),
    ('Re-ID Rank-5 (%)', ['reid_rank5_deep'], '>', 90, 95),
    ('Re-ID mAP (%)', ['reid_map_propuesto', 'reid_map_deep'], '>', 72, 80),
    ('F1-score', ['reid_f1_propuesto', 'reid_f1_deep'], '>', 0.80, 0.87),
    ('Mejora vs deep solo (pp)', ['mejora_vs_deep'], '>', 4, 7),
    ('FPS del pipeline', ['fps'], '>', 15, 25),
    ('Latencia por frame (ms)', ['latencia_ms'], '<', 100, 50),
    ('Memoria GPU (GB)', ['mem_gb'], '<', 6, 4),
    ('Pérdida de frames (%)', ['perdida_pct'], '<', 10, 5),
    ('Estabilidad del pipeline (%)', ['estabilidad_pct'], '>', 90, 95),
    ('Cross-dataset Rank-1 (%)', ['cross_rank1'], '>', 60, 70),
    ('p-value frente a baseline', ['p_baseline'], '<', 0.05, 0.01)]


def tabla_semaforo(ctx):
    filas = []
    for et, claves, sent, mn, fu in CRITERIOS:
        val = next((ctx.clave[k] for k in claves if k in ctx.clave and ctx.clave[k] is not None
                    and not (isinstance(ctx.clave[k], float) and math.isnan(ctx.clave[k]))), None)
        if val is None:
            est = '⚪ sin dato'
        elif (val > fu if sent == '>' else val < fu):
            est = '🟢 valor fuerte'
        elif (val > mn if sent == '>' else val < mn):
            est = '🟡 cumple mínimo'
        else:
            est = '🔴 no cumple'
        filas.append({'Aspecto': et, 'Obtenido': val, 'Mínimo (doc)': f'{sent}{mn}', 'Fuerte (doc)': f'{sent}{fu}', 'Estado': est})
    ctx.tabla('t34_semaforo', 'Tablas 23 y 34: valores obtenidos vs mínimos/fuertes del documento', filas)


# ============================================================================
# REGISTRO DE BLOQUES, DATOS Y SALIDAS
# ============================================================================
def _hay_imagenes(carpeta):
    return os.path.isdir(carpeta) and any(f.lower().endswith(IMG_EXT) for _, _, fs in os.walk(carpeta) for f in fs)


def _hay_reid(ctx_args=None):
    r = os.path.join(DATOS_DIR, 'reid')
    if _hay_imagenes(os.path.join(r, 'gallery')) and _hay_imagenes(os.path.join(r, 'query')):
        return True
    cfg = _config()
    pb = getattr(cfg, 'PERSONAS_BASE', None) if cfg else None
    if pb and os.path.isdir(pb):
        for d in os.listdir(pb):
            sub = os.path.join(pb, d)
            if os.path.isdir(sub) and len(_listar_imgs(sub)) >= 2:
                return True
    return False


def _hay_datasets():
    r = os.path.join(DATOS_DIR, 'datasets')
    return os.path.isdir(r) and any(_hay_imagenes(os.path.join(r, d, 'gallery')) for d in os.listdir(r)
                                    if os.path.isdir(os.path.join(r, d)))


def _hay_videos(gt=False):
    r = os.path.join(DATOS_DIR, 'videos')
    if not os.path.isdir(r):
        return False
    fs = os.listdir(r)
    if gt:
        return any(f.endswith('_gt.txt') for f in fs) and any(f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) for f in fs)
    return any(f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) for f in fs)


BLOQUES = OrderedDict([
    ('deteccion', (bloque_deteccion, 'T1-2', 'datos/deteccion/<cond>/{images,labels}', lambda: bool(_condiciones_con_datos('deteccion')), ())),
    ('recorte', (bloque_recorte, 'T3-4', 'datos/deteccion/<cond>/{images,labels}', lambda: bool(_condiciones_con_datos('deteccion')), ())),
    ('rostro', (bloque_rostro, 'T5-6', 'datos/rostro/<cond>/ (+ Re-ID)', lambda: _hay_imagenes(os.path.join(DATOS_DIR, 'rostro')), ())),
    ('softbio', (bloque_softbio, 'T7-9', 'datos/reid/{gallery,query} o personas_base', _hay_reid, ())),
    ('arquitecturas', (bloque_arquitecturas, 'T10-11, 15', 'datos/reid/{gallery,query} o personas_base', _hay_reid, ())),
    ('fusion', (bloque_fusion, 'T12-14', 'datos/reid/{gallery,query} o personas_base', _hay_reid, ())),
    ('robustez', (bloque_robustez, 'T18-21', 'datos/reid + metadata.csv (altura/ángulo)', _hay_reid, ())),
    ('openset', (bloque_openset, 'T27', 'datos/reid (≥3 identidades)', _hay_reid, ())),
    ('umbral', (bloque_umbral, 'T28', 'datos/reid o personas_base', _hay_reid, ())),
    ('escalabilidad', (bloque_escalabilidad, 'T29', 'datos/reid (+ datos/distractores)', _hay_reid, ())),
    ('tracking', (bloque_tracking, 'T25', 'datos/reid (consultas en orden temporal)', _hay_reid, ())),
    ('mot', (bloque_mot, 'T24', 'datos/videos/<n>.mp4 + <n>_gt.txt (MOT)', lambda: _hay_videos(gt=True), ())),
    ('eficiencia', (bloque_eficiencia, 'T22-23', '--video o datos/videos/*.mp4', lambda: _hay_videos(), ())),
    ('generalizacion', (bloque_generalizacion, 'T26', 'datos/datasets/<nombre>/{gallery,query}', _hay_datasets, ())),
    ('estadistica', (bloque_estadistica, 'T31-32', 'usa resultados de arquitecturas/fusion/tracking', _hay_reid, ('arquitecturas', 'fusion', 'tracking'))),
    ('cualitativa', (bloque_cualitativa, 'T30 + figuras', 'usa resultados de bloques previos', _hay_reid, ('arquitecturas', 'openset', 'umbral', 'robustez', 'escalabilidad'))),
])

LEEME = {
    'deteccion': ('Una carpeta por condición: baja_altura, altura_media, altura_alta, oclusion, grupos, fondo_complejo.\n'
                  'Dentro de cada una:  images/*.jpg  y  labels/*.txt (mismo nombre, formato YOLO: "0 cx cy w h" normalizado).\n'
                  'Clase 0 = persona. Se usa en los bloques: deteccion y recorte.\n'),
    'rostro': ('Una carpeta por condición de vista: frontal_oblicuo, lateral, cenital_parcial, cenital_extremo.\n'
               'Dentro: recortes de persona (*.jpg). Se calcula la tasa de detección facial.\n'),
    'reid': ('gallery/<id_persona>/*.jpg  y  query/<id_persona>/*.jpg  (recortes de persona).\n'
             'Opcional metadata.csv con columnas: archivo,altura_m,angulo   (archivo relativo a esta carpeta,\n'
             'p. ej. query/juan_perez_123/img_004.jpg ; angulo: frontal-oblicuo, lateral-oblicuo, posterior-oblicuo,\n'
             'cenital_parcial, cenital_extremo). Si esta carpeta no existe, se usa config.PERSONAS_BASE\n'
             '(primera mitad de cada persona = galería, segunda mitad = consultas).\n'),
    'datasets': ('Datasets adicionales (PRAI-1581, UAV-Human, AG-VPReID, Market-1501...), ya convertidos a:\n'
                 '<nombre>/gallery/<id>/*.jpg  y  <nombre>/query/<id>/*.jpg\n'),
    'distractores': 'Identidades extra solo para la galería (escalabilidad): <id>/*.jpg\n',
    'videos': ('Videos de prueba (.mp4/.avi) para eficiencia y tracking.\n'
           'Para IDF1/MOTA añade <nombre>_gt.txt en formato MOT: frame,id,x,y,w,h,conf,clase,vis (frame desde 1).\n'),
}


def init_datos():
    os.makedirs(DATOS_DIR, exist_ok=True)
    for c in CONDICIONES_DET:
        os.makedirs(os.path.join(DATOS_DIR, 'deteccion', c, 'images'), exist_ok=True)
        os.makedirs(os.path.join(DATOS_DIR, 'deteccion', c, 'labels'), exist_ok=True)
    for c in ('frontal_oblicuo', 'lateral', 'cenital_parcial', 'cenital_extremo'):
        os.makedirs(os.path.join(DATOS_DIR, 'rostro', c), exist_ok=True)
    os.makedirs(os.path.join(DATOS_DIR, 'reid', 'gallery'), exist_ok=True)
    os.makedirs(os.path.join(DATOS_DIR, 'reid', 'query'), exist_ok=True)
    for d in ('datasets', 'distractores', 'videos'):
        os.makedirs(os.path.join(DATOS_DIR, d), exist_ok=True)
    meta = os.path.join(DATOS_DIR, 'reid', 'metadata.csv')
    if not os.path.exists(meta):
        with open(meta, 'w', encoding='utf-8') as fh:
            fh.write('archivo,altura_m,angulo\n')
    for k, txt in LEEME.items():
        with open(os.path.join(DATOS_DIR, k, 'LEEME.txt'), 'w', encoding='utf-8') as fh:
            fh.write(txt)
    with open(os.path.join(DATOS_DIR, '.gitignore'), 'w', encoding='utf-8') as fh:
        fh.write('# Los datos no se versionan\n*\n!.gitignore\n')
    log(f'✅ Estructura creada en: {DATOS_DIR}')
    log('   Lee los LEEME.txt de cada carpeta. Después:  python pruebas/pruebas_uav_reid.py --listar')


def listar():
    filas = []
    for n, (_, doc, req, chk, deps) in BLOQUES.items():
        try:
            ok = bool(chk())
        except Exception:
            ok = False
        filas.append({'Bloque': n, 'Tablas del doc': doc, 'Datos requeridos': req, 'Estado': '✅ datos listos' if ok else '⚪ faltan datos'})
    log('\nBLOQUES DISPONIBLES (usa --bloques nombre1,nombre2 | todo)\n')
    imprimir_tabla(filas)
    log(f'\nCarpeta de datos: {DATOS_DIR}')


# ============================================================================
# SALIDAS: resumen.md y resultados.json
# ============================================================================
def _md_tabla(filas):
    cols = list(filas[0].keys())
    for f in filas:
        for c in f:
            if c not in cols:
                cols.append(c)
    out = ['| ' + ' | '.join(str(c) for c in cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for f in filas:
        out.append('| ' + ' | '.join(_fmt(f.get(c)).replace('|', '/') for c in cols) + ' |')
    return '\n'.join(out)


def escribir_salidas(ctx, t_total):
    L = ['# Resultados de pruebas UAV Re-ID', '',
         f'- Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}  |  Duración: {t_total / 60:.1f} min  |  Dispositivo: {ctx.device}',
         f'- Modelo principal: {modelo_principal(ctx)}  |  Modelos comparados: {", ".join(resolver_modelos(ctx.args.modelos))}',
         '', '## Estado de los bloques', '',
         _md_tabla([{'Bloque': b, 'Estado': e, 'Detalle': m} for b, (e, m) in ctx.estado.items()]), '']
    for nombre, (tit, filas) in ctx.tablas.items():
        L += [f'## {tit}', '', _md_tabla(filas), '']
    if ctx.figuras:
        L += ['## Figuras', ''] + [f'![{f}]({f})' for f in ctx.figuras] + ['']
    if ctx.notas:
        L += ['## Notas y limitaciones', ''] + [f'- **{b}**: {t}' for b, t in ctx.notas] + ['']
    with open(os.path.join(ctx.salida, 'resumen.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))
    conv = lambda o: (o.item() if hasattr(o, 'item') else str(o))
    with open(os.path.join(ctx.salida, 'resultados.json'), 'w', encoding='utf-8') as fh:
        json.dump({'estado': ctx.estado, 'metricas_clave': ctx.clave, 'notas': ctx.notas,
                   'tablas': {k: v[1] for k, v in ctx.tablas.items()}, 'args': vars(ctx.args)},
                  fh, indent=2, ensure_ascii=False, default=conv)


# ============================================================================
# DATASET SINTÉTICO PARA --autotest (no requiere datos ni GPU)
# ============================================================================
def generar_dataset_sintetico(raiz, n_ids=12, n_img=8, seed=SEED):
    rng = np.random.RandomState(seed)
    angulos = ['frontal-oblicuo', 'lateral-oblicuo', 'posterior-oblicuo', 'cenital_parcial', 'cenital_extremo']
    meta = ['archivo,altura_m,angulo']
    H, W = CROP_HW
    for pid in range(n_ids):
        camisa = tuple(int(x) for x in rng.randint(30, 255, 3))
        pantalon = tuple(int(x) for x in rng.randint(30, 255, 3))
        freq = 2 + (pid % 5) * 2
        for k in range(n_img):
            sub = 'gallery' if k < n_img // 2 else 'query'
            d = os.path.join(raiz, sub, f'persona_{pid:02d}')
            os.makedirs(d, exist_ok=True)
            img = np.full((H, W, 3), 150, np.uint8)
            img += rng.randint(0, 25, img.shape).astype(np.uint8)
            dx, dy = rng.randint(-6, 7), rng.randint(-6, 7)
            cv2.circle(img, (W // 2 + dx, 40 + dy), 18, (150, 170, 210), -1)
            cv2.rectangle(img, (25 + dx, 60 + dy), (W - 25 + dx, 140 + dy), camisa, -1)
            for yy in range(62 + dy, 138 + dy, freq):
                cv2.line(img, (25 + dx, yy), (W - 25 + dx, yy), tuple(int(c * 0.6) for c in camisa), 1)
            cv2.rectangle(img, (30 + dx, 140 + dy), (W - 30 + dx, 235 + dy), pantalon, -1)
            img = np.clip(img.astype(np.float32) * rng.uniform(0.75, 1.25) + rng.randn(*img.shape) * 6, 0, 255).astype(np.uint8)
            nombre = f'img_{k:03d}.jpg'
            cv2.imwrite(os.path.join(d, nombre), img)
            if sub == 'query':
                meta.append(f'query/persona_{pid:02d}/{nombre},{rng.uniform(5, 70):.1f},{angulos[rng.randint(0, 5)]}')
    with open(os.path.join(raiz, 'metadata.csv'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(meta) + '\n')


# ============================================================================
# EJECUCIÓN
# ============================================================================
def parsear():
    ap = argparse.ArgumentParser(description='Suite automática de pruebas UAV Re-ID (documento pruebas_Q1_UAV_ReID)')
    ap.add_argument('--init', action='store_true', help='crea la estructura de carpetas de datos')
    ap.add_argument('--listar', action='store_true', help='lista los bloques y si hay datos para cada uno')
    ap.add_argument('--autotest', action='store_true', help='prueba la suite con datos sintéticos (sin modelos ni GPU)')
    ap.add_argument('--bloques', default='todo', help='lista separada por comas, o "todo"')
    ap.add_argument('--modelos', default='sistema', help='perfil (sistema|documento|sota) o lista: osnet_x1_0,repvit,...')
    ap.add_argument('--modelo-principal', default=None, help='modelo para fusión/robustez/etc. (def.: REID_MODEL o osnet_x1_0)')
    ap.add_argument('--dispositivo', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--video', default=None, help='video o URL rtmp:// para eficiencia/tracking')
    ap.add_argument('--max-frames', type=int, default=300)
    ap.add_argument('--max-por-id', type=int, default=None, help='limita imágenes por persona')
    ap.add_argument('--rapido', action='store_true', help='prueba de humo: pocas imágenes/frames')
    ap.add_argument('--salida', default=None, help='carpeta de resultados (def.: app/data/pruebas_resultados/<fecha>)')
    ap.add_argument('--yolo', default=None, help='pesos YOLO (def.: yolov8n.pt como el sistema)')
    ap.add_argument('--fusion', default='weighted', choices=['early', 'late', 'weighted', 'hibrida'])
    ap.add_argument('--tracklet-len', type=int, default=10)
    ap.add_argument('--sin-silueta', action='store_true')
    ap.add_argument('--sin-rostro', action='store_true')
    ap.add_argument('--sin-pose', action='store_true')
    ap.add_argument('--sin-yolo-ablacion', action='store_true', help='omite A2-A12 (re-detección con YOLO)')
    ap.set_defaults(autotest_dir=None)
    return ap.parse_args()


def seleccionar(texto):
    pedidos = list(BLOQUES) if texto.strip().lower() in ('todo', 'all') else [b.strip() for b in texto.split(',') if b.strip()]
    malos = [b for b in pedidos if b not in BLOQUES]
    if malos:
        raise SystemExit(f'Bloques desconocidos: {malos}. Disponibles: {list(BLOQUES)}')
    sel = set(pedidos)
    for b in pedidos:
        sel.update(BLOQUES[b][4])
    return [b for b in BLOQUES if b in sel]


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    args = parsear()
    if args.init:
        return init_datos()
    if args.listar:
        return listar()
    if args.rapido:
        args.max_por_id = args.max_por_id or 6
        args.max_frames = min(args.max_frames, 100)
    tmp = None
    if args.autotest:
        tmp = tempfile.mkdtemp(prefix='uav_reid_autotest_')
        generar_dataset_sintetico(os.path.join(tmp, 'datos'))
        args.autotest_dir = os.path.join(tmp, 'datos')
        args.bloques = 'softbio,arquitecturas,fusion,robustez,openset,umbral,escalabilidad,tracking,estadistica,cualitativa'
        args.modelos, args.sin_yolo_ablacion = '_sintetico', True
        args.salida = args.salida or os.path.join(tmp, 'resultados')
        _D['desactivado'] = True
        titulo('AUTOTEST con datos sintéticos (valida la suite; los números NO son resultados reales)')
    if args.salida:
        salida = args.salida
    else:
        cfg = _config()
        base = getattr(cfg, 'DATA_DIR', os.path.join(APP_DIR, 'data')) if cfg else os.path.join(APP_DIR, 'data')
        salida = os.path.join(base, 'pruebas_resultados', datetime.now().strftime('%Y%m%d_%H%M%S'))
    orden = seleccionar(args.bloques)
    ctx = Ctx(args, salida)
    titulo(f'SUITE UAV Re-ID | bloques: {", ".join(orden)} | dispositivo: {ctx.device} | salida: {salida}')
    t0 = time.time()
    for nombre in orden:
        fn, doc, _, _, _ = BLOQUES[nombre]
        titulo(f'BLOQUE {nombre.upper()}  ({doc})')
        t = time.time()
        try:
            est, motivo = fn(ctx)
        except Exception as e:
            est, motivo = 'ERROR', f'{type(e).__name__}: {e}'
            log(traceback.format_exc(limit=4))
        ctx.estado[nombre] = (est, motivo)
        simb = {'OK': '✅', 'OMITIDO': '⚪', 'ERROR': '❌'}.get(est, '')
        log(f'\n{simb} {nombre}: {est} {("- " + motivo) if motivo else ""}  ({time.time() - t:.1f}s)')
    tabla_semaforo(ctx)
    escribir_salidas(ctx, time.time() - t0)
    titulo('FIN')
    for n, (e, m) in ctx.estado.items():
        log(f'  {n:<16} {e:<9} {m}')
    log(f'\n📁 Resultados: {salida}\n   Abre resumen.md (tablas + figuras) y los CSV por tabla.')
    if tmp:
        log('   (autotest: carpeta temporal, se puede borrar)')
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)