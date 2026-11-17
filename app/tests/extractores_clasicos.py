"""
EXTRACTORES CLÁSICOS: HOG, LBP y HSV
================================================================================
Usa los MISMOS parámetros que app/visualization_handler.py:
    HOG: 9 orientaciones, celdas 8x8, bloques 2x2 (skimage)
    LBP: 24 puntos, radio 3, método 'uniform'
    HSV: histogramas H/S/V por regiones del cuerpo (cabeza / torso / piernas)

Combinaciones evaluadas (7):  hog | lbp | hsv | hog+lbp | hog+hsv | lbp+hsv | hog+lbp+hsv
Las combinaciones concatenan los descriptores (cada uno normalizado L2) con el mismo peso.

Uso (desde app/):
    python pruebas/extractores_clasicos.py --generar-base     # crea data/base_extractores.pkl
    python pruebas/extractores_clasicos.py --evaluar          # evalúa las 7 combinaciones con tus fotos de webcam
    python pruebas/extractores_clasicos.py --evaluar --con-deep --modelos sistema   # + los 5 modelos profundos y fusiones
    python pruebas/extractores_clasicos.py --todo
Salidas:
    data/base_extractores.pkl           (la usa el monitor en tiempo real)
    data/umbrales_clasicos.json         (umbral óptimo por combinación, lo usa el monitor)
    data/pruebas_resultados/extractores_<fecha>/   (CSV, resumen.md, figuras)
"""
import os
import sys
import csv
import json
import time
import pickle
import argparse
from datetime import datetime

import numpy as np
import cv2

PRUEBAS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(PRUEBAS_DIR)
for _d in (APP_DIR, PRUEBAS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
ORDEN = ('hog', 'lbp', 'hsv')
COMBOS = [('hog',), ('lbp',), ('hsv',), ('hog', 'lbp'), ('hog', 'hsv'), ('lbp', 'hsv'), ('hog', 'lbp', 'hsv')]
TIPO_COMBO = {1: 'individual', 2: 'pareja', 3: 'triple'}
PARAMS = {'hog_orientations': 9, 'hog_pixels_per_cell': (8, 8), 'hog_cells_per_block': (2, 2),
          'lbp_points': 24, 'lbp_radius': 3, 'lbp_method': 'uniform', 'crop_wh': (128, 256)}
REGIONES = ((0.0, 0.20), (0.20, 0.55), (0.55, 1.0))      # cabeza / torso / piernas


def nombre_combo(c):
    return '+'.join(c)


NOMBRES_COMBO = [nombre_combo(c) for c in COMBOS]


def _data_dir():
    try:
        import config
        return config.DATA_DIR, config.PERSONAS_BASE
    except Exception:
        d = os.path.join(APP_DIR, 'data')
        return d, os.path.join(d, 'personas_base')


DATA_DIR, PERSONAS_BASE = _data_dir()
BASE_CLASICA = os.path.join(DATA_DIR, 'base_extractores.pkl')
UMBRALES_CLASICOS = os.path.join(DATA_DIR, 'umbrales_clasicos.json')


# ----------------------------------------------------------------------------
# Descriptores
# ----------------------------------------------------------------------------
def _l2(v):
    v = np.asarray(v, dtype=np.float32).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _hell(h):
    h = np.asarray(h, dtype=np.float32).ravel()
    s = h.sum()
    return np.sqrt(h / s) if s > 0 else h


def _regiones(img_o_gris):
    h = img_o_gris.shape[0]
    return [img_o_gris[int(a * h):max(int(b * h), int(a * h) + 1)] for a, b in REGIONES]


def d_hog(img):
    from skimage.feature import hog
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fd = hog(g, orientations=PARAMS['hog_orientations'], pixels_per_cell=PARAMS['hog_pixels_per_cell'],
             cells_per_block=PARAMS['hog_cells_per_block'], feature_vector=True)
    return _l2(fd)


def d_lbp(img):
    from skimage.feature import local_binary_pattern
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    P, R = PARAMS['lbp_points'], PARAMS['lbp_radius']
    hs = []
    for reg in _regiones(g):
        lbp = local_binary_pattern(reg, P, R, method=PARAMS['lbp_method'])
        hs.append(_hell(np.bincount(lbp.astype(int).ravel(), minlength=P + 2)[:P + 2]))
    return _l2(np.concatenate(hs))


def d_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hs = []
    for reg in _regiones(hsv):
        hs += [_hell(cv2.calcHist([reg], [0], None, [16], [0, 180])),
               _hell(cv2.calcHist([reg], [1], None, [8], [0, 256])),
               _hell(cv2.calcHist([reg], [2], None, [8], [0, 256]))]
    return _l2(np.concatenate(hs))


FUNCS = {'hog': d_hog, 'lbp': d_lbp, 'hsv': d_hsv}


class ExtractoresClasicos:
    """descriptores(img) -> {'hog': v, 'lbp': v, 'hsv': v}  (cada uno con norma L2 = 1)."""

    def descriptores(self, img, medir=False):
        w, h = PARAMS['crop_wh']
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        out, tms = {}, {}
        for k in ORDEN:
            t = time.perf_counter()
            out[k] = FUNCS[k](img)
            tms[k] = (time.perf_counter() - t) * 1000
        return (out, tms) if medir else out

    @staticmethod
    def vector_combo(desc, combo):
        return np.concatenate([desc[k] for k in combo]) / np.sqrt(len(combo))


def parsear_persona(carpeta):
    partes = carpeta.split('_')
    if len(partes) >= 2:
        return ' '.join(partes[:-1]).replace('_', ' ').title(), partes[-1]
    return carpeta.replace('_', ' ').title(), '0000000000'


def listar_personas(personas_dir=None, max_por_persona=None):
    """[(nombre, cedula, [rutas ordenadas])] de data/personas_base."""
    pd = personas_dir or PERSONAS_BASE
    out = []
    if not os.path.isdir(pd):
        return out
    for c in sorted(os.listdir(pd)):
        d = os.path.join(pd, c)
        if not os.path.isdir(d):
            continue
        fs = sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))
        if max_por_persona:
            fs = fs[:max_por_persona]
        if fs:
            n, ced = parsear_persona(c)
            out.append((n, ced, [os.path.join(d, f) for f in fs]))
    return out


# ----------------------------------------------------------------------------
# Base de extractores clásicos (equivalente a base_personas.pkl pero para HOG/LBP/HSV)
# ----------------------------------------------------------------------------
class BaseClasica:
    def __init__(self, personas):
        self.nombres = [p['nombre'] for p in personas]
        self.medias = {k: np.stack([_l2(p[k]) for p in personas]).astype(np.float32) for k in ORDEN}
        self.matrices = {nombre_combo(c): np.concatenate([self.medias[k] for k in c], axis=1) / np.sqrt(len(c))
                         for c in COMBOS}

    def sims(self, desc):
        """desc: dict de descriptores de UN recorte -> {combo: vector de similitudes coseno con cada persona}"""
        return {nombre_combo(c): self.matrices[nombre_combo(c)] @ ExtractoresClasicos.vector_combo(desc, c)
                for c in COMBOS}


def cargar_base_clasica(ruta=None):
    ruta = ruta or BASE_CLASICA
    if not os.path.exists(ruta):
        return None
    with open(ruta, 'rb') as fh:
        datos = pickle.load(fh)
    return BaseClasica(datos['personas'])


def generar_base(personas_dir=None, salida=None, max_por_persona=None):
    personas = listar_personas(personas_dir, max_por_persona)
    if not personas:
        print(f'❌ No hay personas en {personas_dir or PERSONAS_BASE}. Captura con capturar_persona_mejorado.py')
        return None
    ex = ExtractoresClasicos()
    base, tms = [], {k: [] for k in ORDEN}
    print(f'\n🧬 Generando base de extractores clásicos ({len(personas)} personas)...')
    for nombre, ced, rutas in personas:
        acum = {k: [] for k in ORDEN}
        for r in rutas:
            im = cv2.imread(r)
            if im is None:
                continue
            d, t = ex.descriptores(im, medir=True)
            for k in ORDEN:
                acum[k].append(d[k])
                tms[k].append(t[k])
        if not acum['hog']:
            continue
        base.append({'nombre': nombre, 'cedula': ced, 'n_imagenes': len(acum['hog']),
                     **{k: _l2(np.mean(acum[k], axis=0)) for k in ORDEN}})
        print(f'   ✅ {nombre:<30} {len(acum["hog"]):>4} imágenes')
    ruta = salida or BASE_CLASICA
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, 'wb') as fh:
        pickle.dump({'version': 1, 'params': PARAMS, 'fecha': datetime.now().isoformat(), 'personas': base}, fh)
    print(f'\n📁 Base guardada en: {ruta}')
    print('   Tiempo medio por imagen (ms): ' + ', '.join(f'{k}={np.mean(v):.1f}' for k, v in tms.items()))
    return ruta


# ----------------------------------------------------------------------------
# Evaluación offline
# ----------------------------------------------------------------------------
def evaluar_S(S, qlab, n_personas):
    """S: (n_consultas, n_personas) similitudes. Protocolo cerrado (Top-k, MRR) y abierto
    leave-one-person-out (cada persona hace de 'desconocida' por turnos)."""
    S = np.nan_to_num(np.asarray(S, dtype=np.float64), nan=-1e9)
    qlab = np.asarray(qlab)
    orden = np.argsort(-S, axis=1)
    rank = np.argmax(orden == qlab[:, None], axis=1) + 1
    res = {'Top-1 (%)': 100 * float((rank == 1).mean()), 'Top-3 (%)': 100 * float((rank <= 3).mean()),
           'Top-5 (%)': 100 * float((rank <= 5).mean()), 'MRR': float((1.0 / rank).mean()), 'consultas': len(S)}
    if n_personas >= 3:
        sc, ok, kn = [], [], []
        for u in range(n_personas):
            m = np.ones(n_personas, bool)
            m[u] = False
            idx = np.where(m)[0]
            Sm = S[:, m]
            top = idx[Sm.argmax(1)]
            known = qlab != u
            sc.append(Sm.max(1)); ok.append((top == qlab) & known); kn.append(known)
        sc, ok, kn = np.concatenate(sc), np.concatenate(ok), np.concatenate(kn)
        try:
            from sklearn.metrics import roc_auc_score
            res['AUROC open-set'] = float(roc_auc_score(kn.astype(int), sc))
        except Exception:
            res['AUROC open-set'] = float('nan')
        mejor = (0.0, float(sc.min()), 0.0, 0.0, 0.0)
        for t in np.unique(np.quantile(sc, np.linspace(0, 1, 201))):
            acc = sc >= t
            tp = int((acc & ok).sum())
            p, r = tp / max(int(acc.sum()), 1), tp / max(int(kn.sum()), 1)
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            far = float((acc & ~kn).sum()) / max(int((~kn).sum()), 1)
            if f1 > mejor[0]:
                mejor = (f1, float(t), p, r, far)
        res.update({'F1 open-set': mejor[0], 'Umbral óptimo': mejor[1], 'Precision': mejor[2],
                    'Recall': mejor[3], 'FAR': mejor[4]})
    return res


def _fmt(v):
    if isinstance(v, float):
        return '-' if np.isnan(v) else (f'{v:.3f}' if abs(v) < 10 else f'{v:.1f}')
    return '-' if v is None else str(v)


def _tabla(filas):
    cols = list(filas[0].keys())
    ancho = {c: max(len(c), *(len(_fmt(f.get(c))) for f in filas)) for c in cols}
    print('   ' + ' | '.join(c.ljust(ancho[c]) for c in cols))
    print('   ' + '-+-'.join('-' * ancho[c] for c in cols))
    for f in filas:
        print('   ' + ' | '.join(_fmt(f.get(c)).ljust(ancho[c]) for c in cols))


def _md(filas):
    cols = list(filas[0].keys())
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    out += ['| ' + ' | '.join(_fmt(f.get(c)) for c in cols) + ' |' for f in filas]
    return '\n'.join(out)


def _guardar_csv(ruta, filas):
    cols = []
    for f in filas:
        for c in f:
            if c not in cols:
                cols.append(c)
    with open(ruta, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(filas)


def evaluar_offline(args):
    personas = listar_personas(max_por_persona=args.max_por_persona)
    personas = [p for p in personas if len(p[2]) >= 4]
    if len(personas) < 2:
        print('❌ Se necesitan al menos 2 personas con ≥4 imágenes en data/personas_base/')
        return None
    ex = ExtractoresClasicos()
    print(f'\n📊 Evaluando extractores clásicos con {len(personas)} personas reales (webcam)')
    X = {k: [] for k in ORDEN}
    tms = {k: [] for k in ORDEN}
    persona_idx, es_galeria, rutas_all = [], [], []
    for pi, (nombre, ced, rutas) in enumerate(personas):
        corte = max(1, len(rutas) // 2)            # primera mitad = galería, segunda = consultas (temporal)
        for j, r in enumerate(rutas):
            im = cv2.imread(r)
            if im is None:
                continue
            d, t = ex.descriptores(im, medir=True)
            for k in ORDEN:
                X[k].append(d[k]); tms[k].append(t[k])
            persona_idx.append(pi); es_galeria.append(j < corte); rutas_all.append(r)
    X = {k: np.stack(v) for k, v in X.items()}
    pidx, esg = np.array(persona_idx), np.array(es_galeria)
    P = len(personas)
    q = ~esg
    qlab = pidx[q]
    medias = {k: np.stack([_l2(X[k][(pidx == p) & esg].mean(0)) for p in range(P)]) for k in ORDEN}

    def S_combo(c):
        G = np.concatenate([medias[k] for k in c], 1) / np.sqrt(len(c))
        Qm = np.concatenate([X[k][q] for k in c], 1) / np.sqrt(len(c))
        return Qm @ G.T

    filas, Sdict = [], {}
    for c in COMBOS:
        S = S_combo(c)
        Sdict[nombre_combo(c)] = S
        r = evaluar_S(S, qlab, P)
        filas.append({'Extractores': nombre_combo(c), 'Tipo': TIPO_COMBO[len(c)],
                      'dim': int(sum(X[k].shape[1] for k in c)), 'ms/imagen': float(sum(np.mean(tms[k]) for k in c)), **r})
    print(f'\n📋 Extractores clásicos (galería: {int(esg.sum())} img, consultas: {int(q.sum())} img, {P} personas)')
    _tabla(filas)
    tablas = [('extractores_clasicos', 'Extractores clásicos: individual, parejas y triple', filas)]

    # ---- referencia con modelos profundos y fusiones ----
    if args.con_deep:
        try:
            from pruebas_uav_reid import construir_extractor, extraer_embeddings, resolver_modelos
            import torch
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            imgs = [cv2.resize(cv2.imread(r), (128, 256)) for r in rutas_all]
            fd, ff = [], []
            S_cl = Sdict['hog+lbp+hsv']
            for mod in resolver_modelos(args.modelos):
                try:
                    t = time.perf_counter()
                    E = extraer_embeddings(construir_extractor(mod, dev), imgs)
                    ms = (time.perf_counter() - t) * 1000 / len(imgs)
                    E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)
                    G = np.stack([_l2(E[(pidx == p) & esg].mean(0)) for p in range(P)])
                    Sd = E[q] @ G.T
                    fd.append({'Extractores': mod, 'Tipo': 'profundo', 'dim': int(E.shape[1]), 'ms/imagen': ms,
                               **evaluar_S(Sd, qlab, P)})
                    for a in (0.5, 0.7, 0.85):
                        ff.append({'Extractores': f'{mod} + hog+lbp+hsv (α={a})', 'Tipo': 'fusión', 'dim': int(E.shape[1]),
                                   'ms/imagen': ms + filas[-1]['ms/imagen'], **evaluar_S(a * Sd + (1 - a) * S_cl, qlab, P)})
                except Exception as e:
                    print(f'   ❌ {mod}: {type(e).__name__}: {e}')
            if fd:
                print('\n📋 Referencia: modelos profundos y fusión profundo+clásico')
                _tabla(fd + ff)
                tablas.append(('referencia_profundos', 'Modelos profundos y fusión (score-level)', fd + ff))
        except Exception as e:
            print(f'⚠️  --con-deep omitido: {type(e).__name__}: {e}')

    # ---- salidas ----
    salida = args.salida or os.path.join(DATA_DIR, 'pruebas_resultados', datetime.now().strftime('extractores_%Y%m%d_%H%M%S'))
    os.makedirs(salida, exist_ok=True)
    umbrales = {f['Extractores']: round(f['Umbral óptimo'], 4) for f in filas if 'Umbral óptimo' in f}
    with open(UMBRALES_CLASICOS, 'w', encoding='utf-8') as fh:
        json.dump(umbrales, fh, indent=2)
    with open(os.path.join(salida, 'umbrales_clasicos.json'), 'w', encoding='utf-8') as fh:
        json.dump(umbrales, fh, indent=2)
    md = ['# Evaluación de extractores clásicos (HOG / LBP / HSV)', '',
          f'- Personas: {P} | Galería: {int(esg.sum())} imágenes | Consultas: {int(q.sum())} imágenes',
          '- Protocolo: primera mitad temporal de cada persona = galería (media por persona), segunda mitad = consultas.',
          '- Open-set: leave-one-person-out (cada persona actúa una vez como desconocida).', '']
    for nombre, titulo, fl in tablas:
        _guardar_csv(os.path.join(salida, nombre + '.csv'), fl)
        md += [f'## {titulo}', '', _md(fl), '']
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        col = {'individual': 'tab:blue', 'pareja': 'tab:orange', 'triple': 'tab:green'}
        ax.bar([f['Extractores'] for f in filas], [f['Top-1 (%)'] for f in filas], color=[col[f['Tipo']] for f in filas])
        ax.set_ylabel('Top-1 (%)'); ax.set_title('Extractores clásicos: individual / pareja / triple')
        plt.xticks(rotation=30, ha='right'); fig.tight_layout(); fig.savefig(os.path.join(salida, 'fig_top1_combos.png'), dpi=150); plt.close(fig)
        mejor = max(filas, key=lambda f: f['Top-1 (%)'])['Extractores']
        S = Sdict[mejor]
        M = np.zeros((P, P))
        for a, b in zip(qlab, S.argmax(1)):
            M[a, b] += 1
        M = M / np.maximum(M.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        im = ax.imshow(M, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(P)); ax.set_yticks(range(P))
        ax.set_xticklabels([p[0][:10] for p in personas], rotation=60, fontsize=7)
        ax.set_yticklabels([p[0][:10] for p in personas], fontsize=7)
        ax.set_xlabel('Predicha'); ax.set_ylabel('Real'); ax.set_title(f'Matriz de confusión: {mejor}')
        fig.colorbar(im); fig.tight_layout(); fig.savefig(os.path.join(salida, 'fig_confusion_mejor.png'), dpi=150); plt.close(fig)
        md += ['![](fig_top1_combos.png)', '![](fig_confusion_mejor.png)', '']
    except Exception as e:
        print(f'   (figuras omitidas: {type(e).__name__})')
    with open(os.path.join(salida, 'resumen.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(md))
    print(f'\n📁 Resultados: {salida}\n📌 Umbrales guardados en {UMBRALES_CLASICOS} (los usa el monitor en tiempo real)')
    return filas


def main():
    ap = argparse.ArgumentParser(description='Extractores clásicos HOG/LBP/HSV: base y evaluación')
    ap.add_argument('--generar-base', action='store_true')
    ap.add_argument('--evaluar', action='store_true')
    ap.add_argument('--todo', action='store_true')
    ap.add_argument('--con-deep', action='store_true', help='añade los modelos profundos y la fusión como referencia')
    ap.add_argument('--modelos', default='sistema')
    ap.add_argument('--max-por-persona', type=int, default=None)
    ap.add_argument('--salida', default=None)
    a = ap.parse_args()
    if not (a.generar_base or a.evaluar or a.todo):
        ap.print_help()
        return
    if a.generar_base or a.todo:
        generar_base(max_por_persona=a.max_por_persona)
    if a.evaluar or a.todo:
        evaluar_offline(a)


if __name__ == '__main__':
    main()