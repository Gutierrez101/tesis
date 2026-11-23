#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VISUALIZACIONES HOG / HSV / LBP DE LAS CAPTURAS
===============================================
Genera, a partir de las imágenes ORIGINALES que guarda capturar_persona_mejorado.py (data/personas_base/<persona>/),
las mismas visualizaciones que produce el sistema en vivo (usa VisualizationProcessors de visualization_handler.py,
con sus mismos parámetros: HOG 9 orientaciones / celdas 8x8 / bloques 2x2, LBP P=24 R=3 'uniform', HSV H-S-V + histograma Hue).

Uso (desde app/):
    python tests/visualizar_capturas.py                                  # todas las personas
    python tests/visualizar_capturas.py --persona juan_perez_1712345678 --n 6
    python tests/visualizar_capturas.py --persona juan_perez_1712345678 --todas-las-imagenes
    python tests/visualizar_capturas.py --imagen data/personas_base/juan_perez_1712345678/captura_001.jpg

También se ejecuta solo al terminar capturar_persona_mejorado.py (VISUALIZAR_CAPTURAS=0 lo desactiva).

Salida en data/visualizaciones_capturas/<persona>/:
    original/ hsv/ lbp/ hog/            una imagen por captura seleccionada
    <persona>_montaje.png               una fila por captura: Original | HSV | LBP | HOG
    ../resumen_todas_las_personas.png   una fila por persona (solo al procesar a todas)
"""
import os
import sys
import argparse

import numpy as np
import cv2

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(TESTS_DIR)
for _d in (APP_DIR, TESTS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')


def _dirs():
    try:
        import config
        return config.DATA_DIR, config.PERSONAS_BASE
    except Exception:
        d = os.path.join(APP_DIR, 'data')
        return d, os.path.join(d, 'personas_base')


DATA_DIR, PERSONAS_BASE = _dirs()
SALIDA_BASE = os.path.join(DATA_DIR, 'visualizaciones_capturas')


def _procesadores():
    from visualization_handler import VisualizationProcessors, VisualizationConfig
    cfg = {k: getattr(VisualizationConfig, k, d) for k, d in (
        ('LBP_POINTS', 24), ('LBP_RADIUS', 3), ('LBP_METHOD', 'uniform'), ('HOG_ORIENTATIONS', 9),
        ('HOG_PIXELS_PER_CELL', (8, 8)), ('HOG_CELLS_PER_BLOCK', (2, 2)))}
    return VisualizationProcessors, cfg


def visualizar_imagen(img):
    """img BGR (recorte 128x256 de la captura) -> dict con 'original', 'hsv', 'lbp', 'hog' (imágenes BGR)."""
    P, c = _procesadores()
    img = cv2.resize(img, (128, 256)) if img.shape[:2] != (256, 128) else img
    return {'original': img, 'hsv': P.process_hsv(img),
            'lbp': P.process_lbp(img, n_points=c['LBP_POINTS'], radius=c['LBP_RADIUS'], method=c['LBP_METHOD']),
            'hog': P.process_hog(img, orientations=c['HOG_ORIENTATIONS'], pixels_per_cell=c['HOG_PIXELS_PER_CELL'],
                                 cells_per_block=c['HOG_CELLS_PER_BLOCK'])}


def _alto(im, h=512):
    return cv2.resize(im, (max(1, int(im.shape[1] * h / im.shape[0])), h), interpolation=cv2.INTER_CUBIC)


def _fila(v, titulo=None):
    orig = _alto(v['original'], 512)
    if titulo:
        cv2.putText(orig, titulo[:22], (6, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return np.hstack([orig, _alto(v['hsv']), _alto(v['lbp']), _alto(v['hog'])])


def _encabezado(ancho_orig, anchos):
    h = 42
    cab = np.zeros((h, ancho_orig + sum(anchos), 3), np.uint8)
    x = 0
    for txt, w in zip(('Original', 'HSV  (H, S, V + histograma Hue)', 'LBP  (P=24, R=3, uniform)', 'HOG  (9 orient., 8x8, 2x2)'),
                      (ancho_orig, *anchos)):
        cv2.putText(cab, txt, (x + 8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        x += w
    return cab


def _montaje(filas):
    ancho_orig = _alto(np.zeros((256, 128, 3), np.uint8)).shape[1]
    anchos = [filas[0].shape[1] - ancho_orig]
    # ancho de cada panel (hsv/lbp/hog) = mismo, se reparte en 3
    pw = anchos[0] // 3
    cab = _encabezado(ancho_orig, [pw, pw, pw])
    return np.vstack([cab] + filas)


def _lista(carpeta):
    return sorted(f for f in os.listdir(carpeta) if f.lower().endswith(IMG_EXT))


def _seleccion(archivos, n):
    if n <= 0 or n >= len(archivos):
        return archivos
    idx = sorted(set(int(i) for i in np.linspace(0, len(archivos) - 1, n)))
    return [archivos[i] for i in idx]


def visualizar_persona(carpeta, salida_base=None, n=4, todas=False):
    """Genera original/hsv/lbp/hog y un montaje para UNA persona. Devuelve la ruta del montaje (o None)."""
    persona = os.path.basename(os.path.normpath(carpeta))
    archivos = _lista(carpeta)
    if not archivos:
        print(f'   ⚠️  {persona}: sin imágenes')
        return None
    elegidas = archivos if todas else _seleccion(archivos, n)
    out = os.path.join(salida_base or SALIDA_BASE, persona)
    for sub in ('original', 'hsv', 'lbp', 'hog'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    filas = []
    for f in elegidas:
        im = cv2.imread(os.path.join(carpeta, f))
        if im is None:
            continue
        v = visualizar_imagen(im)
        base = os.path.splitext(f)[0]
        for k in ('original', 'hsv', 'lbp', 'hog'):
            cv2.imwrite(os.path.join(out, k, f'{base}_{k}.jpg'), v[k], [cv2.IMWRITE_JPEG_QUALITY, 92])
        filas.append(_fila(v, base))
    if not filas:
        return None
    m = _montaje(filas)
    if m.shape[1] > 1800:
        m = cv2.resize(m, None, fx=1800 / m.shape[1], fy=1800 / m.shape[1], interpolation=cv2.INTER_AREA)
    ruta = os.path.join(out, f'{persona}_montaje.png')
    cv2.imwrite(ruta, m)
    print(f'   🎨 {persona}: {len(filas)} captura(s) -> {out}\n      montaje: {ruta}')
    return ruta


def visualizar_todas(personas_dir=None, salida_base=None, n=4, todas=False):
    pd = personas_dir or PERSONAS_BASE
    personas = [d for d in sorted(os.listdir(pd)) if os.path.isdir(os.path.join(pd, d)) and _lista(os.path.join(pd, d))] \
        if os.path.isdir(pd) else []
    if not personas:
        print(f'❌ No hay personas con imágenes en {pd}')
        return
    print(f'\n🎨 Visualizaciones HOG / HSV / LBP de {len(personas)} persona(s)')
    filas = []
    for p in personas:
        visualizar_persona(os.path.join(pd, p), salida_base, n, todas)
        arch = _lista(os.path.join(pd, p))
        im = cv2.imread(os.path.join(pd, p, arch[len(arch) // 2]))
        if im is not None:
            filas.append(_fila(visualizar_imagen(im), p))
    if len(filas) > 1:
        m = _montaje(filas)
        if m.shape[1] > 1800:
            m = cv2.resize(m, None, fx=1800 / m.shape[1], fy=1800 / m.shape[1], interpolation=cv2.INTER_AREA)
        ruta = os.path.join(salida_base or SALIDA_BASE, 'resumen_todas_las_personas.png')
        cv2.imwrite(ruta, m)
        print(f'\n🖼️  Resumen de todas las personas: {ruta}')


def main():
    ap = argparse.ArgumentParser(description='Visualizaciones HOG/HSV/LBP de las capturas')
    ap.add_argument('--persona', default=None, help='nombre de la carpeta en personas_base (por defecto: todas)')
    ap.add_argument('--imagen', default=None, help='una sola imagen')
    ap.add_argument('--n', type=int, default=4, help='capturas por persona (repartidas); 0 = todas')
    ap.add_argument('--todas-las-imagenes', action='store_true')
    ap.add_argument('--salida', default=None)
    a = ap.parse_args()
    if a.imagen:
        im = cv2.imread(a.imagen)
        if im is None:
            raise SystemExit(f'No se pudo leer {a.imagen}')
        out = a.salida or SALIDA_BASE
        os.makedirs(out, exist_ok=True)
        m = _montaje([_fila(visualizar_imagen(im), os.path.splitext(os.path.basename(a.imagen))[0])])
        ruta = os.path.join(out, os.path.splitext(os.path.basename(a.imagen))[0] + '_viz.png')
        cv2.imwrite(ruta, m)
        print(f'🎨 {ruta}')
    elif a.persona:
        visualizar_persona(os.path.join(PERSONAS_BASE, a.persona), a.salida, a.n, a.todas_las_imagenes)
    else:
        visualizar_todas(None, a.salida, a.n, a.todas_las_imagenes)


if __name__ == '__main__':
    main()