#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DETECCIÓN CON CAJAS ANOTADAS SOBRE EL VIDEO DEL VUELO
=====================================================
Llena el dato "Detección mAP@0.5" del semáforo con datos REALES del dron.
En vuelo no hay cajas de referencia, así que se hace en 3 pasos sobre el video que graba el monitor
(config_pruebas.py -> GRABAR = True  =>  <sesion>/video_origen.mp4):

    1) --extraer N   saca N fotogramas repartidos por todo el vuelo  -> <sesion>/deteccion_gt/images
    2) --anotar      ventana para dibujar con el ratón una caja por persona -> <sesion>/deteccion_gt/labels (formato YOLO)
    3) --evaluar     corre el MISMO detector y preprocesado del sistema (YOLO, escala SCALE_PERCENT, conf MIN_CONFIDENCE)
                     y calcula Precision, Recall, F1 y mAP@0.5 -> <sesion>/deteccion_eval.json
                     (analisis_vuelo.py lo recoge solo en la próxima ejecución)

    python tests/evaluar_deteccion_vuelo.py --sesion data/pruebas_tiempo_real/<sesion> --todo --extraer 40

Atajos de la ventana de anotación:
    arrastrar con el ratón = dibuja una caja      z = deshace la última      c = borra todas las cajas
    n / espacio = guarda y pasa a la siguiente    b = guarda y vuelve a la anterior
    e = marca "sin personas" (guarda vacío) y avanza    s = salta sin guardar    q / Esc = guarda y sale
Las fotos que no anotes se ignoran. También puedes anotar con LabelImg/CVAT y copiar los .txt a deteccion_gt/labels.
"""
import os
import sys
import json
import argparse

import numpy as np
import cv2

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(TESTS_DIR)
for _d in (APP_DIR, TESTS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
R = os.path.basename(TESTS_DIR)


# ----------------------------------------------------------------------------
# Utilidades (probadas sin GUI)
# ----------------------------------------------------------------------------
def cajas_a_yolo(cajas, W, H):
    """[(x1,y1,x2,y2)] en píxeles -> líneas YOLO '0 cx cy w h' normalizadas."""
    out = []
    for x1, y1, x2, y2 in cajas:
        x1, x2 = sorted((max(0, x1), min(W, x2)))
        y1, y2 = sorted((max(0, y1), min(H, y2)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out.append(f'0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}')
    return out


def yolo_a_cajas(lineas, W, H):
    cajas = []
    for l in lineas:
        p = l.split()
        if len(p) >= 5 and int(float(p[0])) == 0:
            cx, cy, w, h = map(float, p[1:5])
            cajas.append((int((cx - w / 2) * W), int((cy - h / 2) * H), int((cx + w / 2) * W), int((cy + h / 2) * H)))
    return cajas


def extraer_frames(video, n, out_dir):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError(f'No se pudo abrir el video: {video}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:                                   # algunos contenedores no informan el total: se cuenta leyendo
        total = 0
        while cap.grab():
            total += 1
        cap.release()
        cap = cv2.VideoCapture(video)
    idx = sorted(set(int(i) for i in np.linspace(0, max(total - 1, 0), n)))
    os.makedirs(out_dir, exist_ok=True)
    guardadas = 0
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok:
            cv2.imwrite(os.path.join(out_dir, f'frame_{i:06d}.jpg'), fr)
            guardadas += 1
    cap.release()
    return guardadas


def _listar(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT)) if os.path.isdir(d) else []


# ----------------------------------------------------------------------------
# Anotador
# ----------------------------------------------------------------------------
def anotar(img_dir, lbl_dir):
    os.makedirs(lbl_dir, exist_ok=True)
    imgs = _listar(img_dir)
    if not imgs:
        print(f'❌ No hay imágenes en {img_dir}. Usa --extraer N primero.')
        return
    pendientes = [i for i, f in enumerate(imgs) if not os.path.exists(os.path.join(lbl_dir, os.path.splitext(f)[0] + '.txt'))]
    i = pendientes[0] if pendientes else 0
    win = 'Anotar personas (n=siguiente z=deshacer c=borrar e=sin personas q=salir)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    est = {'cajas': [], 'p0': None, 'p1': None, 'esc': 1.0}

    def mouse(ev, x, y, fl, par):
        e = est['esc']
        if ev == cv2.EVENT_LBUTTONDOWN:
            est['p0'], est['p1'] = (x, y), (x, y)
        elif ev == cv2.EVENT_MOUSEMOVE and est['p0']:
            est['p1'] = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP and est['p0']:
            (a, b), (c, d) = est['p0'], (x, y)
            if abs(c - a) > 5 and abs(d - b) > 5:
                est['cajas'].append((int(min(a, c) / e), int(min(b, d) / e), int(max(a, c) / e), int(max(b, d) / e)))
            est['p0'] = est['p1'] = None
    cv2.setMouseCallback(win, mouse)

    def cargar(k):
        im = cv2.imread(os.path.join(img_dir, imgs[k]))
        H, W = im.shape[:2]
        est['esc'] = min(1.0, 1280 / W, 720 / H)
        ruta = os.path.join(lbl_dir, os.path.splitext(imgs[k])[0] + '.txt')
        est['cajas'] = yolo_a_cajas(open(ruta).read().splitlines(), W, H) if os.path.exists(ruta) else []
        return im

    def guardar(k, im, vacio_ok=False):
        H, W = im.shape[:2]
        l = cajas_a_yolo(est['cajas'], W, H)
        if l or vacio_ok:
            with open(os.path.join(lbl_dir, os.path.splitext(imgs[k])[0] + '.txt'), 'w') as fh:
                fh.write('\n'.join(l))

    im = cargar(i)
    while True:
        e = est['esc']
        vis = cv2.resize(im, None, fx=e, fy=e) if e != 1.0 else im.copy()
        for x1, y1, x2, y2 in est['cajas']:
            cv2.rectangle(vis, (int(x1 * e), int(y1 * e)), (int(x2 * e), int(y2 * e)), (0, 255, 0), 2)
        if est['p0'] and est['p1']:
            cv2.rectangle(vis, est['p0'], est['p1'], (0, 255, 255), 2)
        hechas = sum(os.path.exists(os.path.join(lbl_dir, os.path.splitext(f)[0] + '.txt')) for f in imgs)
        cv2.putText(vis, f'{i + 1}/{len(imgs)}  cajas: {len(est["cajas"])}  anotadas: {hechas}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(win, vis)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord('n'), ord(' '), 13):
            guardar(i, im)
            i = min(i + 1, len(imgs) - 1)
            im = cargar(i)
        elif k == ord('b'):
            guardar(i, im)
            i = max(i - 1, 0)
            im = cargar(i)
        elif k == ord('z') and est['cajas']:
            est['cajas'].pop()
        elif k == ord('c'):
            est['cajas'] = []
        elif k == ord('e'):
            est['cajas'] = []
            guardar(i, im, vacio_ok=True)
            i = min(i + 1, len(imgs) - 1)
            im = cargar(i)
        elif k == ord('s'):
            i = min(i + 1, len(imgs) - 1)
            im = cargar(i)
        elif k in (ord('q'), 27):
            guardar(i, im)
            break
    cv2.destroyAllWindows()
    print(f'✅ Anotación guardada en {lbl_dir}')


# ----------------------------------------------------------------------------
# Evaluación (mismo detector y preprocesado que deteccion_reid_dron_contador.py)
# ----------------------------------------------------------------------------
def cargar_modelo(yolo=None):
    from ultralytics import YOLO
    peso = yolo or (os.path.join(APP_DIR, 'yolov8n.pt') if os.path.exists(os.path.join(APP_DIR, 'yolov8n.pt')) else 'yolov8n.pt')
    m = YOLO(peso)
    try:
        import torch
        m.to('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception:
        pass
    return m, os.path.basename(peso)


def detectar(modelo, frame, escala, conf=0.05):
    if escala != 100:
        small = cv2.resize(frame, (int(frame.shape[1] * escala / 100), int(frame.shape[0] * escala / 100)))
        f = 100.0 / escala
    else:
        small, f = frame, 1.0
    r = modelo(small, verbose=False, imgsz=640, conf=conf)[0]
    return [(*[float(v) * f for v in b.xyxy[0]], float(b.conf[0])) for b in r.boxes if int(b.cls[0]) == 0]


def evaluar(sesion, yolo=None):
    import pruebas_uav_reid as U
    gt_dir = os.path.join(sesion, 'deteccion_gt')
    img_dir, lbl_dir = os.path.join(gt_dir, 'images'), os.path.join(gt_dir, 'labels')
    archivos = [f for f in _listar(img_dir) if os.path.exists(os.path.join(lbl_dir, os.path.splitext(f)[0] + '.txt'))]
    if not archivos:
        print(f'❌ No hay fotogramas anotados en {lbl_dir}. Usa --anotar.')
        return None
    escala, conf_op = U._sist('SCALE_PERCENT', 60), U._sist('MIN_CONFIDENCE', 0.4)
    modelo, nombre = cargar_modelo(yolo)
    gts, preds = [], []
    for f in archivos:
        img = cv2.imread(os.path.join(img_dir, f))
        H, W = img.shape[:2]
        gts.append(U.leer_etiquetas_yolo(os.path.join(lbl_dir, os.path.splitext(f)[0] + '.txt'), W, H))
        preds.append(detectar(modelo, img, escala))
    r = U.evaluar_detecciones(gts, preds, conf_op)
    alt = None
    ruta_s = os.path.join(sesion, 'sistema.json')
    if os.path.exists(ruta_s):
        alt = json.load(open(ruta_s, encoding='utf-8')).get('altura_inicial')
    out = {'sesion': os.path.basename(os.path.normpath(sesion)), 'imagenes': len(archivos), 'gt': int(r['GT']),
           'tp': int(r['TP']), 'fp': int(r['FP']), 'fn': int(r['FN']), 'precision': float(r['Precision']),
           'recall': float(r['Recall']), 'f1': float(r['F1']), 'map50': float(r['mAP@0.5']),
           'detector': nombre, 'escala_pct': escala, 'conf_operativa': conf_op, 'altura_m': alt}
    with open(os.path.join(sesion, 'deteccion_eval.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print(f'\n📋 Detección sobre {len(archivos)} fotogramas anotados ({r["GT"]} personas) | {nombre}, escala {escala}%, conf ≥ {conf_op}')
    print(f'   Precision {out["precision"]:.3f} | Recall {out["recall"]:.3f} | F1 {out["f1"]:.3f} | mAP@0.5 {out["map50"]:.3f} '
          f'| TP {out["tp"]} FP {out["fp"]} FN {out["fn"]}')
    print(f'📁 Guardado: {os.path.join(sesion, "deteccion_eval.json")}\n   Vuelve a analizar:  python {R}/analisis_vuelo.py "{sesion}"')
    return out


def main():
    ap = argparse.ArgumentParser(description='Detección con cajas anotadas sobre el video del vuelo')
    ap.add_argument('--sesion', required=True, help='carpeta data/pruebas_tiempo_real/<sesion>')
    ap.add_argument('--video', default=None, help='por defecto <sesion>/video_origen.mp4')
    ap.add_argument('--extraer', type=int, default=0, help='número de fotogramas a extraer')
    ap.add_argument('--anotar', action='store_true')
    ap.add_argument('--evaluar', action='store_true')
    ap.add_argument('--todo', action='store_true', help='extraer (si hace falta) + anotar + evaluar')
    ap.add_argument('--yolo', default=None)
    a = ap.parse_args()
    ses = a.sesion
    img_dir, lbl_dir = (os.path.join(ses, 'deteccion_gt', x) for x in ('images', 'labels'))
    if a.extraer or (a.todo and not _listar(img_dir)):
        video = a.video or os.path.join(ses, 'video_origen.mp4')
        if not os.path.exists(video):
            raise SystemExit(f'❌ No existe {video}. Pon GRABAR = True en config_pruebas.py antes del vuelo.')
        n = extraer_frames(video, a.extraer or 40, img_dir)
        print(f'🖼️  {n} fotogramas extraídos en {img_dir}')
    if a.anotar or a.todo:
        anotar(img_dir, lbl_dir)
    if a.evaluar or a.todo:
        evaluar(ses, a.yolo)
    if not (a.extraer or a.anotar or a.evaluar or a.todo):
        ap.print_help()


if __name__ == '__main__':
    main()