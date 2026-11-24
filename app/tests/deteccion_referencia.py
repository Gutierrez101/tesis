#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DETECTOR DE REFERENCIA (verdad terreno AUTOMÁTICA para la detección)
====================================================================
En vuelo no hay cajas anotadas. Este módulo usa un detector MÁS FUERTE (por defecto YOLOv8x a 1280 px,
sobre el fotograma completo, sin reducir) como referencia y mide cuánto coincide con él el detector del
sistema (YOLOv8n a escala reducida): Precision, Recall, F1 y mAP@0.5 + recall según el tamaño de la persona.

IMPORTANTE (para la tesis): son métricas frente a un detector de referencia (pseudo-verdad terreno), NO frente a
anotación humana. Una persona que ambos detectores pierden no cuenta como fallo. Sirven para comparar condiciones
(altura, resolución) y como "recall relativo"; para el valor definitivo anota a mano con evaluar_deteccion_vuelo.py.
"""
import os
import sys
import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(TESTS_DIR)
for _d in (APP_DIR, TESTS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

BINS_TAMANO = [('<64 px', 0, 64), ('64-127 px', 64, 128), ('≥128 px', 128, 1e9)]


class ReferenciaDeteccion:
    def __init__(self, modelo, imgsz=1280, conf=0.5, nombre=''):
        self.modelo, self.imgsz, self.conf, self.nombre = modelo, imgsz, conf, nombre

    def cajas(self, frame):
        """Personas del fotograma COMPLETO -> [(x1,y1,x2,y2)]."""
        r = self.modelo(frame, verbose=False, imgsz=self.imgsz, conf=self.conf)[0]
        return [tuple(float(v) for v in b.xyxy[0]) for b in r.boxes if int(b.cls[0]) == 0]


def cargar_referencia(peso='yolov8x.pt', imgsz=1280, conf=0.5):
    """YOLO de referencia (la primera vez se descarga). Lanza excepción si no se puede cargar."""
    from ultralytics import YOLO
    m = YOLO(peso)
    try:
        import torch
        m.to('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception:
        pass
    return ReferenciaDeteccion(m, imgsz, conf, os.path.basename(str(peso)))


class AcumuladorDeteccion:
    """Compara, fotograma a fotograma, las detecciones del sistema contra las de referencia."""

    def __init__(self, conf_operativa=0.4, referencia=''):
        self.conf_op, self.referencia = conf_operativa, referencia
        self.gts, self.preds = [], []

    def agregar(self, det_cajas, det_confs, ref_cajas):
        self.gts.append(np.array(ref_cajas, dtype=float).reshape(-1, 4))
        confs = list(det_confs) if det_confs is not None and len(det_confs) == len(det_cajas) else [1.0] * len(det_cajas)
        self.preds.append([(*[float(v) for v in b], float(c)) for b, c in zip(det_cajas, confs)])

    @property
    def frames(self):
        return len(self.gts)

    def resumen(self):
        if not self.gts:
            return None
        import pruebas_uav_reid as U
        r = U.evaluar_detecciones(self.gts, self.preds, self.conf_op)
        hit = {n: [0, 0] for n, _, _ in BINS_TAMANO}                 # [aciertos, total] por tamaño de persona
        for g, p in zip(self.gts, self.preds):
            p_op = [d for d in p if d[4] >= self.conf_op]
            _, par, _ = U.emparejar(g, p_op)
            tomados = {j for j in par.values() if j >= 0}
            for j in range(len(g)):
                alto = g[j, 3] - g[j, 1]
                n = next((n for n, lo, hi in BINS_TAMANO if lo <= alto < hi), None)
                if n:
                    hit[n][1] += 1
                    hit[n][0] += int(j in tomados)
        return {'fuente': 'detector de referencia', 'referencia': self.referencia, 'imagenes': self.frames, 'gt': int(r['GT']),
                'tp': int(r['TP']), 'fp': int(r['FP']), 'fn': int(r['FN']), 'precision': float(r['Precision']),
                'recall': float(r['Recall']), 'f1': float(r['F1']), 'map50': float(r['mAP@0.5']),
                'conf_operativa': self.conf_op,
                'recall_por_tamano': {n: (a / t if t else None) for n, (a, t) in hit.items()},
                'personas_por_tamano': {n: t for n, (a, t) in hit.items()}}