#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REPETICIÓN SIN VENTANA DE UN VUELO GRABADO (comparación de arquitecturas con el MISMO proceso de Re-ID)
=====================================================================================================
Lee el video que graba el monitor (<sesion>/video_origen.mp4) y lo pasa por el MISMO bucle que
deteccion_reid_dron_contador.py (YOLO -> tracking por IoU -> Re-ID cada REID_UPDATE_INTERVAL frames, con sus propias
funciones y constantes), pero:
  * sin ventana ni dron ni nginx,
  * con el monitor en modo "completo": sobre cada recorte evalúa los 5 modelos (OSNet, RepViT M0.9, EfficientFormerV2 S0,
    FastViT T8, SwiftFormer XS) + las 7 combinaciones HOG/LBP/HSV + fusiones,
  * midiendo la detección frente a un detector de referencia (verdad terreno automática).
Solo cambia el extractor de características: detección, tracking y recortes son idénticos para todos los modelos,
así que la comparación de estabilidad temporal, acuerdo, etc. entre arquitecturas es justa.

Se ejecuta SOLO al terminar un vuelo en modo 'eficiencia' (config_pruebas.py -> POSTPROCESO_COMPARACION = True) o a mano:
    python tests/reproducir_vuelo.py --video data/pruebas_tiempo_real/<sesion>/video_origen.mp4
    python tests/reproducir_vuelo.py                                  # usa la última sesión grabada
Opciones: --etiquetas tiempos.csv (CSV desde_s,hasta_s,persona) | --sin-ref | --ref-cada 10 | --max-frames N
"""
import os
import sys
import time
import argparse

import cv2

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(TESTS_DIR)
for _d in (APP_DIR, TESTS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)
R = os.path.basename(TESTS_DIR)


def _ultimo_video():
    try:
        from config import DATA_DIR
    except Exception:
        DATA_DIR = os.path.join(APP_DIR, 'data')
    base = os.path.join(DATA_DIR, 'pruebas_tiempo_real')
    if not os.path.isdir(base):
        return None
    for d in sorted(os.listdir(base), reverse=True):
        v = os.path.join(base, d, 'video_origen.mp4')
        if os.path.exists(v):
            return v
    return None


def reproducir(video, sesion='comparacion', max_frames=None, ref=True, ref_cada=10):
    """Repite `video` sin ventana con el monitor en modo completo. Devuelve la carpeta de la sesión."""
    import pickle
    import deteccion_reid_dron_contador as D          # funciones y constantes REALES del sistema (no ejecuta main)
    from feature_extractors import build_extractor, get_base_path, get_model_name
    from ultralytics import YOLO
    from monitor_tiempo_real import MonitorPruebas
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        device = 'cpu'

    primario = get_model_name()
    with open(get_base_path(), 'rb') as fh:
        base = pickle.load(fh)
    model = YOLO('yolov8n.pt')
    model.conf = D.MIN_CONFIDENCE
    extractor = build_extractor(device=device)
    umbral = D.get_threshold()
    monitor = MonitorPruebas(modo='completo', primary_name=primario, primary_base=base, device=device, sesion=sesion)
    monitor.grabar = False                      # la repetición no vuelve a grabar el video
    if ref and monitor.ref is None:
        monitor.activar_referencia(cada=ref_cada)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError(f'No se pudo abrir {video}')
    print(f'\n▶️  Repitiendo {video} sin ventana (modelo principal: {primario}) ...')
    D.PersonTrack._next_id = 1
    tracks, frame_count, t0 = [], 0, time.time()
    while True:
        monitor.iter_inicio()
        ret, frame = cap.read()
        monitor.lectura(ret and frame is not None)
        if not ret or frame is None:
            break
        if frame.size == 0 or frame.shape[0] < 10 or frame.shape[1] < 10:
            continue
        frame_count += 1
        if max_frames and frame_count > max_frames:
            break
        original_frame = frame.copy()
        monitor.on_frame(original_frame)
        if D.SCALE_PERCENT != 100:
            frame_small = cv2.resize(frame, (int(frame.shape[1] * D.SCALE_PERCENT / 100), int(frame.shape[0] * D.SCALE_PERCENT / 100)))
            scale_factor = 100 / D.SCALE_PERCENT
        else:
            frame_small, scale_factor = frame, 1.0
        if frame_count % D.FRAME_SKIP == 0:
            _t_det = time.perf_counter()
            detections, detection_crops, detection_confs = [], [], []
            try:
                results = model(frame_small, verbose=False, imgsz=640)
                for r in results:
                    for box in r.boxes:
                        if int(box.cls[0]) == 0:
                            conf = float(box.conf[0])
                            if conf >= D.MIN_CONFIDENCE:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                if scale_factor != 1.0:
                                    x1, y1, x2, y2 = int(x1 * scale_factor), int(y1 * scale_factor), int(x2 * scale_factor), int(y2 * scale_factor)
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(original_frame.shape[1], x2), min(original_frame.shape[0], y2)
                                if x2 > x1 and y2 > y1:
                                    detections.append((x1, y1, x2, y2))
                                    detection_confs.append(conf)
                                    crop = original_frame[y1:y2, x1:x2]
                                    if crop.size > 0:
                                        detection_crops.append(crop)
            except Exception:
                detections, detection_crops, detection_confs = [], [], []
                monitor.on_error('deteccion')
            monitor.etapa('deteccion', (time.perf_counter() - _t_det) * 1000)
            monitor.on_detecciones(detections, original_frame.shape, detection_confs)
            _t_trk = time.perf_counter()
            matched, unmatched_dets, unmatched_tracks = D.match_detections_to_tracks(detections, tracks)
            for det_idx, track_idx in matched:
                tracks[track_idx].update(detections[det_idx], frame_count)
            for det_idx in unmatched_dets:
                tracks.append(D.PersonTrack(detections[det_idx], frame_count))
            for track_idx in unmatched_tracks:
                tracks[track_idx].increment_age()
            monitor.etapa('tracking', (time.perf_counter() - _t_trk) * 1000)
            for i, track in enumerate(tracks):
                if track.is_confirmed() and track.age == 0 and (frame_count - track.last_reid_frame) >= D.REID_UPDATE_INTERVAL:
                    crop_idx = next((d for d, t in matched if t == i), None)
                    if crop_idx is not None and crop_idx < len(detection_crops):
                        crop = detection_crops[crop_idx]
                        if crop is not None and crop.size > 0 and len(crop.shape) == 3:
                            try:
                                crop_resized = cv2.resize(crop, (128, 256))
                                t_reid = time.perf_counter()
                                emb = extractor(crop_resized)[0]
                                identity, confidence = D.identificar(emb, base, threshold=umbral)
                                ms_reid = (time.perf_counter() - t_reid) * 1000
                                track.update_identity(identity, confidence, frame_count)
                                monitor.on_reid(track, crop_resized, emb, identity, confidence, ms_reid, frame_count)
                            except Exception:
                                monitor.on_error('reid')
            tracks = [t for t in tracks if t.is_alive()]
            monitor.on_tracks(tracks, frame_count)
        monitor.fin_iteracion()
        if frame_count % 200 == 0:
            print(f'   ... {frame_count} frames ({frame_count / max(time.time() - t0, 1e-6):.1f} fps)')
    cap.release()
    print(f'✅ Repetición terminada: {frame_count} frames en {time.time() - t0:.0f}s')
    monitor.finalizar()
    return monitor.dir


def main():
    ap = argparse.ArgumentParser(description='Repite un vuelo grabado sin ventana con los 5 modelos')
    ap.add_argument('--video', default=None, help='por defecto: video_origen.mp4 de la última sesión')
    ap.add_argument('--sesion', default='comparacion')
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--etiquetas', default=None, help='CSV desde_s,hasta_s,persona (etiquetado automático por tiempos)')
    ap.add_argument('--fps-video', type=float, default=None)
    ap.add_argument('--sin-ref', action='store_true', help='no medir la detección contra el detector de referencia')
    ap.add_argument('--ref-cada', type=int, default=10, help='1 de cada N frames se compara con el detector de referencia')
    a = ap.parse_args()
    video = a.video or _ultimo_video()
    if not video or not os.path.exists(video):
        raise SystemExit('❌ No hay video. Graba un vuelo con GRABAR = True en config_pruebas.py o pasa --video.')
    if a.etiquetas:
        os.environ['PRUEBAS_ETIQUETAS_CSV'] = os.path.abspath(a.etiquetas)
        os.environ['PRUEBAS_ETIQUETAS_BASE'] = 'frames'
    if a.fps_video:
        os.environ['PRUEBAS_FPS_STREAM'] = str(a.fps_video)
    reproducir(video, a.sesion, a.max_frames, not a.sin_ref, a.ref_cada)


if __name__ == '__main__':
    main()