#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MONITOR DE PRUEBAS EN TIEMPO REAL  (se activa desde deteccion_reid_dron_contador.py)
====================================================================================
Se enciende con  MODO  en pruebas/config_pruebas.py  (o la variable de entorno PRUEBAS_MODO):
    off         (por defecto) el sistema funciona EXACTAMENTE igual que siempre
    eficiencia  solo mide tiempos / memoria / pérdida de frames / estabilidad (ligero, no altera el FPS)
    completo    además compara EN VIVO sobre el mismo recorte del dron:
                  - los modelos profundos con base .pkl (OSNet, RepViT M0.9, EfficientFormerV2 S0, FastViT T8, SwiftFormer XS)
                  - los extractores clásicos HOG / LBP / HSV: individual, en parejas y los 3 juntos (7 combinaciones)
                  - fusiones modelo profundo + HOG+LBP+HSV
                con ETIQUETADO EN VIVO por el operador para obtener precisión real.

Teclas / ratón (ventana "ReID + Contador"):
    clic izquierdo sobre una persona ..... selecciona su track
    1..9, 0 ............................. asigna la persona enrolada Nº1..9, 10 al track seleccionado
                                           (si hay un solo track confirmado en pantalla no hace falta clic)
    u ................................... marca el track como DESCONOCIDO (persona NO enrolada → open-set)
    x ................................... borra la etiqueta del track seleccionado
    l ................................... modo sesión: los dígitos fijan la persona de TODA la sesión mientras
                                           haya un único track confirmado (útil cuando camina una persona a la vez)
    + / - ............................... altura del dron ±5 m (se registra en cada evento)
    a ................................... cambia el ángulo de cámara
    o ................................... marca/desmarca oclusión
    g ................................... inicia/detiene la grabación del video original del vuelo
    t ................................... muestra/oculta el panel de pruebas
    s ................................... guarda una copia parcial ahora (además de las teclas existentes q/s/r/m/v)

Configuración: edita pruebas/config_pruebas.py (MODO, SESION, ALTURA_M, ANGULO, GRABAR...) y ejecuta
               python deteccion_reid_dron_contador.py   . Las variables PRUEBAS_* (si existen) tienen prioridad.
Salida: data/pruebas_tiempo_real/<fecha>_<sesion>/  (eventos_reid.csv, etiquetas.csv, sistema.json, telemetria.csv,
        video_origen.mp4, crops/, analisis/ con resumen.md, CSV y figuras)
"""
import os
import sys
import csv
import json
import time
import threading
import subprocess
import platform
from collections import defaultdict
from datetime import datetime

import numpy as np
import cv2

PRUEBAS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(PRUEBAS_DIR)
# Imports independientes del nombre de la carpeta (pruebas / tests / otro)
for _d in (APP_DIR, PRUEBAS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from extractores_clasicos import (ExtractoresClasicos, cargar_base_clasica, COMBOS, nombre_combo,
                                  UMBRALES_CLASICOS)
from analisis_vuelo import analizar_sesiones, COLS_EVENTOS, DESCONOCIDO

R = os.path.basename(PRUEBAS_DIR)      # nombre real de la carpeta, para los mensajes

ANGULOS = ['frontal-oblicuo', 'lateral-oblicuo', 'posterior-oblicuo', 'cenital_parcial', 'cenital_extremo']
TRIPLE = 'hog+lbp+hsv'


def _cfg(env, attr, defecto):
    """Prioridad: variable de entorno > pruebas/config_pruebas.py > valor por defecto."""
    v = os.environ.get(env)
    if v not in (None, ''):
        return v
    try:
        import config_pruebas as c
        return getattr(c, attr, defecto)
    except Exception:
        return defecto


def _bool(v):
    return str(v).strip().lower() in ('1', 'true', 'si', 'sí', 'yes', 'on')


def _np(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _l2(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _pcts(l):
    if not l:
        return None
    a = np.asarray(l, dtype=float)
    return {'mean': float(a.mean()), 'p50': float(np.percentile(a, 50)), 'p95': float(np.percentile(a, 95)),
            'p99': float(np.percentile(a, 99)), 'max': float(a.max())}


def _mide(fn):
    """Acumula el tiempo que gasta el monitor para poder restárselo al FPS del sistema."""
    def w(self, *a, **k):
        t = time.perf_counter()
        try:
            return fn(self, *a, **k)
        finally:
            self._ov += (time.perf_counter() - t) * 1000
    return w


class _Modelo:
    def __init__(self, nombre, ext, base, umbral):
        self.nombre, self.ext, self.umbral = nombre, ext, umbral
        self.names = [p['nombre'] for p in base]
        M = np.array([np.asarray(p['embedding'], dtype=np.float32).reshape(-1) for p in base])
        self.M = M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-9, None)


class MonitorPruebas:
    def __init__(self, modo, primary_name, primary_base, device='cpu', window='ReID + Contador', sesion=None):
        from config import DATA_DIR
        self.modo, self.completo = modo, modo == 'completo'
        self.primario, self.device, self.window = primary_name, device, window
        sesion = sesion or _cfg('PRUEBAS_SESION', 'SESION', 'sesion')
        self.t0 = time.time()
        self.dir = os.path.join(DATA_DIR, 'pruebas_tiempo_real', f'{datetime.now():%Y%m%d_%H%M%S}_{sesion}')
        os.makedirs(self.dir, exist_ok=True)
        self.altura = self.altura_inicial = float(_cfg('PRUEBAS_ALTURA_M', 'ALTURA_M', 10))
        self.angulo = str(_cfg('PRUEBAS_ANGULO', 'ANGULO', ANGULOS[0]))
        self.ocl = 0
        self.fps_nominal = float(_cfg('PRUEBAS_FPS_STREAM', 'FPS_STREAM', 30))
        self.crop_cada = int(_cfg('PRUEBAS_CROP_CADA', 'CROP_CADA', 3))
        self.grabar = _bool(_cfg('PRUEBAS_GRABAR', 'GRABAR', False))
        self.mostrar_panel, self.modo_sesion, self.sesion_gt = True, False, None
        self.writer, self.n_crops = None, 0
        # --- contadores de tiempo / sistema ---
        self.lat = defaultdict(list)
        self.frames = self.frames_err = self.intentos = self.fallos = self.reconexiones = self.tardios = 0
        self.errores = defaultdict(int)
        self._ov, self._st, self._t_iter, self._t_post, self._err_frame = 0.0, defaultdict(float), time.perf_counter(), time.perf_counter(), False
        self.n_personas, self.altos, self.sin_personas = [], [], 0
        self.telemetria, self._parar = [], False
        # --- etiquetado y eventos ---
        self.labels, self.eventos, self.ev_por_track, self.log_etiquetas = {}, [], defaultdict(list), []
        self.sel_track, self._tracks_vis, self._n_confirm = None, [], 0
        self.cont = defaultdict(lambda: {'n': 0, 'top1': 0, 'top5': 0})
        self.tipos, self.umbrales = {}, {}
        # --- modelos ---
        from feature_extractors import ALL_MODELS, build_extractor, get_base_path, get_threshold
        self.prim = _Modelo(primary_name, None, primary_base, get_threshold(primary_name))
        self.tipos[primary_name] = 'deep'
        self.tipos[primary_name + '+voto'] = 'voto'
        self.umbrales[primary_name] = self.prim.umbral
        self.enrolados = sorted(self.prim.names)
        self.deeps, self.clasica, self.fusiones = [], None, []
        print(f'\n🔬 MONITOR DE PRUEBAS activado  (modo: {modo})  ->  {self.dir}')
        if self.completo:
            import pickle
            for m in ALL_MODELS:
                if m == primary_name:
                    continue
                p = get_base_path(m)
                if not os.path.exists(p):
                    print(f'   ⚪ {m}: sin base ({os.path.basename(p)}), se omite')
                    continue
                try:
                    with open(p, 'rb') as fh:
                        base = pickle.load(fh)
                    ext = build_extractor(device=device, model_name=m)
                    mod = _Modelo(m, ext, base, get_threshold(m))
                    dim = _np(ext(np.zeros((256, 128, 3), np.uint8))[0]).shape[0]
                    if dim != mod.M.shape[1]:
                        raise ValueError(f'dimensión {dim} != base {mod.M.shape[1]} (¿base regenerada tras cambiar de modelo?)')
                    self.deeps.append(mod)
                    self.tipos[m], self.umbrales[m] = 'deep', mod.umbral
                    print(f'   ✅ {m}: {len(mod.names)} persona(s), dim={dim}, umbral={mod.umbral}')
                except Exception as e:
                    print(f'   ❌ {m}: {type(e).__name__}: {e} (se omite)')
            self.clasica = cargar_base_clasica()
            if self.clasica is None:
                print(f'   ⚠️  Sin base clásica: ejecuta  python {R}/extractores_clasicos.py --generar-base')
            else:
                self.ex_clasico = ExtractoresClasicos()
                um = {}
                if os.path.exists(UMBRALES_CLASICOS):
                    with open(UMBRALES_CLASICOS, encoding='utf-8') as fh:
                        um = json.load(fh)
                for c in COMBOS:
                    n = nombre_combo(c)
                    self.tipos[n], self.umbrales[n] = 'clasico', float(um.get(n, 0.90))
                print(f'   ✅ extractores clásicos: {len(self.clasica.nombres)} persona(s), 7 combinaciones')
                comunes = sorted(set(self.prim.names) & set(self.clasica.nombres))
                alfas = [float(a) for a in str(_cfg('PRUEBAS_ALFAS', 'ALFAS', '0.5,0.7')).split(',') if a.strip()]
                if comunes:
                    self.f_idx_p = [self.prim.names.index(n) for n in comunes]
                    self.f_idx_c = [self.clasica.nombres.index(n) for n in comunes]
                    self.f_names = comunes
                    for a in alfas:
                        n = f'{primary_name}+{TRIPLE}@{a}'
                        self.fusiones.append((n, a))
                        self.tipos[n] = 'fusion'
                        self.umbrales[n] = a * self.prim.umbral + (1 - a) * self.umbrales[TRIPLE]
                    print(f'   ✅ fusiones: {", ".join(n for n, _ in self.fusiones)}')
        print(f'   👥 Personas enroladas (dígitos): ' + ', '.join(f'{(i + 1) % 10}={n}' for i, n in enumerate(self.enrolados[:10])))
        if self.grabar:
            print('   🔴 Grabación del video origen activada')
        threading.Thread(target=self._telemetria, daemon=True).start()

    # ------------------------------------------------------------------ ciclo de frame
    def iter_inicio(self):
        self._t_iter = time.perf_counter()
        self._ov, self._st, self._err_frame = 0.0, defaultdict(float), False

    def lectura(self, ok):
        now = time.perf_counter()
        self._st['lectura'] = (now - self._t_iter) * 1000
        self._t_post = now
        self.intentos += 1
        if not ok:
            self.fallos += 1

    def on_reconexion(self):
        self.reconexiones += 1

    @_mide
    def on_frame(self, frame):
        self.frames += 1
        if self.grabar:
            if self.writer is None:
                h, w = frame.shape[:2]
                self.writer = cv2.VideoWriter(os.path.join(self.dir, 'video_origen.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                              float(_cfg('PRUEBAS_VIDEO_FPS', 'VIDEO_FPS', 25)), (w, h))
            self.writer.write(frame)

    def etapa(self, nombre, ms):
        self._st[nombre] += ms

    def on_error(self, que):
        self.errores[que] += 1
        self._err_frame = True

    def on_detecciones(self, detecciones, shape):
        self.n_personas.append(len(detecciones))
        if not detecciones:
            self.sin_personas += 1
        for d in detecciones[:20]:
            self.altos.append(d[3] - d[1])

    def on_tracks(self, tracks, frame_count):
        vis = [t for t in tracks if t.is_confirmed()]
        self._tracks_vis = [(t.track_id, t.bbox) for t in vis]
        self._n_confirm = len(vis)

    def fin_iteracion(self):
        now = time.perf_counter()
        wall = (now - self._t_iter) * 1000
        proc = (now - self._t_post) * 1000 - self._ov
        self.lat['iter_wall'].append(wall)
        self.lat['wall_sin_monitor'].append(max(wall - self._ov, 0.0))
        self.lat['proc_sin_monitor'].append(max(proc, 0.0))
        self.lat['overhead_monitor'].append(self._ov)
        for k, v in self._st.items():
            self.lat[k].append(v)
        if proc > 1000.0 / self.fps_nominal:
            self.tardios += 1
        if self._err_frame:
            self.frames_err += 1

    # ------------------------------------------------------------------ Re-ID por evento
    def _res(self, tipo, s, names, ms):
        o = np.argsort(-s)
        return {'tipo': tipo, 'orden': [names[i] for i in o], 'score': float(s[o[0]]), 'ms': float(ms)}

    @_mide
    def on_reid(self, track, crop_resized, emb, identity, confidence, ms_reid, frame):
        self._st['reid_primario'] += ms_reid
        if not self.completo:
            return
        tid = track.track_id
        gt = self.labels.get(tid)
        if gt is None and self.modo_sesion and self.sesion_gt and self._n_confirm == 1:
            gt = self.sesion_gt
        ev = {'id': len(self.eventos), 't': time.time() - self.t0, 'frame': frame, 'track': tid, 'gt': gt,
              'altura': self.altura, 'angulo': self.angulo, 'ocl': self.ocl,
              'crop_h': int(track.bbox[3] - track.bbox[1]), 'crop_w': int(track.bbox[2] - track.bbox[0]), 'v': {}}
        # principal (embedding que ya calculó el sistema: no se repite la inferencia)
        s = self.prim.M @ _l2(_np(emb))
        ev['v'][self.primario] = self._res('deep', s, self.prim.names, ms_reid)
        ev['v'][self.primario + '+voto'] = {'tipo': 'voto', 'orden': [track.current_identity],
                                            'score': float(track.current_confidence), 'ms': 0.0}
        for d in self.deeps:
            try:
                t = time.perf_counter()
                e2 = _np(d.ext(crop_resized)[0])
                ev['v'][d.nombre] = self._res('deep', d.M @ _l2(e2), d.names, (time.perf_counter() - t) * 1000)
            except Exception:
                self.errores['monitor_' + d.nombre] += 1
        sims_c = None
        if self.clasica is not None:
            try:
                desc, tms = self.ex_clasico.descriptores(crop_resized, medir=True)
                sims_c = self.clasica.sims(desc)
                for c in COMBOS:
                    n = nombre_combo(c)
                    ev['v'][n] = self._res('clasico', sims_c[n], self.clasica.nombres, sum(tms[k] for k in c))
            except Exception:
                self.errores['monitor_clasicos'] += 1
        if sims_c is not None and self.fusiones:
            sp = s[self.f_idx_p]
            sc = sims_c[TRIPLE][self.f_idx_c]
            for n, a in self.fusiones:
                ev['v'][n] = self._res('fusion', a * sp + (1 - a) * sc, self.f_names, ms_reid + sum(tms.values()))
        self.eventos.append(ev)
        self.ev_por_track[tid].append(ev)
        if gt:
            self._contar(ev, +1)
        if self.crop_cada and len(self.eventos) % self.crop_cada == 0 and self.n_crops < 1500:
            os.makedirs(os.path.join(self.dir, 'crops'), exist_ok=True)
            cv2.imwrite(os.path.join(self.dir, 'crops', f'ev{ev["id"]:06d}_t{tid}.jpg'), crop_resized)
            self.n_crops += 1

    # ------------------------------------------------------------------ etiquetado
    def _contar(self, ev, signo):
        gt = ev['gt']
        if gt is None or gt == DESCONOCIDO:
            return
        for v, x in ev['v'].items():
            c = self.cont[v]
            c['n'] += signo
            c['top1'] += signo * int(x['orden'][0] == gt)
            c['top5'] += signo * int(gt in x['orden'][:5]) if x['tipo'] != 'voto' else 0

    def _reasignar(self, ev, nuevo):
        if ev['gt'] == nuevo:
            return
        self._contar(ev, -1)
        ev['gt'] = nuevo
        self._contar(ev, +1)

    def _set_label(self, tid, gt):
        if gt is None:
            self.labels.pop(tid, None)
        else:
            self.labels[tid] = gt
        for ev in self.ev_por_track.get(tid, []):
            self._reasignar(ev, gt)
        self.log_etiquetas.append((round(time.time() - self.t0, 2), tid, gt or '', 'borrar' if gt is None else 'asignar'))
        print(f'   🏷️  track {tid} -> {gt if gt else "(sin etiqueta)"}')

    def _objetivo(self):
        if self.sel_track is not None:
            return self.sel_track
        if self._n_confirm == 1:
            return self._tracks_vis[0][0]
        return None

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        hit, area = None, None
        for tid, b in self._tracks_vis:
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                a = (b[2] - b[0]) * (b[3] - b[1])
                if area is None or a < area:
                    hit, area = tid, a
        self.sel_track = hit
        if hit is not None:
            print(f'   🖱️  track {hit} seleccionado')

    def on_key(self, key):
        if key in (255, -1, 0):
            return False
        ch = chr(key) if 32 <= key < 127 else ''
        if ch and (ch.isdigit() or ch == 'u'):
            nombre = DESCONOCIDO if ch == 'u' else None
            if ch.isdigit():
                i = 9 if ch == '0' else int(ch) - 1
                if i >= len(self.enrolados):
                    print(f'   ⚠️  No hay persona enrolada Nº{ch}')
                    return True
                nombre = self.enrolados[i]
            if self.modo_sesion:
                self.sesion_gt = nombre
                print(f'   🏷️  modo sesión: toda la sesión = {nombre}')
                return True
            t = self._objetivo()
            if t is None:
                print('   ⚠️  Haz clic sobre una persona (o deja un solo track en pantalla) antes de etiquetar')
            else:
                self._set_label(t, nombre)
            return True
        if ch == 'x':
            if self.modo_sesion:
                self.sesion_gt = None
            elif self._objetivo() is not None:
                self._set_label(self._objetivo(), None)
            return True
        if ch == 'l':
            self.modo_sesion = not self.modo_sesion
            print(f'   🔁 modo sesión: {"ON" if self.modo_sesion else "OFF"}')
            return True
        if ch in ('+', '='):
            self.altura += 5
            return True
        if ch in ('-', '_'):
            self.altura = max(0.0, self.altura - 5)
            return True
        if ch == 'a':
            self.angulo = ANGULOS[(ANGULOS.index(self.angulo) + 1) % len(ANGULOS)] if self.angulo in ANGULOS else ANGULOS[0]
            return True
        if ch == 'o':
            self.ocl = 1 - self.ocl
            return True
        if ch == 'g':
            self.grabar = not self.grabar
            if not self.grabar and self.writer is not None:
                self.writer.release()
                self.writer = None
            print(f'   {"🔴 grabando" if self.grabar else "⏹️  grabación detenida"}')
            return True
        if ch == 't':
            self.mostrar_panel = not self.mostrar_panel
            return True
        return False

    # ------------------------------------------------------------------ panel
    @_mide
    def dibujar(self, frame, tracks):
        for tid, b in self._tracks_vis:
            gt = self.labels.get(tid)
            if tid == self.sel_track:
                cv2.rectangle(frame, (b[0] - 3, b[1] - 3), (b[2] + 3, b[3] + 3), (0, 255, 255), 3)
            if gt:
                cv2.putText(frame, f'GT: {gt}', (b[0], min(frame.shape[0] - 5, b[3] + 18)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 255), 2)
        if not self.mostrar_panel:
            return
        filas = []
        if self.completo:
            for v, c in self.cont.items():
                if c['n'] > 0:
                    filas.append((v.replace(self.primario, self.primario[:8]), c['n'], 100 * c['top1'] / c['n']))
            filas.sort(key=lambda r: -r[2])
        w, lh = 430, 18
        h = 14 + lh * (6 + min(len(filas), 14))
        x0, y0 = frame.shape[1] - w - 10, 10
        ov = frame.copy()
        cv2.rectangle(ov, (x0, y0), (x0 + w, y0 + h), (20, 20, 20), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
        y = y0 + 16
        wall = np.mean(self.lat['wall_sin_monitor'][-60:]) if self.lat['wall_sin_monitor'] else 0
        rec = '  [REC]' if self.grabar else ''
        lineas = [(f'PRUEBAS [{self.modo}]  FPS sin monitor: {1000 / wall:.1f}{rec}' if wall > 0 else f'PRUEBAS [{self.modo}]', (0, 255, 200)),
                  (f'Altura: {self.altura:.0f} m  Angulo: {self.angulo}  Oclusion: {"SI" if self.ocl else "no"}', (255, 255, 255)),
                  (f'Sel: {self.sel_track}  GT sesion: {self.sesion_gt if self.modo_sesion else "-"}  eventos: {len(self.eventos)}', (255, 255, 255)),
                  ('Dig 1-9/0=persona  u=desconocido  x=borrar  l=sesion  clic=elegir', (180, 180, 180))]
        if self.completo:
            lineas.append(('VARIANTE                  n    Top-1%', (200, 200, 200)))
        for txt, col in lineas:
            cv2.putText(frame, txt, (x0 + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
            y += lh
        for nombre, n, t1 in filas[:14]:
            cv2.putText(frame, f'{nombre[:24]:<24}{n:>5}{t1:>8.1f}', (x0 + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
            y += lh

    # ------------------------------------------------------------------ telemetría de GPU
    def _telemetria(self):
        ultimo_frames, ultimo_t = 0, time.time()
        while not self._parar:
            time.sleep(1.0)
            ahora = time.time()
            fps = (self.frames - ultimo_frames) / max(ahora - ultimo_t, 1e-6)
            ultimo_frames, ultimo_t = self.frames, ahora
            util = mem = temp = None
            try:
                out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu',
                                      '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2).stdout
                util, mem, temp = [float(x) for x in out.strip().splitlines()[0].split(',')]
            except Exception:
                pass
            try:
                import psutil
                ram = psutil.Process().memory_info().rss / 1e9
            except Exception:
                ram = None
            self.telemetria.append((round(ahora - self.t0, 1), round(fps, 2), util, mem, temp, ram))

    # ------------------------------------------------------------------ guardado
    def _hw(self):
        hw = {'cpu': platform.processor() or platform.machine(), 'so': platform.platform()}
        try:
            import torch
            if torch.cuda.is_available():
                hw['gpu'] = torch.cuda.get_device_name(0)
        except Exception:
            pass
        return hw

    def _sistema_json(self):
        dur = time.time() - self.t0
        wsm = self.lat['wall_sin_monitor']
        proc = self.lat['proc_sin_monitor']
        gpu_pico = gpu_res = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_pico = torch.cuda.max_memory_allocated() / 1e9
                gpu_res = torch.cuda.max_memory_reserved() / 1e9
        except Exception:
            pass
        ram = [t[5] for t in self.telemetria if t[5] is not None]
        util = [t[2] for t in self.telemetria if t[2] is not None]
        temp = [t[4] for t in self.telemetria if t[4] is not None]
        smi_mem = [t[3] for t in self.telemetria if t[3] is not None]
        if gpu_pico is None and smi_mem:
            gpu_pico = max(smi_mem) / 1000.0
        altos = np.asarray(self.altos) if self.altos else None
        return {
            'sesion': os.path.basename(self.dir), 'modo': self.modo, 'primario': self.primario, 'hw': self._hw(),
            'duracion_s': dur, 'frames': self.frames, 'altura_inicial': self.altura_inicial,
            'fps_efectivo': 1000.0 / float(np.mean(wsm)) if wsm else None,
            'capacidad_fps': 1000.0 / float(np.mean(proc)) if proc and np.mean(proc) > 0 else None,
            'lat_ms': {k: _pcts(self.lat[k]) for k in ('iter_wall', 'wall_sin_monitor', 'proc_sin_monitor', 'lectura',
                                                       'deteccion', 'tracking', 'reid_primario', 'overhead_monitor')},
            'overhead_monitor_ms_medio': float(np.mean(self.lat['overhead_monitor'])) if self.lat['overhead_monitor'] else 0.0,
            'gpu_mem_pico_gb': gpu_pico, 'gpu_mem_reservada_gb': gpu_res, 'ram_gb': max(ram) if ram else None,
            'gpu_util_media': float(np.mean(util)) if util else None, 'gpu_temp_max': max(temp) if temp else None,
            'lecturas': {'intentos': self.intentos, 'fallos': self.fallos,
                         'pct_fallos': 100 * self.fallos / max(self.intentos, 1)},
            'reconexiones': self.reconexiones, 'tardios_pct': 100 * self.tardios / max(self.frames, 1),
            'fps_nominal_usado': self.fps_nominal, 'errores': dict(self.errores),
            'estabilidad_pct': 100 * (1 - self.frames_err / max(self.frames, 1)),
            'detecciones': {'personas_por_frame_media': float(np.mean(self.n_personas)) if self.n_personas else 0.0,
                            'frames_sin_personas_pct': 100 * self.sin_personas / max(len(self.n_personas), 1),
                            'alto_caja_px': ({'p10': float(np.percentile(altos, 10)), 'p50': float(np.percentile(altos, 50)),
                                              'p90': float(np.percentile(altos, 90))} if altos is not None else None)},
            'umbrales': self.umbrales, 'enrolados': self.enrolados, 'variantes': self.tipos, 'n_eventos': len(self.eventos),
            'nota': 'proc_sin_monitor = tiempo de procesamiento por frame sin esperar el stream y sin el costo de las pruebas'}

    def guardar(self):
        d = self.dir
        with open(os.path.join(d, 'eventos_reid.csv'), 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(COLS_EVENTOS)
            for ev in self.eventos:
                gt = ev['gt'] or ''
                for v, x in ev['v'].items():
                    orden = x['orden']
                    rank = (orden.index(gt) + 1) if (gt and gt != DESCONOCIDO and x['tipo'] != 'voto' and gt in orden) else ''
                    top5 = '' if x['tipo'] == 'voto' else '|'.join(orden[:5])
                    w.writerow([ev['id'], f"{ev['t']:.2f}", ev['frame'], ev['track'], gt, ev['altura'], ev['angulo'], ev['ocl'],
                                ev['crop_h'], ev['crop_w'], v, x['tipo'], orden[0] if orden else '', f"{x['score']:.4f}",
                                rank, top5, f"{x['ms']:.2f}"])
        with open(os.path.join(d, 'etiquetas.csv'), 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['t_s', 'track', 'gt', 'accion'])
            w.writerows(self.log_etiquetas)
        with open(os.path.join(d, 'telemetria.csv'), 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['t_s', 'fps', 'gpu_util_pct', 'gpu_mem_mb', 'gpu_temp_c', 'ram_gb'])
            w.writerows(self.telemetria)
        with open(os.path.join(d, 'sistema.json'), 'w', encoding='utf-8') as fh:
            json.dump(self._sistema_json(), fh, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
        print(f'   💾 Sesión guardada en {d}  ({len(self.eventos)} eventos)')

    def finalizar(self, total_reconnections=None):
        self._parar = True
        if self.writer is not None:
            self.writer.release()
        if total_reconnections is not None:
            self.reconexiones = max(self.reconexiones, total_reconnections)
        self.guardar()
        print('\n' + '=' * 70 + '\n🔬 ANÁLISIS AUTOMÁTICO DE LA SESIÓN\n' + '=' * 70)
        try:
            analizar_sesiones([self.dir], os.path.join(self.dir, 'analisis'))
        except Exception as e:
            import traceback
            print(f'⚠️  El análisis automático falló ({type(e).__name__}: {e}). Los datos están guardados; ejecuta:\n'
                  f'    python {R}/analisis_vuelo.py "{self.dir}"')
            traceback.print_exc(limit=3)