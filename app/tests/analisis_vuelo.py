"""
ANÁLISIS DE SESIONES DE VUELO (datos reales con el dron)
========================================================
Lee lo que escribe monitor_tiempo_real.py en  data/pruebas_tiempo_real/<sesion>/
    eventos_reid.csv   un renglón por (evento de Re-ID, variante)  [variante = modelo profundo,
                       combinación HOG/LBP/HSV, o fusión]
    sistema.json       tiempos, memoria, pérdida de frames, estabilidad, umbrales, etc.
    etiquetas.csv      etiquetas puestas en vivo por el operador
y genera las tablas del documento "pruebas_Q1_UAV_ReID" que se pueden medir con datos reales.

Uso:
    python pruebas/analisis_vuelo.py data/pruebas_tiempo_real/<sesion>            # una sesión
    python pruebas/analisis_vuelo.py data/pruebas_tiempo_real/*  --salida data/consolidado   # varias (p. ej. alturas distintas)
"""
import os
import sys
import csv
import json
import math
import argparse
from collections import OrderedDict, Counter, defaultdict
from datetime import datetime

import numpy as np

DESCONOCIDO = 'DESCONOCIDO'
COLS_EVENTOS = ['evento', 't_s', 'frame', 'track', 'gt', 'altura_m', 'angulo', 'oclusion', 'crop_h', 'crop_w',
                'variante', 'tipo', 'top1', 'score', 'rank_gt', 'top5', 'ms']
GRUPOS_ALTURA = [('<5 m', 0, 5), ('5-10 m', 5, 10), ('10-20 m', 10, 20), ('20-40 m', 20, 40),
                 ('40-60 m', 40, 60), ('>60 m', 60, 1e9)]
GRUPOS_RES = [('<64 px', 0, 64), ('64-95 px', 64, 96), ('96-127 px', 96, 128), ('128-191 px', 128, 192), ('≥192 px', 192, 1e9)]

CRITERIOS = [  # etiqueta, claves, sentido, mínimo, fuerte   (Tablas 23 y 34 del documento)
    ('Detección mAP@0.5 (%)', ['det_map50'], '>', 85, 90),
    ('Re-ID Rank-1 / Top-1 en vivo (%)', ['reid_rank1_propuesto'], '>', 82, 88),
    ('Re-ID Rank-5 / Top-5 en vivo (%)', ['reid_rank5'], '>', 90, 95),
    ('Re-ID mAP (se mide MRR en vivo, x100)', ['reid_mrr'], '>', 72, 80),
    ('F1-score (umbral óptimo)', ['reid_f1'], '>', 0.80, 0.87),
    ('Mejora vs modelo profundo solo (pp)', ['mejora_vs_deep'], '>', 4, 7),
    ('FPS del pipeline', ['fps'], '>', 15, 25),
    ('Latencia de procesamiento por frame (ms)', ['latencia_ms'], '<', 100, 50),
    ('Memoria GPU pico (GB)', ['mem_gb'], '<', 6, 4),
    ('Pérdida de frames (%)', ['perdida_pct'], '<', 10, 5),
    ('Estabilidad del pipeline (%)', ['estabilidad_pct'], '>', 90, 95),
    ('Cross-domain webcam→dron: Top-1 modelo principal (%)', ['cross_rank1'], '>', 60, 70),
    ('p-value (McNemar) frente al modelo principal', ['p_baseline'], '<', 0.05, 0.01)]


# ----------------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------------
def _f(x, d=None):
    try:
        return float(x) if x not in ('', None) else d
    except ValueError:
        return d


def cargar_sesion(d):
    nombre = os.path.basename(os.path.normpath(d))
    eventos = OrderedDict()
    ruta = os.path.join(d, 'eventos_reid.csv')
    if os.path.exists(ruta):
        with open(ruta, newline='', encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                ev = eventos.get(r['evento'])
                if ev is None:
                    ev = {'sesion': nombre, 'id': f"{nombre}#{r['evento']}", 't': _f(r['t_s'], 0.0),
                          'frame': int(float(r['frame'] or 0)), 'track': f"{nombre}:{r['track']}",
                          'gt': r['gt'] or None, 'altura': _f(r['altura_m']), 'angulo': r['angulo'] or '',
                          'ocl': int(float(r['oclusion'] or 0)), 'crop_h': int(float(r['crop_h'] or 0)),
                          'crop_w': int(float(r['crop_w'] or 0)), 'v': OrderedDict()}
                    eventos[r['evento']] = ev
                ev['v'][r['variante']] = {'tipo': r['tipo'], 'top1': r['top1'], 'score': _f(r['score'], 0.0),
                                          'rank': int(float(r['rank_gt'])) if r['rank_gt'] else None,
                                          'top5': r['top5'].split('|') if r['top5'] else [], 'ms': _f(r['ms'], 0.0)}
    sistema = {}
    rs = os.path.join(d, 'sistema.json')
    if os.path.exists(rs):
        with open(rs, encoding='utf-8') as fh:
            sistema = json.load(fh)
    sistema.setdefault('sesion', nombre)
    tracks = []
    rt = os.path.join(d, 'tracks.csv')
    if os.path.exists(rt):
        with open(rt, newline='', encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                tracks.append({'sesion': nombre, 'track': f"{nombre}:{r['track']}", 'primer': int(float(r['primer_frame'])),
                               'ultimo': int(float(r['ultimo_frame'])), 'n': int(float(r['frames_vistos'])),
                               'dur': _f(r['duracion_s'], 0.0), 'conf': r['confirmado'] == '1',
                               'ident': r['identidad_final'], 'conf_id': _f(r['conf_final'], 0.0), 'gt': r['gt'] or None})
    sistema['_tracks'] = tracks
    rd = os.path.join(d, 'deteccion_eval.json')
    if os.path.exists(rd):
        with open(rd, encoding='utf-8') as fh:
            sistema['deteccion_gt'] = json.load(fh)
    rr = os.path.join(d, 'deteccion_ref.json')
    if os.path.exists(rr):
        with open(rr, encoding='utf-8') as fh:
            sistema['deteccion_ref'] = json.load(fh)
    sistema['_dir'] = d
    return list(eventos.values()), sistema


def _conoc(e):
    return e['gt'] is not None and e['gt'] != DESCONOCIDO


def _mean(l):
    return float(np.mean(l)) if len(l) else None


def _pct(l):
    return 100 * float(np.mean(l)) if len(l) else None


# ----------------------------------------------------------------------------
# Métricas de decisión
# ----------------------------------------------------------------------------
def _decision(evs, v, t):
    tp = fp = fn = 0
    unk = acc_unk = 0
    for e in evs:
        x = e['v'][v]
        acepta = x['score'] >= t
        if e['gt'] == DESCONOCIDO:
            unk += 1
            acc_unk += int(acepta)
            fp += int(acepta)
        else:
            ok = acepta and x['top1'] == e['gt']
            tp += int(ok)
            fp += int(acepta and not ok)
            fn += int(not ok)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {'precision': p, 'recall': r, 'f1': 2 * p * r / (p + r) if p + r else 0.0,
            'far': (acc_unk / unk) if unk else None}


def _barrido(evs, v, ts=None):
    ts = np.arange(0.20, 0.99, 0.01) if ts is None else ts
    return [dict(umbral=float(t), **_decision(evs, v, t)) for t in ts]


def _auroc(evs, v):
    k = [e['v'][v]['score'] for e in evs if _conoc(e)]
    u = [e['v'][v]['score'] for e in evs if e['gt'] == DESCONOCIDO]
    if not k or not u:
        return None
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score([1] * len(k) + [0] * len(u), k + u))
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Tablas
# ----------------------------------------------------------------------------
def tabla_reid(eventos, variantes, umbrales, primario):
    filas = []
    for v, tipo in variantes.items():
        evs = [e for e in eventos if v in e['v'] and e['gt'] is not None]
        kn = [e for e in evs if _conoc(e)]
        if not kn:
            continue
        ok = [e['v'][v]['top1'] == e['gt'] for e in kn]
        con5 = [e for e in kn if e['v'][v]['top5']]
        ranks = [e['v'][v]['rank'] for e in kn if e['v'][v]['rank']]
        thr = umbrales.get(v, 0.70)
        d = _decision(evs, v, thr)
        sw = _barrido(evs, v)
        mejor = max(sw, key=lambda r: r['f1'])
        filas.append({'Variante': v + (' (principal)' if v == primario else ''), 'Tipo': tipo, 'Eventos': len(kn),
                      'Top-1 (%)': _pct(ok),
                      'Top-5 (%)': _pct([e['gt'] in e['v'][v]['top5'] for e in con5]) if con5 else None,
                      'MRR': _mean([1.0 / r for r in ranks]) if ranks else None,
                      'Score correctos': _mean([e['v'][v]['score'] for e, o in zip(kn, ok) if o]),
                      'Score incorrectos': _mean([e['v'][v]['score'] for e, o in zip(kn, ok) if not o]),
                      'ms/recorte': _mean([e['v'][v]['ms'] for e in evs]),
                      f'Precision@{thr:.2f}': d['precision'], f'Recall@{thr:.2f}': d['recall'], f'F1@{thr:.2f}': d['f1'],
                      'FAR': d['far'], 'AUROC known/unknown': _auroc(evs, v),
                      'F1 óptimo': mejor['f1'], 'Umbral óptimo': mejor['umbral']})
    return filas


def tabla_umbral(eventos, variantes, ref):
    filas = []
    for v in ([ref] if ref in variantes else []) + [x for x in variantes if x != ref][:0]:
        evs = [e for e in eventos if v in e['v'] and e['gt'] is not None]
        for t in (0.5, 0.6, 0.7, 0.8, 0.9):
            d = _decision(evs, v, t)
            filas.append({'Variante': v, 'Umbral': t, 'Precision': d['precision'], 'Recall': d['recall'],
                          'F1': d['f1'], 'FAR': d['far']})
    return filas


def _grupo(valor, grupos):
    if valor is None:
        return None
    return next((n for n, lo, hi in grupos if lo <= valor < hi), None)


def tabla_robustez(eventos, vars_sel, clave, grupos=None, orden=None):
    gr = defaultdict(list)
    for e in eventos:
        if not _conoc(e):
            continue
        g = clave(e)
        if g:
            gr[g].append(e)
    nombres = orden or sorted(gr)
    filas = []
    for g in nombres:
        if g not in gr:
            continue
        evs = gr[g]
        fila = {'Grupo': g, 'Eventos': len(evs)}
        for v in vars_sel:
            xs = [e['v'][v]['top1'] == e['gt'] for e in evs if v in e['v']]
            fila[f'{v} Top-1 (%)'] = _pct(xs)
        filas.append(fila)
    return filas


def tabla_imagen_video(eventos, variantes, primario):
    por_track = defaultdict(list)
    for e in eventos:
        if _conoc(e):
            por_track[e['track']].append(e)
    filas = []
    for v in variantes:
        acc = {1: [], 3: [], 5: []}
        for evs in por_track.values():
            evs = [e for e in evs if v in e['v']]
            for i in range(4, len(evs)):               # subconjunto común (≥5 eventos previos) para comparar k justo
                for k in acc:
                    voto = Counter(x['v'][v]['top1'] for x in evs[i - k + 1:i + 1]).most_common(1)[0][0]
                    acc[k].append(voto == evs[i]['gt'])
        if acc[1]:
            filas.append({'Variante': v, 'Imagen única (%)': _pct(acc[1]), 'Voto 3 frames (%)': _pct(acc[3]),
                          'Voto 5 frames (%)': _pct(acc[5]), 'Evaluaciones': len(acc[1])})
    vv = f'{primario}+voto'
    sel = [e for e in eventos if _conoc(e) and vv in e['v']]
    if sel:
        filas.append({'Variante': f'{primario} con voto del tracker (sistema real)',
                      'Imagen única (%)': _pct([e['v'][primario]['top1'] == e['gt'] for e in sel if primario in e['v']]),
                      'Voto del tracker (%)': _pct([e['v'][vv]['top1'] == e['gt'] for e in sel]), 'Evaluaciones': len(sel)})
    return filas


DET_AUTO = 'automático: consenso de los modelos sobre las personas enroladas en la captura (no es verdad terreno)'


def auto_etiquetas(eventos, umbrales, min_ev=3, share=0.6, mezcla_share=0.25):
    """Etiqueta cada track SIN intervención humana usando las personas enroladas (capturar_persona_mejorado.py):
    en cada evento vota la identidad de los modelos profundos que superan su umbral; la etiqueta del track es la identidad
    mayoritaria. 'mezcla' = dos o más identidades con evidencia fuerte dentro del mismo track."""
    por_track = defaultdict(list)
    for e in eventos:
        por_track[e['track']].append(e)
    out = {}
    for tk, evs in por_track.items():
        ids = []
        for e in evs:
            votos = [x['top1'] for v, x in e['v'].items() if x['tipo'] == 'deep' and x['score'] >= umbrales.get(v, 0.70)]
            if votos:
                ids.append(Counter(votos).most_common(1)[0][0])
        n = len(ids)
        if n >= min_ev:
            c = Counter(ids)
            mejor, nm = c.most_common(1)[0]
            out[tk] = {'label': mejor if nm / n >= share else None, 'n': n,
                       'mezcla': sum(1 for k, v in c.items() if v / n >= mezcla_share and v >= 3) >= 2}
        else:
            out[tk] = {'label': None, 'n': n, 'mezcla': False}
    return out


VENTANA_REAPARICION_S = 5.0          # una identidad que reaparece en un track nuevo antes de este tiempo = fragmentación probable


def _gaps_por_identidad(tracks, fps_ses, usar_gt):
    """Pares de tracks consecutivos de la misma persona. Devuelve (pares, personas)."""
    por = {}
    for t in tracks:
        if not t['conf']:
            continue
        clave = t['gt'] if (usar_gt and t['gt'] and t['gt'] != DESCONOCIDO) else (
            t['ident'] if (not usar_gt and t['ident'] and t['ident'] != 'Desconocido') else None)
        if clave:
            por.setdefault((t['sesion'], clave), []).append(t)
    pares = []
    for (ses, persona), ts in por.items():
        ts.sort(key=lambda t: t['primer'])
        fps = fps_ses.get(ses, 30.0)
        for a, b in zip(ts, ts[1:]):
            gap = (b['primer'] - a['ultimo']) / fps
            pares.append({'Sesión': ses, 'Persona': persona, 'Track A': a['track'].split(':')[-1], 'Track B': b['track'].split(':')[-1],
                          'Hueco (s)': gap, 'Tipo': 'solapados (duplicado)' if gap < 0 else
                          ('reaparición rápida' if gap <= VENTANA_REAPARICION_S else 'reaparición tardía')})
    return pares, por


def tabla_tracking(eventos, variantes, primario, sistemas=()):
    tracks = [t for s in sistemas for t in (s.get('_tracks') or [])]
    fps_ses = {s.get('sesion'): float(s.get('fps_nominal_usado') or 30.0) for s in sistemas}
    umbrales_t = {}
    for s_ in sistemas:
        umbrales_t.update(s_.get('umbrales', {}))
    auto = auto_etiquetas(eventos, umbrales_t)
    n_mezcla_auto = sum(1 for a_ in auto.values() if a_['mezcla'])
    tracks_auto = [dict(t, gt=(auto.get(t['track']) or {}).get('label')) for t in tracks]
    n_sw_auto = len(_gaps_por_identidad(tracks_auto, fps_ses, usar_gt=True)[0]) if tracks else 0
    ayuda_gt = 'Etiqueta personas (clic + número) o usa ETIQUETAS_CSV'
    por_gt, por_track_gt = defaultdict(set), defaultdict(set)
    for e in eventos:
        if _conoc(e):
            por_gt[e['gt']].add(e['track'])
            por_track_gt[e['track']].add(e['gt'])
    filas = []
    if tracks:
        conf = [t for t in tracks if t['conf']]
        durs = np.array([t['dur'] for t in conf]) if conf else np.array([])
        minutos = sum((s.get('duracion_s') or 0) for s in sistemas) / 60.0
        filas += [{'Concepto': 'Tracks creados', 'Valor': len(tracks), 'Detalle': 'incluye los que nunca llegaron a confirmarse'},
                  {'Concepto': 'Tracks confirmados', 'Valor': len(conf), 'Detalle': 'superaron el mínimo de frames del sistema'}]
        if len(durs):
            filas += [{'Concepto': 'Duración de un track confirmado (s), mediana', 'Valor': float(np.median(durs)),
                       'Detalle': f'p10={np.percentile(durs, 10):.1f}s  p90={np.percentile(durs, 90):.1f}s'},
                      {'Concepto': 'Tracks cortos (<1 s) (%)', 'Valor': 100 * float((durs < 1.0).mean()),
                       'Detalle': 'valor alto = fragmentación o falsos positivos'}]
        if minutos > 0:
            filas.append({'Concepto': 'Tracks confirmados por minuto', 'Valor': len(conf) / minutos, 'Detalle': f'{minutos:.1f} min de sesión'})
        simult = [s.get('detecciones') or {} for s in sistemas]
        if simult:
            filas.append({'Concepto': 'Personas simultáneas (media | máx)',
                          'Valor': f"{np.mean([d.get('personas_por_frame_media', 0) for d in simult]):.2f} | {max(d.get('personas_max', 0) for d in simult)}",
                          'Detalle': 'detecciones por frame'})
        pares, _ = _gaps_por_identidad(tracks, fps_ses, usar_gt=False)
        filas += [{'Concepto': f'Reapariciones de la misma identidad en un track nuevo (≤{VENTANA_REAPARICION_S:.0f} s)',
                   'Valor': sum(p['Tipo'] == 'reaparición rápida' for p in pares),
                   'Detalle': 'fragmentación probable, medida con la identidad del sistema (sin etiquetas)'},
                  {'Concepto': 'Tracks simultáneos con la misma identidad (duplicados)',
                   'Valor': sum(p['Tipo'].startswith('solapados') for p in pares), 'Detalle': 'dos tracks para una misma persona a la vez'},
                  {'Concepto': 'Tracks confirmados con identidad "Desconocido" (%)',
                   'Valor': 100 * sum(t['ident'] in ('Desconocido', '') for t in conf) / max(len(conf), 1),
                   'Detalle': 'no superaron el umbral de identificación'}]
        gt_t = [t for t in tracks if t['gt']]
        if gt_t:
            pg, _ = _gaps_por_identidad(tracks, fps_ses, usar_gt=True)
            conoc = [t for t in gt_t if t['gt'] != DESCONOCIDO]
            filas += [{'Concepto': 'ID switches reales por fragmentación (Σ tracks por persona − 1)', 'Valor': len(pg), 'Detalle': 'con etiquetas del operador'},
                      {'Concepto': 'Tracks con identidad final correcta (%)',
                       'Valor': 100 * sum(t['ident'] == t['gt'] for t in conoc) / max(len(conoc), 1), 'Detalle': f'{len(conoc)} tracks etiquetados'}]
        else:
            filas.append({'Concepto': 'ID switches por fragmentación (Σ tracks por persona − 1)',
                          'Valor': n_sw_auto if auto else None, 'Detalle': DET_AUTO if auto else ayuda_gt})
        filas.append({'Concepto': 'Tracks con 2+ personas distintas (mezcla de identidades)',
                      'Valor': int(sum(len(g) > 1 for g in por_track_gt.values())) if por_track_gt else (n_mezcla_auto if auto else None),
                      'Detalle': 'con etiquetas' if por_track_gt else (DET_AUTO if auto else ayuda_gt)})
    else:       # sesiones antiguas sin tracks.csv: solo lo que se pueda deducir de los eventos
        filas = [{'Concepto': 'Tracks con Re-ID', 'Valor': len({e['track'] for e in eventos}), 'Detalle': 'sesión sin tracks.csv (versión anterior)'},
                 {'Concepto': 'Tracks etiquetados', 'Valor': len(por_track_gt), 'Detalle': ''},
                 {'Concepto': 'Personas etiquetadas', 'Valor': len(por_gt), 'Detalle': ''},
                 {'Concepto': 'ID switches por fragmentación (Σ tracks por persona − 1)',
                  'Valor': int(sum(max(len(t) - 1, 0) for t in por_gt.values())), 'Detalle': 'con etiquetas'},
                 {'Concepto': 'Tracks con 2+ personas distintas', 'Valor': int(sum(len(g) > 1 for g in por_track_gt.values())), 'Detalle': ''}]

    # ---------------- estabilidad temporal por variante (no necesita etiquetas) ----------------
    por_track = defaultdict(list)
    for e in eventos:
        por_track[e['track']].append(e)
    est = []
    for v, tipo in variantes.items():
        trans = cambios = c2i = i2c = 0
        rachas, consist, minutos_v = [], [], 0.0
        for evs in por_track.values():
            evs = [e for e in evs if v in e['v']]
            if not evs:
                continue
            ids = [e['v'][v]['top1'] for e in evs]
            trans += len(ids) - 1
            cambios += sum(a != b for a, b in zip(ids, ids[1:]))
            run = 1
            for a, b in zip(ids, ids[1:]):
                if a == b:
                    run += 1
                else:
                    rachas.append(run)
                    run = 1
            rachas.append(run)
            if len(ids) >= 3:
                consist.append(Counter(ids).most_common(1)[0][1] / len(ids))
            minutos_v += max(evs[-1]['t'] - evs[0]['t'], 0.0) / 60.0
            for a, b, ea in zip(ids, ids[1:], evs):
                if _conoc(ea):
                    c2i += int(a == ea['gt'] and b != ea['gt'])
                    i2c += int(a != ea['gt'] and b == ea['gt'])
        if trans:
            est.append({'Variante': v, 'Tipo': tipo, 'Transiciones': trans, 'Cambios de identidad': cambios,
                        'Cambios (%)': 100 * cambios / trans, 'Estabilidad (%)': 100 - 100 * cambios / trans,
                        'Racha media (eventos)': float(np.mean(rachas)) if rachas else None,
                        'Consistencia del track (%)': 100 * float(np.mean(consist)) if consist else None,
                        'Cambios por minuto': (cambios / minutos_v) if minutos_v > 0 else None,
                        'Correcto→incorrecto': c2i if por_gt else None, 'Incorrecto→correcto': i2c if por_gt else None})
    # ---------------- tracks por persona y fragmentos ----------------
    personas, fragmentos = [], []
    if tracks:
        usar_gt = any(t['gt'] and t['gt'] != DESCONOCIDO for t in tracks)
        pares, grupos = _gaps_por_identidad(tracks, fps_ses, usar_gt)
        acum = defaultdict(lambda: {'tracks': 0, 'frames': 0, 'dur': 0.0, 'gaps': [], 'rapidas': 0})
        for (ses, persona), ts in grupos.items():
            a = acum[persona]
            a['tracks'] += len(ts)
            a['frames'] += sum(t['n'] for t in ts)
            a['dur'] += sum(t['dur'] for t in ts)
        for pr in pares:
            a = acum[pr['Persona']]
            a['gaps'].append(pr['Hueco (s)'])
            a['rapidas'] += int(0 <= pr['Hueco (s)'] <= VENTANA_REAPARICION_S)
        for persona, a in sorted(acum.items()):
            personas.append({'Persona': persona, 'Fuente': 'etiqueta (GT)' if usar_gt else 'identidad del sistema',
                             'Tracks': a['tracks'], 'Frames vistos': a['frames'], 'Duración total (s)': a['dur'],
                             'Hueco mediano entre tracks (s)': float(np.median(a['gaps'])) if a['gaps'] else None,
                             f'Reapariciones ≤{VENTANA_REAPARICION_S:.0f} s': a['rapidas'],
                             'ID switches estimados (tracks − sesiones)': a['tracks'] - len({s for (s, p) in grupos if p == persona})})
        fragmentos = sorted(pares, key=lambda r: abs(r['Hueco (s)']))[:30]
    return filas, est, personas, fragmentos


def _mcnemar(a, b):
    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    c10 = sum(1 for x, y in zip(a, b) if (not x) and y)
    if b01 + c10 == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(min(b01, c10), b01 + c10, 0.5).pvalue)
    except Exception:
        try:
            from scipy.stats import binom_test
            return float(binom_test(min(b01, c10), b01 + c10, 0.5))
        except Exception:
            return float('nan')


def tabla_estadistica(eventos, variantes, ref):
    from scipy import stats
    filas = []
    base = [e for e in eventos if _conoc(e) and ref in e['v']]
    rng = np.random.RandomState(42)
    for v in variantes:
        if v == ref:
            continue
        com = [e for e in base if v in e['v']]
        if len(com) < 10:
            continue
        a = np.array([e['v'][ref]['top1'] == e['gt'] for e in com], dtype=float)
        b = np.array([e['v'][v]['top1'] == e['gt'] for e in com], dtype=float)
        d = (b - a) * 100
        sd = d.std(ddof=1)
        t_p = float(stats.ttest_rel(b, a).pvalue) if sd > 0 else 1.0
        try:
            w_p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        except ValueError:
            w_p = 1.0
        boots = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p_mc = _mcnemar(a.astype(bool), b.astype(bool))
        sig = 'significativa (p<0.01)' if p_mc < 0.01 else 'significativa (p<0.05)' if p_mc < 0.05 else 'no significativa'
        filas.append({'Variante vs ' + ref: v, 'n': len(com), 'Δ Top-1 (pp)': float(d.mean()), 'McNemar p': p_mc,
                      't-test p': t_p, 'Wilcoxon p': w_p, 'IC95% inf': float(lo), 'IC95% sup': float(hi),
                      "Cohen's d": float(d.mean() / sd) if sd > 0 else None, 'Interpretación': sig})
    return filas


def tabla_anova(eventos, variantes_deep):
    from scipy import stats
    if len(variantes_deep) < 3:
        return []
    com = [e for e in eventos if _conoc(e) and all(v in e['v'] for v in variantes_deep)]
    if len(com) < 10:
        return []
    M = [np.array([e['v'][v]['top1'] == e['gt'] for e in com], dtype=float) for v in variantes_deep]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')         # sin varianza (todos aciertan) da avisos inofensivos
        f = stats.f_oneway(*M)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fr = stats.friedmanchisquare(*M)
        fr_row = {'Prueba': 'Friedman (medidas repetidas)', 'Estadístico': float(fr.statistic), 'p-value': float(fr.pvalue)}
    except Exception:
        fr_row = {'Prueba': 'Friedman (medidas repetidas)', 'Estadístico': None, 'p-value': None}
    return [{'Prueba': 'ANOVA (f_oneway)', 'Estadístico': float(f.statistic), 'p-value': float(f.pvalue),
             'Modelos': len(M), 'Eventos': len(com)}, {**fr_row, 'Modelos': len(M), 'Eventos': len(com)}]


def tabla_acuerdo(eventos, variantes):
    """Comparación entre arquitecturas SIN etiquetas: sobre los MISMOS recortes, cuánto coinciden sus identidades."""
    vs = [v for v, t in variantes.items() if t in ('deep', 'clasico', 'fusion')]
    if len(vs) < 2:
        return [], [], None
    deeps = [v for v in vs if variantes[v] == 'deep']
    ids = {v: {e['id']: e['v'][v]['top1'] for e in eventos if v in e['v']} for v in vs}
    cons = {}
    if len(deeps) >= 2:
        for e in eventos:
            vot = [e['v'][v]['top1'] for v in deeps if v in e['v']]
            if len(vot) >= 2:
                top = Counter(vot).most_common(2)
                if len(top) == 1 or top[0][1] > top[1][1]:
                    cons[e['id']] = top[0][0]
    prim = next((v for v in vs if variantes[v] == 'deep'), None)
    filas = []
    for v in vs:
        com = [k for k in ids[v] if k in cons]
        filas.append({'Variante': v, 'Tipo': variantes[v], 'Eventos': len(ids[v]),
                      'Acuerdo con el consenso de los modelos profundos (%)': (100 * np.mean([ids[v][k] == cons[k] for k in com])) if com else None,
                      'Eventos con consenso': len(com)})
    M = np.zeros((len(vs), len(vs)))
    for i, a in enumerate(vs):
        for j, b in enumerate(vs):
            com = [k for k in ids[a] if k in ids[b]]
            M[i, j] = 100 * np.mean([ids[a][k] == ids[b][k] for k in com]) if com else np.nan
    pares = [{'Variante': a, **{b: M[i, j] for j, b in enumerate(vs)}} for i, a in enumerate(vs)]
    return filas, pares, (vs, M)


def _fig_acuerdo(salida, datos):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return []
    vs, M = datos
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(vs) + 3), max(4.5, 0.5 * len(vs) + 2.5)))
    im = ax.imshow(M, vmin=0, vmax=100, cmap='viridis')
    ax.set_xticks(range(len(vs))); ax.set_yticks(range(len(vs)))
    ax.set_xticklabels([v[:16] for v in vs], rotation=60, ha='right', fontsize=7)
    ax.set_yticklabels([v[:16] for v in vs], fontsize=7)
    for i in range(len(vs)):
        for j in range(len(vs)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', color='w' if M[i, j] < 60 else 'k', fontsize=6)
    ax.set_title('Acuerdo de identidad entre variantes (%) sobre los mismos recortes', fontsize=9)
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(os.path.join(salida, 'fig_acuerdo_modelos.png'), dpi=150); plt.close(fig)
    return ['fig_acuerdo_modelos.png']


def tabla_eficiencia(sistemas):
    filas = []
    for s in sistemas:
        lat = s.get('lat_ms', {})
        g = lambda k, q: (lat.get(k) or {}).get(q)
        filas.append({'Sesión': s.get('sesion'), 'Modo': s.get('modo'), 'Modelo principal': s.get('primario'),
                      'Equipo': (s.get('hw') or {}).get('gpu') or (s.get('hw') or {}).get('cpu'),
                      'Altura (m)': s.get('altura_inicial'), 'Duración (s)': s.get('duracion_s'), 'Frames': s.get('frames'),
                      'FPS efectivo': s.get('fps_efectivo'), 'Capacidad (FPS sin esperar stream)': s.get('capacidad_fps'),
                      'Proc p50 (ms)': g('proc_sin_monitor', 'p50'), 'Proc p95 (ms)': g('proc_sin_monitor', 'p95'),
                      'Proc p99 (ms)': g('proc_sin_monitor', 'p99'), 'YOLO p50 (ms)': g('deteccion', 'p50'),
                      'Tracking p50 (ms)': g('tracking', 'p50'), 'Re-ID principal p50 (ms)': g('reid_primario', 'p50'),
                      'GPU mem pico (GB)': s.get('gpu_mem_pico_gb'), 'RAM (GB)': s.get('ram_gb'),
                      'GPU util media (%)': s.get('gpu_util_media'), 'GPU temp máx (°C)': s.get('gpu_temp_max'),
                      'Fallos de lectura (%)': (s.get('lecturas') or {}).get('pct_fallos'),
                      'Frames tardíos (%)': s.get('tardios_pct'), 'Reconexiones': s.get('reconexiones'),
                      'Estabilidad (%)': s.get('estabilidad_pct')})
    return filas


def tabla_deteccion_vivo(sistemas):
    filas = []
    for s in sistemas:
        d = s.get('detecciones') or {}
        a = d.get('alto_caja_px') or {}
        filas.append({'Sesión': s.get('sesion'), 'Altura (m)': s.get('altura_inicial'),
                      'Personas/frame (media)': d.get('personas_por_frame_media'),
                      'Frames sin personas (%)': d.get('frames_sin_personas_pct'),
                      'Alto de caja p10 (px)': a.get('p10'), 'Alto p50 (px)': a.get('p50'), 'Alto p90 (px)': a.get('p90')})
    return filas


# ----------------------------------------------------------------------------
# Semáforo
# ----------------------------------------------------------------------------
def tabla_deteccion_gt(sistemas):
    """Detección: anotación humana (evaluar_deteccion_vuelo.py) y/o detector de referencia automático (pseudo-GT)."""
    filas = []
    for s_ in sistemas:
        for clave, fuente in (('deteccion_gt', 'anotación humana'), ('deteccion_ref', 'detector de referencia (automático)')):
            d = s_.get(clave)
            if d:
                rt = d.get('recall_por_tamano') or {}
                fila = {'Sesión': s_.get('sesion'), 'Fuente': d.get('fuente_etiquetas') or fuente,
                        'Altura (m)': s_.get('altura_inicial'), 'Imágenes': d.get('imagenes'),
                        'Personas de referencia': d.get('gt'), 'TP': d.get('tp'), 'FP': d.get('fp'), 'FN': d.get('fn'),
                        'Precision': d.get('precision'), 'Recall': d.get('recall'), 'F1': d.get('f1'), 'mAP@0.5': d.get('map50')}
                for k, v in rt.items():
                    fila[f'Recall persona {k}'] = v
                filas.append(fila)
    return filas


R_DIR = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
AYUDA = {
    'det_map50': f'Automático: POSTPROCESO_COMPARACION=True (o DETECCION_REF_EN_VIVO=True) o python {R_DIR}/evaluar_deteccion_vuelo.py --sesion <carpeta> --todo --auto; con anotación humana: --todo',
    'reid_rank1_propuesto': 'Modo completo + etiquetar personas en vivo (clic + número) o repetir el vuelo grabado con ETIQUETAS_CSV',
    'reid_rank5': 'Igual que Rank-1: necesita etiquetas',
    'reid_mrr': 'Igual que Rank-1: necesita etiquetas',
    'reid_f1': 'Necesita etiquetas (idealmente también personas DESCONOCIDO con la tecla u)',
    'mejora_vs_deep': 'Necesita etiquetas y una variante distinta al modelo principal (fusión o clásicos): genera la base clásica y las de los demás modelos',
    'fps': 'Ejecuta una sesión (modo eficiencia o completo)', 'latencia_ms': 'Ejecuta una sesión (modo eficiencia o completo)',
    'mem_gb': 'Ejecuta una sesión con GPU (nvidia-smi / torch)', 'perdida_pct': 'Ejecuta una sesión (modo eficiencia o completo)',
    'estabilidad_pct': 'Ejecuta una sesión (modo eficiencia o completo)',
    'cross_rank1': 'Necesita etiquetas: mide cuánto cae la precisión de webcam (base) a dron (vuelo)',
    'p_baseline': 'Necesita etiquetas y ≥2 variantes con ≥10 eventos en común (completo + bases de los demás modelos / base clásica)'}


def tabla_cobertura(eventos, sistemas, variantes, n_conoc, n_desc, n_tracks):
    vid = sum(os.path.exists(os.path.join(s.get('_dir', ''), 'video_origen.mp4')) for s in sistemas)
    det = sum(1 for s in sistemas if s.get('deteccion_gt'))
    modos = ', '.join(sorted({str(s.get('modo')) for s in sistemas}))
    return [
        {'Dato': 'Sesiones analizadas', 'Cantidad': len(sistemas), 'Para qué sirve / cómo obtenerlo': f'modos: {modos}'},
        {'Dato': 'Eventos de Re-ID', 'Cantidad': len(eventos), 'Para qué sirve / cómo obtenerlo': 'estabilidad temporal y fragmentación (no necesitan etiquetas)'},
        {'Dato': 'Eventos con persona enrolada etiquetada', 'Cantidad': n_conoc,
         'Para qué sirve / cómo obtenerlo': 'Top-1, Top-5, MRR, F1, robustez, estadística. Etiqueta en vivo (clic + número) o usa ETIQUETAS_CSV'},
        {'Dato': 'Eventos DESCONOCIDO (persona no enrolada)', 'Cantidad': n_desc,
         'Para qué sirve / cómo obtenerlo': 'open-set, FAR y AUROC. Tecla u sobre una persona que no está en la base'},
        {'Dato': 'Variantes comparadas', 'Cantidad': len(variantes),
         'Para qué sirve / cómo obtenerlo': 'MODO completo + bases de los 5 modelos + base clásica (preparar_pruebas.py)'},
        {'Dato': 'Tracks registrados', 'Cantidad': n_tracks, 'Para qué sirve / cómo obtenerlo': 'tracking y fragmentación (tracks.csv)'},
        {'Dato': 'Sesiones con video grabado', 'Cantidad': vid, 'Para qué sirve / cómo obtenerlo': 'GRABAR=True: permite anotar cajas y repetir el vuelo'},
        {'Dato': 'Sesiones con detección vs detector de referencia (automático)', 'Cantidad': sum(1 for s in sistemas if s.get('deteccion_ref')),
         'Para qué sirve / cómo obtenerlo': 'Precision/Recall/mAP frente a YOLOv8x: se genera solo con POSTPROCESO_COMPARACION=True'},
        {'Dato': 'Sesiones con detección anotada (cajas)', 'Cantidad': det,
         'Para qué sirve / cómo obtenerlo': f'mAP/Precision/Recall reales: python {R_DIR}/evaluar_deteccion_vuelo.py --sesion <carpeta> --todo'}]


def semaforo(clave, notas=None):
    filas = []
    for et, claves, sent, mn, fu in CRITERIOS:
        val = next((clave[k] for k in claves if clave.get(k) is not None and not
                    (isinstance(clave[k], float) and math.isnan(clave[k]))), None)
        if val is None:
            est = '⚪ sin dato'
        elif (val > fu if sent == '>' else val < fu):
            est = '🟢 valor fuerte'
        elif (val > mn if sent == '>' else val < mn):
            est = '🟡 cumple mínimo'
        else:
            est = '🔴 no cumple'
        filas.append({'Aspecto': et, 'Obtenido': val, 'Mínimo (doc)': f'{sent}{mn}', 'Fuerte (doc)': f'{sent}{fu}', 'Estado': est,
                      'Cómo obtenerlo': AYUDA.get(claves[0], '') if val is None else (notas or {}).get(claves[0], '')})
    return filas


# ----------------------------------------------------------------------------
# Salidas
# ----------------------------------------------------------------------------
def _fmt(v):
    if v is None:
        return '-'
    if isinstance(v, float):
        return '-' if math.isnan(v) else (f'{v:.3f}' if abs(v) < 10 else f'{v:.1f}')
    return str(v)


def _md(filas):
    cols = []
    for f in filas:
        for c in f:
            if c not in cols:
                cols.append(c)
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    out += ['| ' + ' | '.join(_fmt(f.get(c)).replace('|', '/') for c in cols) + ' |' for f in filas]
    return '\n'.join(out)


def _print(filas):
    cols = []
    for f in filas:
        for c in f:
            if c not in cols:
                cols.append(c)
    w = {c: min(max(len(c), *(len(_fmt(f.get(c))) for f in filas)), 30) for c in cols}
    print('   ' + ' | '.join(c[:w[c]].ljust(w[c]) for c in cols))
    print('   ' + '-+-'.join('-' * w[c] for c in cols))
    for f in filas:
        print('   ' + ' | '.join(_fmt(f.get(c))[:w[c]].ljust(w[c]) for c in cols))


def _csv(ruta, filas):
    cols = []
    for f in filas:
        for c in f:
            if c not in cols:
                cols.append(c)
    with open(ruta, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(filas)


def _figuras(salida, eventos, variantes, tabla_r, sistemas, sw_curvas, rob_alt, rob_res, primario):
    figs = []
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return figs
    if tabla_r:
        fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(tabla_r)), 4))
        col = {'deep': 'tab:blue', 'clasico': 'tab:orange', 'fusion': 'tab:green'}
        ax.bar([r['Variante'].replace(' (principal)', '*') for r in tabla_r], [r['Top-1 (%)'] for r in tabla_r],
               color=[col.get(r['Tipo'], 'gray') for r in tabla_r])
        ax.set_ylabel('Top-1 en vivo (%)'); ax.set_title('Precisión en vuelo por variante (azul=profundo, naranja=clásico, verde=fusión)', fontsize=9)
        plt.xticks(rotation=45, ha='right', fontsize=7); fig.tight_layout()
        fig.savefig(os.path.join(salida, 'fig_top1_variantes.png'), dpi=150); plt.close(fig); figs.append('fig_top1_variantes.png')
    if sw_curvas:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for v, cur in sw_curvas.items():
            ax.plot([c['recall'] for c in cur], [c['precision'] for c in cur], label=v)
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('Precision-Recall por umbral'); ax.grid(alpha=.3); ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(os.path.join(salida, 'fig_precision_recall.png'), dpi=150); plt.close(fig); figs.append('fig_precision_recall.png')
    for nombre, filas, tit in (('fig_vs_altura.png', rob_alt, 'Top-1 vs altura del dron'), ('fig_vs_resolucion.png', rob_res, 'Top-1 vs alto del recorte (px)')):
        if filas:
            fig, ax = plt.subplots(figsize=(6.5, 4))
            for k in [c for c in filas[0] if c.endswith('Top-1 (%)')]:
                ax.plot([f['Grupo'] for f in filas], [f.get(k) for f in filas], marker='o', label=k.replace(' Top-1 (%)', ''))
            ax.set_ylabel('Top-1 (%)'); ax.set_title(tit); ax.grid(alpha=.3); ax.legend(fontsize=6)
            fig.tight_layout(); fig.savefig(os.path.join(salida, nombre), dpi=150); plt.close(fig); figs.append(nombre)
    kn = [e for e in eventos if _conoc(e) and primario in e['v']]
    if kn:
        nombres = sorted({e['gt'] for e in kn} | {e['v'][primario]['top1'] for e in kn})
        idx = {n: i for i, n in enumerate(nombres)}
        M = np.zeros((len(nombres), len(nombres)))
        for e in kn:
            M[idx[e['gt']], idx[e['v'][primario]['top1']]] += 1
        M = M / np.maximum(M.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(6, 5.2))
        im = ax.imshow(M, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(nombres))); ax.set_yticks(range(len(nombres)))
        ax.set_xticklabels([n[:10] for n in nombres], rotation=60, fontsize=7); ax.set_yticklabels([n[:10] for n in nombres], fontsize=7)
        ax.set_xlabel('Predicha'); ax.set_ylabel('Real'); ax.set_title(f'Matriz de confusión en vuelo ({primario})', fontsize=9)
        fig.colorbar(im); fig.tight_layout(); fig.savefig(os.path.join(salida, 'fig_confusion_principal.png'), dpi=150); plt.close(fig); figs.append('fig_confusion_principal.png')
    return figs


def analizar_sesiones(dirs, salida=None, verbose=True):
    eventos, sistemas = [], []
    for d in dirs:
        ev, sis = cargar_sesion(d)
        eventos += ev
        sistemas.append(sis)
    if not sistemas:
        print('❌ Sin sesiones para analizar')
        return None
    salida = salida or os.path.join(dirs[0], 'analisis') if len(dirs) == 1 else (salida or os.path.join(os.path.dirname(dirs[0]), 'consolidado'))
    os.makedirs(salida, exist_ok=True)
    variantes = OrderedDict()
    for e in eventos:
        for v, x in e['v'].items():
            variantes.setdefault(v, x['tipo'])
    umbrales = {}
    for s in sistemas:
        umbrales.update(s.get('umbrales', {}))
    primario = next((s.get('primario') for s in sistemas if s.get('primario')), None)
    tablas = OrderedDict()
    clave = {}

    def T(nombre, titulo, filas):
        if filas:
            tablas[nombre] = (titulo, filas)
            _csv(os.path.join(salida, nombre + '.csv'), filas)
            if verbose:
                print(f'\n📋 {titulo}')
                _print(filas)

    n_etiq = sum(1 for e in eventos if e['gt'] is not None)
    if verbose:
        print(f'\n🛰️  Sesiones: {len(sistemas)} | eventos de Re-ID: {len(eventos)} | etiquetados: {n_etiq} | variantes: {len(variantes)}')
    n_conoc = sum(1 for e in eventos if _conoc(e))
    n_desc = sum(1 for e in eventos if e['gt'] == DESCONOCIDO)
    n_tracks = sum(len(s.get('_tracks') or []) for s in sistemas)
    T('cobertura', 'Cobertura de datos de la sesión (qué hay y qué falta para llenar cada métrica)',
      tabla_cobertura(eventos, sistemas, variantes, n_conoc, n_desc, n_tracks))
    if verbose and n_etiq == 0:
        print('\n⚠️  SIN ETIQUETAS: no se pueden calcular Top-1/Top-5/MRR/F1/robustez/estadística.\n'
              '    Durante el vuelo haz clic sobre una persona y pulsa su número (o u si no está enrolada),\n'
              '    o repite el vuelo grabado con ETIQUETAS_CSV. La estabilidad y la fragmentación sí se calculan sin etiquetas.')
    T('eficiencia', 'Eficiencia del pipeline con datos reales del dron', tabla_eficiencia(sistemas))
    T('deteccion_gt', 'Detección con cajas anotadas sobre fotogramas del vuelo (Precision, Recall, F1, mAP@0.5)', tabla_deteccion_gt(sistemas))
    T('deteccion_vivo', '(Proxy en vivo): detecciones por frame y tamaño de las cajas (sin anotaciones de caja no hay P/R/mAP)',
      tabla_deteccion_vivo(sistemas))
    tr = tabla_reid(eventos, variantes, umbrales, primario)
    T('reid_variantes', 'Re-ID por variante (modelos profundos, HOG/LBP/HSV, fusiones)', tr)
    T('extractores_clasicos_vivo', 'Extractores clásicos individual / pareja / triple',
      [r for r in tr if r['Tipo'] == 'clasico'])
    T('fusiones_vivo', 'Fusión modelo profundo + extractores clásicos', [r for r in tr if r['Tipo'] == 'fusion'])
    if tr and primario:
        evs_ref = [e for e in eventos if e['gt'] is not None and primario in e['v']]
        T('umbral', f'Umbral de similitud ({primario})', tabla_umbral(eventos, variantes, primario))
    sw_curvas = {}
    for r in sorted(tr, key=lambda r: -(r['Top-1 (%)'] or 0))[:5]:
        v = r['Variante'].replace(' (principal)', '')
        sw_curvas[v] = _barrido([e for e in eventos if v in e['v'] and e['gt'] is not None], v)
    rec = {}
    for r in tr:
        v = r['Variante'].replace(' (principal)', '')
        rec[v] = round(r['Umbral óptimo'], 2)
    with open(os.path.join(salida, 'umbrales_recomendados.json'), 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, indent=2)
    T('openset', 'Tabla 27: open-set en vuelo (personas marcadas como DESCONOCIDO vs enroladas)',
      [{'Variante': r['Variante'], 'AUROC': r['AUROC known/unknown'], 'F1 óptimo': r['F1 óptimo'], 'Umbral óptimo': r['Umbral óptimo'],
        'FAR': r['FAR']} for r in tr if r['AUROC known/unknown'] is not None])

    def mejor(tipo):
        c = [r for r in tr if r['Tipo'] == tipo]
        return max(c, key=lambda r: r['Top-1 (%)'])['Variante'].replace(' (principal)', '') if c else None
    sel = [v for v in OrderedDict.fromkeys([primario, mejor('deep'), mejor('clasico'), mejor('fusion')]) if v]
    ra = tabla_robustez(eventos, sel, lambda e: _grupo(e['altura'], GRUPOS_ALTURA), orden=[n for n, _, _ in GRUPOS_ALTURA])
    T('robustez_altura', 'Robustez por altura del dron (altura registrada por el operador)', ra)
    T('robustez_angulo', 'Robustez por ángulo de cámara (registrado por el operador)',
      tabla_robustez(eventos, sel, lambda e: e['angulo'] or None))
    T('robustez_oclusion', 'Robustez ante oclusión (marcada por el operador)',
      tabla_robustez(eventos, sel, lambda e: 'con oclusión' if e['ocl'] else 'sin oclusión'))
    rr = tabla_robustez(eventos, sel, lambda e: _grupo(e['crop_h'], GRUPOS_RES), orden=[n for n, _, _ in GRUPOS_RES])
    T('robustez_resolucion', 'Robustez por resolución real del recorte (alto en píxeles)', rr)
    T('imagen_vs_video', 'Imagen única vs voto temporal por track',
      tabla_imagen_video(eventos, [v for v in sel], primario))
    t1, t2, t3, t4 = tabla_tracking(eventos, variantes, primario, sistemas)
    T('tracking_resumen', 'Tracking y fragmentación de identidades (proxy en vivo + etiquetas si existen)', t1)
    T('tracking_estabilidad', 'Estabilidad temporal de la identidad por variante', t2)
    ac1, ac2, ac_dat = tabla_acuerdo(eventos, variantes)
    T('acuerdo_modelos', 'Comparación entre arquitecturas sin etiquetas: acuerdo con el consenso de los modelos profundos', ac1)
    T('acuerdo_pares', 'Acuerdo de identidad entre pares de variantes (%) sobre los mismos recortes', ac2)
    T('tracking_personas', 'Tracks por persona (fragmentación de identidades)', t3)
    T('tracking_fragmentos', 'Reapariciones de una misma identidad en tracks distintos (los 30 casos más cercanos)', t4)
    if primario:
        T('estadistica', f'Significancia estadística de cada variante frente a {primario} (mismos eventos)',
          tabla_estadistica(eventos, variantes, primario))
    T('anova', 'Comparación global de arquitecturas profundas',
      tabla_anova(eventos, [v for v, t in variantes.items() if t == 'deep']))

    # ---- métricas clave para el semáforo ----
    prop = mejor('fusion') or mejor('deep')
    row = lambda v: next((r for r in tr if r['Variante'].replace(' (principal)', '') == v), None)
    rp, r0 = row(prop) if prop else None, row(primario) if primario else None
    if rp:
        clave['reid_rank1_propuesto'] = rp['Top-1 (%)']
        clave['reid_rank5'] = rp['Top-5 (%)'] if rp['Top-5 (%)'] is not None else (r0 or {}).get('Top-5 (%)')
        clave['reid_mrr'] = 100 * rp['MRR'] if rp['MRR'] is not None else None
        clave['reid_f1'] = rp['F1 óptimo']
    if rp and r0 and prop != primario:
        clave['mejora_vs_deep'] = rp['Top-1 (%)'] - r0['Top-1 (%)']
    if r0:
        clave['cross_rank1'] = r0['Top-1 (%)']
    alts = [r for r in tr if r['Tipo'] != 'voto' and not r['Variante'].endswith('(principal)')]
    if alts and primario:                      # p-valor del mejor competidor frente al modelo principal
        mejor_alt = max(alts, key=lambda r: r['Top-1 (%)'] or 0)['Variante']
        est = tablas.get('estadistica')
        fila = next((f for f in est[1] if f[f'Variante vs {primario}'] == mejor_alt), None) if est else None
        if fila:
            clave['p_baseline'] = fila['McNemar p']
    dg = [s['deteccion_gt']['map50'] for s in sistemas if s.get('deteccion_gt') and s['deteccion_gt'].get('map50') is not None]
    dr = [s['deteccion_ref']['map50'] for s in sistemas if s.get('deteccion_ref') and s['deteccion_ref'].get('map50') is not None]
    notas_sem = {}
    if dg:
        clave['det_map50'] = 100 * float(np.mean(dg))
    elif dr:
        clave['det_map50'] = 100 * float(np.mean(dr))
        notas_sem['det_map50'] = 'valor frente al detector de referencia (pseudo-verdad terreno), no anotación humana'
    efi = [s for s in sistemas if s.get('modo') == 'eficiencia'] or sistemas
    med = lambda k: (float(np.median([s[k] for s in efi if s.get(k) is not None])) if any(s.get(k) is not None for s in efi) else None)
    clave['fps'] = med('fps_efectivo')
    lat = [((s.get('lat_ms') or {}).get('proc_sin_monitor') or {}).get('p50') for s in efi]
    lat = [x for x in lat if x is not None]
    clave['latencia_ms'] = float(np.median(lat)) if lat else None
    mem = [s.get('gpu_mem_pico_gb') for s in efi if s.get('gpu_mem_pico_gb') is not None]
    clave['mem_gb'] = max(mem) if mem else None
    per = [max((s.get('lecturas') or {}).get('pct_fallos') or 0, s.get('tardios_pct') or 0) for s in efi if s.get('frames')]
    clave['perdida_pct'] = max(per) if per else None
    est_ = [s.get('estabilidad_pct') for s in efi if s.get('estabilidad_pct') is not None]
    clave['estabilidad_pct'] = min(est_) if est_ else None
    T('semaforo', 'Valores obtenidos vs mínimos/fuertes de la sesion', semaforo(clave, notas_sem))

    figs = _figuras(salida, eventos, variantes, tr, sistemas, sw_curvas, ra, rr, primario)
    if ac_dat is not None:
        figs += _fig_acuerdo(salida, ac_dat)
    notas = ['Métricas con etiquetas puestas en vivo por el operador.',
             'Top-1/Top-5/MRR.',
             'El mAP@0.5 de detección y el IDF1/MOTA reales requieren cajas/tracks anotados: graba el vuelo (PRUEBAS_GRABAR=1) y '
             'úsalo con pruebas_uav_reid.py (bloques deteccion y mot).',
             'La altura, el ángulo y la oclusión los declara el operador con las teclas + / - , a y o.']
    md = ['# Análisis de sesiones de vuelo', '', f'- Sesiones: {", ".join(s.get("sesion", "") for s in sistemas)}',
          f'- Eventos: {len(eventos)} (etiquetados: {n_etiq}) | Fecha de análisis: {datetime.now():%Y-%m-%d %H:%M}', '']
    for n, (tit, fl) in tablas.items():
        md += [f'## {tit}', '', _md(fl), '']
    if figs:
        md += ['## Figuras', ''] + [f'![]({f})' for f in figs] + ['']
    md += ['## Notas', ''] + [f'- {n}' for n in notas]
    with open(os.path.join(salida, 'resumen.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(md))
    with open(os.path.join(salida, 'analisis.json'), 'w', encoding='utf-8') as fh:
        json.dump({'metricas_clave': clave, 'tablas': {k: v[1] for k, v in tablas.items()}}, fh, indent=2, ensure_ascii=False,
                  default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    if verbose:
        print(f'\n📁 Análisis guardado en: {salida}  (resumen.md, CSV por tabla, figuras, umbrales_recomendados.json)')
    return {'tablas': tablas, 'clave': clave, 'salida': salida}


def main():
    ap = argparse.ArgumentParser(description='Analiza una o varias sesiones de vuelo')
    ap.add_argument('sesiones', nargs='+', help='carpetas data/pruebas_tiempo_real/<sesion>')
    ap.add_argument('--salida', default=None)
    a = ap.parse_args()
    dirs = [d for d in a.sesiones if os.path.isdir(d)]
    if not dirs:
        raise SystemExit('No hay carpetas de sesión válidas')
    analizar_sesiones(dirs, a.salida)


if __name__ == '__main__':
    main()