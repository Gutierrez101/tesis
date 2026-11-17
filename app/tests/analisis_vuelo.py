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


def tabla_tracking(eventos, variantes, primario):
    por_gt = defaultdict(set)
    por_track_gt = defaultdict(set)
    for e in eventos:
        if _conoc(e):
            por_gt[e['gt']].add(e['track'])
            por_track_gt[e['track']].add(e['gt'])
    filas = [{'Concepto': 'Tracks con Re-ID', 'Valor': len({e['track'] for e in eventos})},
             {'Concepto': 'Tracks etiquetados', 'Valor': len(por_track_gt)},
             {'Concepto': 'Personas etiquetadas (enroladas)', 'Valor': len(por_gt)},
             {'Concepto': 'ID switches por fragmentación (Σ tracks por persona − 1)',
              'Valor': int(sum(max(len(t) - 1, 0) for t in por_gt.values()))},
             {'Concepto': 'Tracks con 2+ personas distintas (mezcla de identidades)',
              'Valor': int(sum(len(g) > 1 for g in por_track_gt.values()))}]
    por_track = defaultdict(list)
    for e in eventos:
        por_track[e['track']].append(e)
    est = []
    for v in variantes:
        sw = tr = 0
        for evs in por_track.values():
            ids = [e['v'][v]['top1'] for e in evs if v in e['v']]
            tr += max(len(ids) - 1, 0)
            sw += sum(a != b for a, b in zip(ids, ids[1:]))
        if tr:
            est.append({'Variante': v, 'Transiciones': tr, 'Cambios de identidad': sw, 'Cambios (%)': 100 * sw / tr})
    return filas, est


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
    f = stats.f_oneway(*M)
    try:
        fr = stats.friedmanchisquare(*M)
        fr_row = {'Prueba': 'Friedman (medidas repetidas)', 'Estadístico': float(fr.statistic), 'p-value': float(fr.pvalue)}
    except Exception:
        fr_row = {'Prueba': 'Friedman (medidas repetidas)', 'Estadístico': None, 'p-value': None}
    return [{'Prueba': 'ANOVA (f_oneway)', 'Estadístico': float(f.statistic), 'p-value': float(f.pvalue),
             'Modelos': len(M), 'Eventos': len(com)}, {**fr_row, 'Modelos': len(M), 'Eventos': len(com)}]


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
def semaforo(clave):
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
        filas.append({'Aspecto': et, 'Obtenido': val, 'Mínimo (doc)': f'{sent}{mn}', 'Fuerte (doc)': f'{sent}{fu}', 'Estado': est})
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
    T('eficiencia', 'Eficiencia del pipeline con datos reales del dron', tabla_eficiencia(sistemas))
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
    t1, t2 = tabla_tracking(eventos, variantes, primario)
    T('tracking_resumen', '(Proxy en vivo): tracking y fragmentación de identidades', t1)
    T('tracking_estabilidad', 'Estabilidad temporal de la identidad por variante', t2)
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
    if rp and r0:
        clave['mejora_vs_deep'] = rp['Top-1 (%)'] - r0['Top-1 (%)']
    if r0:
        clave['cross_rank1'] = r0['Top-1 (%)']
    if prop and primario and prop != primario:
        est = tablas.get('estadistica')
        if est:
            fila = next((f for f in est[1] if f[f'Variante vs {primario}'] == prop), None)
            if fila:
                clave['p_baseline'] = fila['McNemar p']
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
    T('semaforo', 'Valores obtenidos vs mínimos/fuertes de la sesion', semaforo(clave))

    figs = _figuras(salida, eventos, variantes, tr, sistemas, sw_curvas, ra, rr, primario)
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