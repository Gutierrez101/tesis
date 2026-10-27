"""
Comparación EN VIVO de varios modelos de Re-ID durante la detección.

Idea: el modelo principal (REID_MODEL) sigue mandando en la pantalla, los
tracks y el conteo, igual que siempre. Cada vez que el sistema hace Re-ID de
una persona, este módulo pasa EL MISMO recorte por los demás modelos cuya base
.pkl exista y guarda estadísticas por modelo.

IMPORTANTE: en video en vivo no hay "respuesta correcta" conocida, así que
estas estadísticas miden comportamiento (identificación, confianza, estabilidad,
velocidad y acuerdo entre modelos), NO accuracy. El accuracy sale de la matriz de
confusión que genera generar_todas_las_bases.py con las imágenes de personas_base.
"""
import os
import csv
import json
import time
import pickle
from collections import Counter
from datetime import datetime

import numpy as np
import cv2

from feature_extractors import build_extractor, get_base_path, get_threshold, ALL_MODELS


def _to_numpy(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _l2(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


class ModelStats:
    def __init__(self):
        self.n = 0                  # inferencias (eventos de Re-ID)
        self.n_ident = 0            # identificadas (>= umbral)
        self.conf_all = []
        self.conf_ident = []
        self.ms = []                # latencia (embedding + comparación)
        self._first_ms_skipped = False
        self.names = set()          # personas distintas identificadas
        self.tracks = set()
        self.last_ident = {}        # track_id -> última identidad
        self.transitions = 0        # comparaciones consecutivas dentro de un track
        self.switches = 0           # cuántas veces cambió la identidad
        self.agree_primary = 0
        self.n_primary = 0
        self.agree_consensus = 0
        self.n_consensus = 0
        self.errors = 0

    def record(self, track_id, ident, conf, ms, same_as_primary, same_as_consensus):
        self.n += 1
        self.conf_all.append(float(conf))
        if ident != "Desconocido":
            self.n_ident += 1
            self.conf_ident.append(float(conf))
            self.names.add(ident)
        if self._first_ms_skipped:          # se descarta la 1ª (calentamiento)
            self.ms.append(float(ms))
        self._first_ms_skipped = True
        self.tracks.add(track_id)
        if track_id in self.last_ident:
            self.transitions += 1
            if self.last_ident[track_id] != ident:
                self.switches += 1
        self.last_ident[track_id] = ident
        if same_as_primary is not None:
            self.n_primary += 1
            self.agree_primary += int(same_as_primary)
        if same_as_consensus is not None:
            self.n_consensus += 1
            self.agree_consensus += int(same_as_consensus)


class _Slot:
    def __init__(self, name, extractor, base):
        self.name = name
        self.extractor = extractor              # None para el modelo principal
        self.names = [p["nombre"] for p in base]
        self.matrix = _l2(np.array([np.asarray(p["embedding"], dtype=np.float32).reshape(-1)
                                    for p in base]))
        self.threshold = get_threshold(name)
        self.stats = ModelStats()

    def identify(self, emb):
        sims = self.matrix @ _l2(emb)
        i = int(np.argmax(sims))
        score = float(sims[i])
        return (self.names[i] if score >= self.threshold else "Desconocido"), score


class MultiModelEvaluator:
    def __init__(self, primary_name, primary_base, device='cpu',
                 models=None, output_dir='.'):
        self.primary = primary_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.slots = {}
        self.events = []
        self.start = datetime.now()

        self.slots[primary_name] = _Slot(primary_name, None, primary_base)
        print(f"\n🔬 Comparación de modelos activada (principal: {primary_name})")

        for name in (models or ALL_MODELS):
            if name == primary_name:
                continue
            path = get_base_path(name)
            if not os.path.exists(path):
                print(f"   ⚪ {name}: sin base ({os.path.basename(path)}), se omite")
                continue
            try:
                with open(path, 'rb') as f:
                    base = pickle.load(f)
                extractor = build_extractor(device=device, model_name=name)
                slot = _Slot(name, extractor, base)
                # Calentamiento + validación de dimensión (base y extractor deben coincidir)
                dummy = np.zeros((256, 128, 3), dtype=np.uint8)
                dim = _to_numpy(extractor(dummy)[0]).shape[0]
                if dim != slot.matrix.shape[1]:
                    raise ValueError(f"dimensión del modelo ({dim}) != base ({slot.matrix.shape[1]})")
                self.slots[name] = slot
                print(f"   ✅ {name}: {len(base)} persona(s), dim={dim}, umbral={slot.threshold}")
            except Exception as e:
                print(f"   ❌ {name}: {type(e).__name__}: {e} (se omite)")

        if len(self.slots) < 2:
            print("   ⚠️  Solo hay 1 modelo disponible; genera más bases con "
                  "generar_todas_las_bases.py")

    @property
    def n_models(self):
        return len(self.slots)

    # ------------------------------------------------------------------
    def evaluate(self, track_id, crop_resized, frame_count, primary_result):
        """primary_result = (identidad, confianza, ms) del modelo principal,
        que main() ya calculó (no se repite su inferencia)."""
        results = {self.primary: tuple(primary_result)}
        for name, slot in self.slots.items():
            if name == self.primary:
                continue
            try:
                t0 = time.perf_counter()
                emb = _to_numpy(slot.extractor(crop_resized)[0])   # .cpu() sincroniza la GPU
                ident, conf = slot.identify(emb)
                results[name] = (ident, conf, (time.perf_counter() - t0) * 1000)
            except Exception:
                slot.stats.errors += 1

        # Consenso = identidad más votada (solo con >= 3 modelos y sin empate)
        consensus = None
        if len(results) >= 3:
            top = Counter(r[0] for r in results.values()).most_common(2)
            if len(top) == 1 or top[0][1] > top[1][1]:
                consensus = top[0][0]

        prim_ident = results[self.primary][0]
        ts = datetime.now().isoformat(timespec='milliseconds')
        for name, (ident, conf, ms) in results.items():
            same_p = None if name == self.primary else (ident == prim_ident)
            same_c = None if consensus is None else (ident == consensus)
            self.slots[name].stats.record(track_id, ident, conf, ms, same_p, same_c)
            self.events.append((ts, frame_count, track_id, name, ident,
                                round(float(conf), 4), round(float(ms), 2)))
        return results

    # ------------------------------------------------------------------
    def summary(self):
        rows = []
        for name, slot in self.slots.items():
            s = slot.stats
            ms = np.array(s.ms) if s.ms else None
            rows.append({
                "modelo": name + (" (principal)" if name == self.primary else ""),
                "personas_en_base": len(slot.names),
                "umbral": slot.threshold,
                "inferencias": s.n,
                "identificadas": s.n_ident,
                "pct_identificadas": round(100 * s.n_ident / s.n, 1) if s.n else 0.0,
                "conf_media": round(float(np.mean(s.conf_all)), 3) if s.conf_all else 0.0,
                "conf_media_identificadas": round(float(np.mean(s.conf_ident)), 3) if s.conf_ident else 0.0,
                "ms_medio": round(float(ms.mean()), 1) if ms is not None else None,
                "ms_p95": round(float(np.percentile(ms, 95)), 1) if ms is not None else None,
                "personas_distintas_identificadas": len(s.names),
                "tracks_evaluados": len(s.tracks),
                "cambios_de_identidad": s.switches,
                "pct_cambios": round(100 * s.switches / s.transitions, 1) if s.transitions else 0.0,
                "pct_acuerdo_con_principal": (round(100 * s.agree_primary / s.n_primary, 1)
                                              if s.n_primary else None),
                "pct_acuerdo_con_consenso": (round(100 * s.agree_consensus / s.n_consensus, 1)
                                             if s.n_consensus else None),
                "errores": s.errors,
            })
        return rows

    def print_table(self):
        rows = self.summary()
        print("\n" + "=" * 100)
        print("🔬 COMPARACIÓN DE MODELOS EN VIVO")
        print("=" * 100)
        print(f"{'Modelo':<28}{'Inf.':>6}{'%Ident':>8}{'Conf':>7}{'ms':>7}"
              f"{'Cambios%':>10}{'Acuerdo%':>10}{'Consenso%':>11}")
        print("-" * 100)
        f = lambda v: "-" if v is None else v
        for r in rows:
            print(f"{r['modelo']:<28}{r['inferencias']:>6}{r['pct_identificadas']:>8}"
                  f"{r['conf_media']:>7}{str(f(r['ms_medio'])):>7}{r['pct_cambios']:>10}"
                  f"{str(f(r['pct_acuerdo_con_principal'])):>10}"
                  f"{str(f(r['pct_acuerdo_con_consenso'])):>11}")
        print("=" * 100)
        print("Cambios% = cuánto cambia la identidad de un mismo track (menos = más estable)")
        print("Acuerdo% = coincidencia con el modelo principal | Consenso% = con la mayoría (≥3 modelos)")

    def save(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rows = self.summary()
        p_csv = os.path.join(self.output_dir, f"comparacion_modelos_{ts}.csv")
        with open(p_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        p_json = os.path.join(self.output_dir, f"comparacion_modelos_{ts}.json")
        detalle = {
            "inicio": self.start.isoformat(), "fin": datetime.now().isoformat(),
            "modelo_principal": self.primary, "resumen": rows,
            "personas_identificadas": {n: sorted(s.stats.names) for n, s in self.slots.items()},
        }
        with open(p_json, 'w', encoding='utf-8') as f:
            json.dump(detalle, f, indent=2, ensure_ascii=False)
        p_ev = os.path.join(self.output_dir, f"comparacion_eventos_{ts}.csv")
        with open(p_ev, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "frame", "track_id", "modelo", "identidad", "confianza", "ms"])
            w.writerows(self.events)
        print(f"📊 Comparación guardada: {p_csv}\n   (+ .json y eventos: {os.path.basename(p_ev)})")

    def reset(self):
        for s in self.slots.values():
            s.stats = ModelStats()
        self.events.clear()
        self.start = datetime.now()

    # ------------------------------------------------------------------
    def draw_panel(self, frame):
        """Panel pequeño abajo a la izquierda (tecla 'm' lo oculta/muestra)."""
        rows = self.summary()
        line_h, pad = 20, 8
        h = pad * 2 + line_h * (len(rows) + 1)
        w = 430
        y0 = frame.shape[0] - h - 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, y0), (10 + w, y0 + h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        y = y0 + pad + 14
        cv2.putText(frame, f"{'MODELO':<20}{'%ID':>5}{'CONF':>7}{'ms':>7}{'CAMB%':>7}",
                    (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        for r in rows:
            y += line_h
            nombre = r['modelo'].replace(' (principal)', '*')[:19]
            ms = "-" if r['ms_medio'] is None else f"{r['ms_medio']:.0f}"
            txt = (f"{nombre:<20}{r['pct_identificadas']:>5.0f}{r['conf_media']:>7.2f}"
                   f"{ms:>7}{r['pct_cambios']:>7.0f}")
            cv2.putText(frame, txt, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)