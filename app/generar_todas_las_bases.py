"""
Genera AUTOMÁTICAMENTE la base de embeddings (.pkl) de todos los modelos
y evalúa cada uno (matriz de confusión) sobre data/personas_base/.

Uso:
    python generar_todas_las_bases.py                      # los 5 modelos
    python generar_todas_las_bases.py repvit fastvit       # solo algunos
    python generar_todas_las_bases.py --sin-osnet          # solo los 4 nuevos

Salida:
    data/base_personas.pkl                       (OSNet, igual que siempre)
    data/base_personas_<modelo>.pkl              (cada modelo nuevo)
    data/resultados/<modelo>/...                 (matriz de confusión y métricas)
    data/resumen_generacion_modelos.csv          (tabla comparativa)
"""
import os
import sys
import gc
import csv
import time

# Sin ventanas de matplotlib (si no, plt.show() bloquearía el lote).
# Debe definirse ANTES de importar matplotlib.
os.environ.setdefault('MPLBACKEND', 'Agg')

import torch
from feature_extractors import ALL_MODELS
from config import DATA_DIR
import generar_base_personas_mejorado as gen


def _liberar_memoria():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv):
    sin_osnet = '--sin-osnet' in argv
    pedidos = [a.lower() for a in argv if not a.startswith('--')]
    modelos = pedidos or [m for m in ALL_MODELS if not (sin_osnet and m == 'osnet_x1_0')]

    desconocidos = [m for m in modelos if m not in ALL_MODELS]
    if desconocidos:
        print(f"❌ Modelos no válidos: {desconocidos}\n   Opciones: {ALL_MODELS}")
        return

    print("=" * 70)
    print(f"🏭 GENERACIÓN AUTOMÁTICA DE BASES: {', '.join(modelos)}")
    print("=" * 70)

    filas = []
    for i, modelo in enumerate(modelos, 1):
        print(f"\n\n{'#' * 70}\n# [{i}/{len(modelos)}] MODELO: {modelo}\n{'#' * 70}")
        os.environ['REID_MODEL'] = modelo      # lo leen feature_extractors y el generador
        t0 = time.time()
        fila = {"modelo": modelo, "estado": "OK", "error": ""}
        try:
            info = gen.main()
            if info is None:
                fila.update(estado="SIN_DATOS", error="No se procesó ninguna persona")
            else:
                fila.update(info)
        except Exception as e:                  # un modelo con error no detiene al resto
            fila.update(estado="ERROR", error=f"{type(e).__name__}: {e}")
            print(f"\n❌ {modelo} falló: {fila['error']}")
        fila["segundos"] = round(time.time() - t0, 1)
        filas.append(fila)
        _liberar_memoria()

    # ---------------- resumen ----------------
    columnas = ["modelo", "estado", "dim", "personas", "imagenes", "calidad_prom",
                "consistencia_prom", "umbral", "accuracy", "segundos", "base_path", "error"]
    ruta = os.path.join(DATA_DIR, "resumen_generacion_modelos.csv")
    with open(ruta, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columnas, extrasaction='ignore')
        w.writeheader()
        w.writerows(filas)

    print("\n\n" + "=" * 70)
    print("📊 RESUMEN COMPARATIVO (evaluación sobre las imágenes de personas_base)")
    print("=" * 70)
    print(f"{'Modelo':<20}{'Estado':<10}{'Dim':>6}{'Accuracy':>11}{'Consist.':>10}{'Tiempo(s)':>11}")
    print("-" * 70)
    for r in filas:
        acc = f"{r['accuracy']:.2f}%" if 'accuracy' in r else "-"
        con = f"{r['consistencia_prom']:.1f}%" if 'consistencia_prom' in r else "-"
        print(f"{r['modelo']:<20}{r['estado']:<10}{r.get('dim', '-'):>6}{acc:>11}{con:>10}{r['segundos']:>11}")
    print(f"\n💾 Resumen guardado en: {ruta}")
    print("💡 Siguiente paso:  REID_COMPARE=1 python deteccion_reid_dron_contador.py")


if __name__ == "__main__":
    main(sys.argv[1:])