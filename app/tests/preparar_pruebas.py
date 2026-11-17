"""
PREPARACIÓN OFFLINE (datos reales capturados con la webcam)
===========================================================
Automatiza, en orden, lo que hay que tener listo ANTES de volar:
  1) bases .pkl de los 5 modelos (OSNet, RepViT M0.9, EfficientFormerV2 S0, FastViT T8, SwiftFormer XS)
  2) base y evaluación de los extractores clásicos HOG / LBP / HSV (individual, parejas, triple)
  3) (opcional) pruebas offline del documento sobre tus fotos de webcam: arquitecturas, umbral, fusión, etc.

Uso (desde app/):
    python pruebas/preparar_pruebas.py                     # pasos 1 y 2
    python pruebas/preparar_pruebas.py --con-suite         # + paso 3
    python pruebas/preparar_pruebas.py --solo-clasicos     # solo el paso 2
"""
import os
import sys
import time
import argparse
import subprocess

PRUEBAS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(PRUEBAS_DIR)
R = os.path.basename(PRUEBAS_DIR)      # nombre real de la carpeta (pruebas / tests)


def paso(titulo, cmd):
    print('\n' + '#' * 72 + f'\n# {titulo}\n#   $ ' + ' '.join(cmd) + '\n' + '#' * 72, flush=True)
    t = time.time()
    r = subprocess.run([sys.executable] + cmd, cwd=APP_DIR)
    ok = r.returncode == 0
    print(f'\n{"✅" if ok else "❌"} {titulo}  ({time.time() - t:.0f}s)')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--con-suite', action='store_true', help='ejecuta también pruebas_uav_reid.py sobre las fotos de webcam')
    ap.add_argument('--solo-clasicos', action='store_true')
    ap.add_argument('--sin-deep-en-clasicos', action='store_true', help='no compara los modelos profundos en la evaluación de clásicos')
    a = ap.parse_args()
    res = []
    if not a.solo_clasicos:
        res.append(('Bases .pkl de los 5 modelos', paso('1) Generar bases de los 5 modelos', ['generar_todas_las_bases.py'])))
    cmd = [R + '/extractores_clasicos.py', '--todo'] + ([] if a.sin_deep_en_clasicos else ['--con-deep', '--modelos', 'sistema'])
    res.append(('Extractores clásicos (HOG/LBP/HSV)', paso('2) Base + evaluación de extractores clásicos', cmd)))
    if a.con_suite and os.path.exists(os.path.join(PRUEBAS_DIR, 'pruebas_uav_reid.py')):
        res.append(('Suite offline', paso('3) Pruebas offline del documento (fotos de webcam)',
                                          [R + '/pruebas_uav_reid.py', '--bloques', 'softbio,arquitecturas,fusion,openset,umbral,tracking,estadistica,cualitativa',
                                           '--modelos', 'sistema', '--sin-yolo-ablacion']))) 
    print('\n' + '=' * 72 + '\nRESUMEN DE LA PREPARACIÓN\n' + '=' * 72)
    for n, ok in res:
        print(f'  {"✅" if ok else "❌"} {n}')
    print('\n➡️  Siguiente paso: python ' + R + '/lanzar_sesion.py --solo-verificar')


if __name__ == '__main__':
    main()