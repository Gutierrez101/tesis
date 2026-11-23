"""
LANZADOR DE SESIONES DE PRUEBA CON EL DRON
==========================================
Verifica todo lo necesario (contenedor nginx, stream del dron, GPU, bases .pkl), configura las
variables de entorno y ejecuta  python deteccion_reid_dron_contador.py  con el monitor de pruebas.

Ejemplos (desde app/):
    python pruebas/lanzar_sesion.py --solo-verificar
    python pruebas/lanzar_sesion.py --modo eficiencia --sesion vuelo_eficiencia --altura 15
    python pruebas/lanzar_sesion.py --modo completo --sesion vuelo_20m --altura 20 --angulo cenital_parcial --grabar
    python pruebas/lanzar_sesion.py --consolidar          # junta TODAS las sesiones guardadas en un solo análisis
    python pruebas/lanzar_sesion.py --analizar            # re-analiza la última sesión
    python pruebas/lanzar_sesion.py --video data/pruebas_tiempo_real/<sesion>/video_origen.mp4 --etiquetas tiempos.csv --fps-video 25
                                                          # REPITE un vuelo grabado (sin dron ni nginx) con etiquetas automáticas

Equivale a (PowerShell):
    $env:PRUEBAS_MODO="completo"; $env:PRUEBAS_ALTURA_M="20"; python deteccion_reid_dron_contador.py
"""
import os
import sys
import time
import socket
import argparse
import subprocess
import urllib.request

PRUEBAS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(PRUEBAS_DIR)
for _d in (APP_DIR, PRUEBAS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)
R = os.path.basename(PRUEBAS_DIR)

URL_POR_DEFECTO = 'rtmp://127.0.0.1:1935/live/stream'
ANGULOS = ['frontal-oblicuo', 'lateral-oblicuo', 'posterior-oblicuo', 'cenital_parcial', 'cenital_extremo']


def ok(msg):
    print(f'  ✅ {msg}')


def aviso(msg):
    print(f'  ⚠️  {msg}')


def mal(msg):
    print(f'  ❌ {msg}')


def puerto_abierto(host, puerto, t=1.5):
    try:
        with socket.create_connection((host, puerto), timeout=t):
            return True
    except OSError:
        return False


def stream_activo(stat_url='http://127.0.0.1:8080/stat'):
    """Consulta el monitor de nginx-rtmp. Devuelve (accesible, hay_publicador, texto_xml)."""
    try:
        with urllib.request.urlopen(stat_url, timeout=3) as r:
            xml = r.read().decode('utf-8', 'replace')
        return True, ('<publishing' in xml), xml
    except Exception:
        return False, False, ''


def verificar(args, modo):
    problemas = 0
    print('\n🔎 VERIFICACIÓN PREVIA')
    print(f'  Python: {sys.version.split()[0]}', end='')
    print('  ✅' if sys.version_info[:2] >= (3, 10) else '  ⚠️ (se recomienda 3.10)')
    # GPU
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        ok(f'GPU: {out.splitlines()[0]}') if out else aviso('nvidia-smi no devolvió GPU (se usará CPU)')
    except Exception:
        aviso('nvidia-smi no disponible: las pruebas de eficiencia correrán en CPU y NO serán representativas')
    if getattr(args, 'video', None):
        if os.path.exists(args.video):
            ok(f'Repetición del vuelo grabado: {args.video} (no se necesita nginx ni dron)')
        else:
            mal(f'No existe el video: {args.video}')
            problemas += 1
        if getattr(args, 'etiquetas', None):
            (ok if os.path.exists(args.etiquetas) else mal)(f'CSV de etiquetas: {args.etiquetas}')
            problemas += 0 if os.path.exists(args.etiquetas) else 1
    else:
        # nginx
        host = args.host
        if puerto_abierto(host, 1935):
            ok(f'Servidor RTMP responde en {host}:1935')
        else:
            mal(f'Nada escucha en {host}:1935. Levanta el contenedor:  docker compose up -d --build rtmp')
            problemas += 1
        acc, pub, _ = stream_activo(f'http://{host}:8080/stat')
        if not acc:
            aviso(f'No se pudo leer http://{host}:8080/stat (¿contenedor sin el puerto 8080:80?)')
        elif pub:
            ok('Hay un publicador transmitiendo (el dron / OBS está enviando video)')
        else:
            aviso('El servidor está arriba pero NADIE transmite todavía: inicia la transmisión del dron a '
                  f'rtmp://IP_DE_ESTA_PC:1935/live/stream')
            if args.esperar_stream:
                print(f'     esperando hasta {args.esperar_stream}s a que llegue el stream...')
                t0 = time.time()
                while time.time() - t0 < args.esperar_stream:
                    if stream_activo(f'http://{host}:8080/stat')[1]:
                        ok('Stream detectado')
                        break
                    time.sleep(2)
                else:
                    mal('No llegó el stream a tiempo')
                    problemas += 1
    # bases
    try:
        from feature_extractors import ALL_MODELS, get_base_path
        print('  Bases de embeddings:')
        faltan = []
        for m in ALL_MODELS:
            p = get_base_path(m)
            (ok if os.path.exists(p) else aviso)(f'{m:<18} {os.path.basename(p)}' + ('' if os.path.exists(p) else '  (genera con generar_todas_las_bases.py)'))
            if not os.path.exists(p):
                faltan.append(m)
        if args.modelo not in faltan:
            pass
        else:
            mal(f'El modelo principal "{args.modelo}" no tiene base: no se puede iniciar')
            problemas += 1
        if modo == 'completo' and faltan:
            aviso(f'Modo completo: se omitirán {len(faltan)} modelo(s) sin base')
    except Exception as e:
        aviso(f'No se pudieron revisar las bases ({type(e).__name__}: {e})')
    try:
        from extractores_clasicos import BASE_CLASICA
        (ok if os.path.exists(BASE_CLASICA) else aviso)('Base de extractores clásicos (HOG/LBP/HSV): ' +
                                                         ('lista' if os.path.exists(BASE_CLASICA) else
                                                          f'FALTA -> python {R}/extractores_clasicos.py --generar-base'))
    except Exception:
        pass
    print('  Variables que se usarán:')
    print(f'     RTMP_URL={args.url}   REID_MODEL={args.modelo}   PRUEBAS_MODO={modo}')
    return problemas


def consolidar():
    try:
        from config import DATA_DIR
    except Exception:
        DATA_DIR = os.path.join(APP_DIR, 'data')
    base = os.path.join(DATA_DIR, 'pruebas_tiempo_real')
    dirs = sorted(os.path.join(base, d) for d in os.listdir(base)
                  if os.path.exists(os.path.join(base, d, 'sistema.json'))) if os.path.isdir(base) else []
    if not dirs:
        print(f'❌ No hay sesiones en {base}')
        return
    print(f'📚 Consolidando {len(dirs)} sesión(es):\n   ' + '\n   '.join(os.path.basename(d) for d in dirs))
    from analisis_vuelo import analizar_sesiones
    analizar_sesiones(dirs, os.path.join(base, 'consolidado'))


def _base_sesiones():
    try:
        from config import DATA_DIR
    except Exception:
        DATA_DIR = os.path.join(APP_DIR, 'data')
    return os.path.join(DATA_DIR, 'pruebas_tiempo_real')


def analizar(carpeta):
    """Analiza (o re-analiza) una sesión. 'ultima' = la más reciente."""
    base = _base_sesiones()
    if carpeta in (None, '', 'ultima'):
        cand = sorted(d for d in os.listdir(base) if os.path.exists(os.path.join(base, d, 'sistema.json'))) if os.path.isdir(base) else []
        if not cand:
            print(f'❌ No hay sesiones en {base}')
            return
        carpeta = os.path.join(base, cand[-1])
    elif not os.path.isdir(carpeta):
        carpeta = os.path.join(base, carpeta)
    print(f'📊 Analizando: {carpeta}')
    from analisis_vuelo import analizar_sesiones
    analizar_sesiones([carpeta], os.path.join(carpeta, 'analisis'))


def main():
    ap = argparse.ArgumentParser(description='Lanza una sesión de pruebas con el dron')
    ap.add_argument('--modo', default='completo', choices=['eficiencia', 'completo'])
    ap.add_argument('--modelo', default=os.environ.get('REID_MODEL', 'osnet_x1_0'),
                    help='modelo principal del sistema: osnet_x1_0|repvit|efficientformerv2|fastvit|swiftformer')
    ap.add_argument('--sesion', default='vuelo')
    ap.add_argument('--altura', type=float, default=10.0, help='altura inicial del dron en metros (+/- en vivo)')
    ap.add_argument('--angulo', default=ANGULOS[0], choices=ANGULOS)
    ap.add_argument('--grabar', action='store_true', help='graba el video original del vuelo (para pruebas offline de detección/MOT)')
    ap.add_argument('--url', default=os.environ.get('RTMP_URL', URL_POR_DEFECTO))
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--fps-stream', type=float, default=30.0, help='FPS nominal del stream (para contar frames tardíos)')
    ap.add_argument('--esperar-stream', type=int, default=0, help='segundos a esperar a que el dron empiece a transmitir')
    ap.add_argument('--solo-verificar', action='store_true')
    ap.add_argument('--consolidar', action='store_true')
    ap.add_argument('--analizar', nargs='?', const='ultima', default=None, metavar='CARPETA',
                    help='re-analiza una sesión guardada (sin argumento: la última)')
    ap.add_argument('--video', default=None, help='REPETIR un vuelo grabado (video_origen.mp4) en vez de usar el dron')
    ap.add_argument('--etiquetas', default=None, help='CSV  desde_s,hasta_s,persona  para etiquetar solo al repetir el video')
    ap.add_argument('--fps-video', type=float, default=None, help='FPS del video repetido (para convertir segundos a frames)')
    ap.add_argument('--forzar', action='store_true', help='lanza aunque la verificación detecte problemas')
    a = ap.parse_args()
    if a.consolidar:
        return consolidar()
    if a.analizar is not None:
        return analizar(a.analizar)
    problemas = verificar(a, a.modo)
    if a.solo_verificar:
        print('\n' + ('✅ Todo listo' if not problemas else f'❌ {problemas} problema(s) por resolver'))
        return
    if problemas and not a.forzar:
        print(f'\n❌ {problemas} problema(s). Resuélvelos o usa --forzar.')
        sys.exit(1)
    if a.video:
        a.url = a.video
    env = dict(os.environ, PRUEBAS_MODO=a.modo, REID_MODEL=a.modelo, RTMP_URL=a.url, PRUEBAS_SESION=a.sesion,
               PRUEBAS_ALTURA_M=str(a.altura), PRUEBAS_ANGULO=a.angulo, PRUEBAS_FPS_STREAM=str(a.fps_stream),
               PRUEBAS_GRABAR='1' if a.grabar else '0')
    if a.video:
        env['PRUEBAS_FPS_STREAM'] = str(a.fps_video or a.fps_stream)
        env['PRUEBAS_GRABAR'] = '0'
        if a.etiquetas:
            env['PRUEBAS_ETIQUETAS_CSV'] = os.path.abspath(a.etiquetas)
            env['PRUEBAS_ETIQUETAS_BASE'] = 'frames'
    print('\n🚀 Lanzando deteccion_reid_dron_contador.py ...')
    print('   Teclas: clic=elegir persona | 1-9/0=asignar persona | u=desconocido | x=borrar | l=modo sesión | '
          '+/-=altura | a=ángulo | o=oclusión | g=grabar | t=panel | s=guardar | q=salir (guarda y analiza)\n')
    subprocess.run([sys.executable, 'deteccion_reid_dron_contador.py'], cwd=APP_DIR, env=env)
    print(f'\n💡 Para juntar varias sesiones (alturas/ángulos distintos):  python {R}/lanzar_sesion.py --consolidar')


if __name__ == '__main__':
    main()