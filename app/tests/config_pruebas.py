"""
CONFIGURACIÓN DE LAS PRUEBAS EN TIEMPO REAL
===========================================
Edita este archivo y ejecuta simplemente:

        python deteccion_reid_dron_contador.py

y el monitor de pruebas se activa solo. Las variables de entorno (PRUEBAS_MODO, PRUEBAS_SESION,
PRUEBAS_ALTURA_M, ...) si existen, tienen prioridad sobre este archivo.
"""
MODO = 'completo'              # 'off' = sistema normal | 'eficiencia' = solo tiempos/memoria | 'completo' = todo + etiquetado en vivo
SESION = 'vuelo_10m'          # nombre de la carpeta de resultados (cámbialo en cada vuelo)
ALTURA_M = 10.0           # altura inicial del dron en metros (en vivo: teclas + y -)
ANGULO = 'frontal-oblicuo'  # frontal-oblicuo | lateral-oblicuo | posterior-oblicuo | cenital_parcial | cenital_extremo (tecla a)
GRABAR = True            # True = graba el video original del vuelo (para pruebas offline de detección/MOT)
FPS_STREAM = 30.0         # FPS nominal del stream del dron (para contar frames tardíos)
ALFAS = '0.5,0.7'         # pesos de la fusión modelo profundo + HOG/LBP/HSV
CROP_CADA = 3             # guarda 1 recorte cada N eventos para análisis cualitativo (0 = no guardar)
VIDEO_FPS = 25            # FPS del video grabado