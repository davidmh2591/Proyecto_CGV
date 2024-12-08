import mediapipe as mp  # Importamos MediaPipe
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions  # Nos ayuda a las opciones de configuración
import cv2  # Importamos OpenCV
import time

detection_result_list = []

# Función callback para procesar los resultados de detección
def detection_callback(result, output_image, timestamp_ms):
    detection_result_list.append(result)

# Especificar la configuración del detector de objetos
options = vision.ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=r"C:\Users\HP 14\Desktop\David UTH\Computacion Grafica y Visual\1er Parcial- periodo 3\Parcial3\ReconocimientoObjetos\Modelos\efficientdet_lite0float32.tflite"),
    max_results=5,
    score_threshold=0.15,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=detection_callback
)

detector = vision.ObjectDetector.create_from_options(options)

# Leer el video de entrada
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertimos el frame a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Detección de objetos sobre el frame
    detection_result = detector.detect_async(mp_image, time.time_ns() // 1_000_000)

    # Limpiar el texto del frame actual antes de dibujar nuevos resultados
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Vuelve a convertir la imagen a BGR después de la detección

    # Procesamos los resultados de detección
    if detection_result_list:
        for detection in detection_result_list[0].detections:
            bbox = detection.bounding_box
            bbox_x, bbox_y, bbox_w, bbox_h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            category = detection.categories[0]
            score = category.score * 100
            category_name = category.category_name

            # Dibujamos las cajas y etiquetas en el frame
            cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (100, 255, 0), 2)
            cv2.putText(frame, f"{category_name}: {score:.2f}%", (bbox_x, bbox_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Mostrar el frame actualizado
        cv2.imshow("Video", frame)
        detection_result_list.clear()  # Limpiamos la lista de resultados para evitar dibujar elementos de más

    # Salimos si se presiona 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
