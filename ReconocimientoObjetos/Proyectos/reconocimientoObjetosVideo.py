import mediapipe as mp  # Importamos MediaPipe
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions  # Nos ayuda a las opciones de configuración
import cv2  # Importamos OpenCV

# Opciones de configuración
options = vision.ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=r"C:\Users\HP 14\Desktop\David UTH\Computacion Grafica y Visual\1er Parcial- periodo 3\Parcial3\ReconocimientoObjetos\Modelos\efficientdet_lite0float32.tflite"),
    max_results=5,
    score_threshold=0.2,
    running_mode=vision.RunningMode.VIDEO,
)
detector = vision.ObjectDetector.create_from_options(options)

# Leer el video de entrada
cap = cv2.VideoCapture("./Data/video_01.mp4")

if not cap.isOpened():
    print("Error al abrir el archivo de video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertimos el frame a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Calculo de la marca temporal del frame actual (en milisegundos)
    frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Índice actual del frame
    frame_timestamp_ms = int(1000 * frame_index / fps)

    # Detección de objetos en el cuadro actual
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    # Dibujamos los resultados
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        bbox_x, bbox_y, bbox_w, bbox_h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

        category = detection.categories[0]
        score = category.score * 100
        category_name = category.category_name

        # Dibujamos las cajas y etiquetas en el frame
        cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (100, 255, 0), 2)
        cv2.putText(frame, f"{category_name}: {score:.2f}%", (bbox_x, bbox_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Redimensionamos el frame para que se muestre más pequeño
    frame_resized = cv2.resize(frame, (640, 480))  # Cambia las dimensiones según lo necesario

    # Mostramos el video
    cv2.imshow("Video", frame_resized)

    # Salimos si se presiona 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
