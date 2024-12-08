import mediapipe as mp  # Importamos MediaPipe
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions  # Nos ayuda con las opciones de configuración
import cv2  # Importamos OpenCV

# Opciones de configuración
options = vision.ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=r"C:\Users\HP 14\Desktop\David UTH\Computacion Grafica y Visual\1er Parcial- periodo 3\Parcial3\ReconocimientoObjetos\Modelos\efficientdet_lite0float32.tflite"),
    max_results=5,
    score_threshold=0.2,
    running_mode=vision.RunningMode.IMAGE)
detector = vision.ObjectDetector.create_from_options(options)

# Leer la imagen de entrada
image = cv2.imread("./Data/image_01.jpg")

# Redimensionar la imagen a un tamaño específico (por ejemplo, 800x600)
image_resized = cv2.resize(image, (800, 600))

# Convertir la imagen a formato RGB
image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
image_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

# Detección de objetos sobre la imagen
detection_result = detector.detect(image_rgb)

for detection in detection_result.detections:
    print(detection)
    # Bounding box
    bbox = detection.bounding_box
    bbox_x, bbox_y, bbox_w, bbox_h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

    # Accedemos a los resultados de Category y Score
    category = detection.categories[0]
    score = category.score * 100
    category_name = category.category_name

    # Dibujar bounding box
    cv2.rectangle(image_resized, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y - 30), (100, 255, 0), -1)
    cv2.rectangle(image_resized, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (100, 255, 0), 2)
    cv2.putText(image_resized, f"{category_name}: {score:.2f}%", (bbox_x + 5, bbox_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Mostrar la imagen redimensionada con los resultados
cv2.imshow("Image", image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
