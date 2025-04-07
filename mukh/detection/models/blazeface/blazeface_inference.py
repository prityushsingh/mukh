import cv2

from ..blazeface import BlazeFaceDetector

image = cv2.imread("demo_images/1.jpg")

blazeface_detector = BlazeFaceDetector()
faces = blazeface_detector.detect(image)
faces, annotated_image = blazeface_detector.detect_with_landmarks(image)
cv2.imwrite("output_images/blazeface_output.png", annotated_image)
