import cv2

from ..mediapipe import MediaPipeFaceDetector

image = cv2.imread("demo_images/1.jpg")

mediapipe_detector = MediaPipeFaceDetector()
faces = mediapipe_detector.detect(image)
faces, annotated_image = mediapipe_detector.detect_with_landmarks(image)
cv2.imwrite("output_images/mediapipe_output.png", annotated_image)
