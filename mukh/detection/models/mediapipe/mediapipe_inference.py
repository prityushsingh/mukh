import cv2
from .mediapipe import MediaPipeFaceDetector

# Initialize detector
detector = MediaPipeFaceDetector()

# Read image
image = cv2.imread('mukh/detection/models/mediapipe/demo_img.jpg')
print(image)

# Method 1: Get just the detections
faces = detector(image)

# Method 2: Get detections and visualized image
faces, annotated_image = detector(image, return_annotated=True)

# Display results

output_path = "mukh/detection/models/mediapipe/output.png"  # Change this to your desired output path
cv2.imwrite(output_path, annotated_image)
print(f"Saved detection result to {output_path}")