import numpy as np
import torch
import cv2
from .blazeface import BlazeFace

def init_blazeface(weights_path="mukh/detection/models/blazeface/blazeface.pth", anchors_path="mukh/detection/models/blazeface/anchors.npy"):
    """Initialize the BlazeFace model"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = BlazeFace().to(device)
    net.load_weights(weights_path)
    net.load_anchors(anchors_path)
    
    # Set default thresholds
    net.min_score_thresh = 0.75
    net.min_suppression_threshold = 0.3
    
    return net, device

def process_image(image_path, net, device):
    """Process a single image and return detections"""
    # Read and convert image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Make prediction
    detections = net.predict_on_image(img)
    
    # Convert detections to numpy if on GPU
    if device.type == "cuda":
        detections = detections.cpu().numpy()
    else:
        detections = detections.numpy()
        
    return img, detections

def draw_detections(img, detections):
    """Draw bounding boxes and keypoints on the image"""
    # Create a copy of the image for drawing
    img_draw = img.copy()
    
    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)
    
    # Draw each detection
    for detection in detections:
        # Get coordinates
        ymin, xmin = detection[0] * img.shape[0], detection[1] * img.shape[1]
        ymax, xmax = detection[2] * img.shape[0], detection[3] * img.shape[1]
        
        # Draw bounding box
        cv2.rectangle(img_draw, 
                     (int(xmin), int(ymin)), 
                     (int(xmax), int(ymax)),
                     (0, 255, 0), 2)
        
        # Draw keypoints
        for k in range(6):
            kp_x = int(detection[4 + k*2] * img.shape[1])
            kp_y = int(detection[4 + k*2 + 1] * img.shape[0])
            cv2.circle(img_draw, (kp_x, kp_y), 2, (255, 0, 0), 2)
    
    return img_draw

def main():
    # Initialize model
    net, device = init_blazeface()
    print(f"Running on device: {device}")
    
    # Process image
    image_path = "mukh/detection/models/blazeface/1face.png"  # Change this to your image path
    img, detections = process_image(image_path, net, device)
    
    # Draw and display results
    num_faces = detections.shape[0]
    print(f"Found {num_faces} faces")
    
    result_img = draw_detections(img, detections)
    
    # Save result image
    output_path = "mukh/detection/models/blazeface/output.png"  # Change this to your desired output path
    cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    print(f"Saved detection result to {output_path}")

if __name__ == "__main__":
    main()
