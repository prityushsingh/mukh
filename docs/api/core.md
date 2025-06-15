# Core API

## Data Types

### BoundingBox

```python
@dataclass
class BoundingBox:
    x1: float
    y1: float  
    x2: float
    y2: float
    confidence: float
    
    @property
    def width(self) -> float
    
    @property  
    def height(self) -> float
```

### FaceDetection

```python
@dataclass
class FaceDetection:
    bbox: BoundingBox
    landmarks: Optional[np.ndarray] = None
```

### DeepfakeDetection

```python
@dataclass
class DeepfakeDetection:
    frame_number: int
    is_deepfake: bool
    confidence: float
    model_name: str
```

## Factory Pattern

All modules use factory methods for model creation:

```python
# Face Detection
detector = FaceDetector.create("mediapipe")  # "blazeface", "mediapipe", "ultralight"

# Face Reenactment
reenactor = FaceReenactor.create("tps")  # "tps"

# Deepfake Detection  
detector = DeepfakeDetector("efficientnet")  # "efficientnet", "resnet_inception"
```