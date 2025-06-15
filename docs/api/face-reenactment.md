# Face Reenactment API

## FaceReenactor

Factory class for creating face reenactment model instances.

### Methods

```python
create(model, **kwargs) -> BaseFaceReenactor
```

Creates a face reenactor instance of the specified type.

**Parameters:**
- `model`: The type of reenactor to create ("tps")

**Returns:**
- A BaseFaceReenactor instance

```python
list_available_models() -> List[str]
```

**Returns:**
- List of supported model names

### BaseFaceReenactor.reenact_from_video()

```python
def reenact_from_video(
    source_path: str,
    driving_video_path: str,
    output_path: Optional[str] = None,
    save_comparison: Optional[bool] = False, 
    resize_to_image_resolution: Optional[bool] = False
) -> str
```

**Parameters:**  
- `source_path`: Path to the source image (face to be animated)  
- `driving_video_path`: Path to the driving video (facial motion to transfer)  
- `output_path`: Optional path to save the output video  
- `save_comparison`: Whether to save the comparison video  
- `resize_to_image_resolution`: Whether to resize the output to image resolution  

**Returns:**
- Path to the generated output video

## Available Models

- `tps` (Thin Plate Spline)