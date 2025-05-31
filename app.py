"""
Gradio application for Face Detection, Face Reenactment, and Deepfake Detection.

This application provides a web interface for:
1. Face Detection using multiple models (BlazeFace, MediaPipe, UltraLight)
2. Face Reenactment using TPS model
3. Deepfake Detection using multiple models in a pipeline
"""

import os
import shutil
import tempfile
from typing import List, Optional, Tuple

import gradio as gr
import pandas as pd

from mukh.face_detection import FaceDetector
from mukh.pipelines.deepfake_detection import DeepfakeDetectionPipeline
from mukh.reenactment import FaceReenactor


class GradioApp:
    """Main Gradio application class for face processing tasks."""

    def __init__(self):
        """Initialize the Gradio application with default models."""
        # Initialize models
        self.face_detector = None
        self.face_reenactor = None
        self.deepfake_pipeline = None

        # Create output directory
        self.output_dir = "gradio_output"
        os.makedirs(self.output_dir, exist_ok=True)

    def detect_faces(
        self,
        image_path: str,
        detection_model: str,
        save_csv: bool = True,
        save_annotated: bool = True,
    ) -> Tuple[str, str, str]:
        """Detect faces in an image using the specified model.

        Args:
            image_path: Path to the input image
            detection_model: Model to use for detection
            save_csv: Whether to save CSV results
            save_annotated: Whether to save annotated image

        Returns:
            Tuple of (annotated_image_path, csv_path, results_text)
        """
        try:
            # Create detector
            detector = FaceDetector.create(detection_model)

            # Create output paths
            output_folder = os.path.join(
                self.output_dir, "face_detection", detection_model
            )
            csv_path = os.path.join(output_folder, "detections.csv")

            # Detect faces
            detections = detector.detect(
                image_path=image_path,
                save_csv=save_csv,
                csv_path=csv_path,
                save_annotated=save_annotated,
                output_folder=output_folder,
            )

            # Find annotated image
            annotated_image_path = None
            if save_annotated:
                image_name = os.path.basename(image_path)
                name, ext = os.path.splitext(image_name)

                # Try different possible naming patterns
                possible_names = [
                    f"{name}_annotated{ext}",
                    f"{name}_detected{ext}",
                    f"{name}_detection{ext}",
                ]

                for possible_name in possible_names:
                    potential_path = os.path.join(output_folder, possible_name)
                    if os.path.exists(potential_path):
                        annotated_image_path = potential_path
                        break

            # Create results text
            results_text = f"Found {len(detections)} face(s)\n\n"
            for i, detection in enumerate(detections):
                results_text += f"Face {i+1}:\n"
                results_text += f"  Confidence: {detection.bbox.confidence:.3f}\n"
                results_text += f"  Bounding Box: ({detection.bbox.x1}, {detection.bbox.y1}, {detection.bbox.x2}, {detection.bbox.y2})\n\n"

            return (
                (
                    annotated_image_path
                    if annotated_image_path and os.path.exists(annotated_image_path)
                    else None
                ),
                csv_path if save_csv and os.path.exists(csv_path) else None,
                results_text,
            )

        except Exception as e:
            return None, None, f"Error: {str(e)}"

    def reenact_face(
        self,
        source_image: str,
        driving_video: str,
        reenactor_model: str = "tps",
        save_comparison: bool = True,
        resize_to_image_resolution: bool = False,
    ) -> Tuple[str, str]:
        """Reenact face from source image using driving video.

        Args:
            source_image: Path to source image
            driving_video: Path to driving video
            reenactor_model: Model to use for reenactment
            save_comparison: Whether to save comparison video
            resize_to_image_resolution: Whether to resize to image resolution

        Returns:
            Tuple of (reenacted_video_path, comparison_video_path)
        """
        try:
            # Create reenactor
            reenactor = FaceReenactor.create(reenactor_model)

            # Create output path
            output_folder = os.path.join(
                self.output_dir, "face_reenactment", reenactor_model
            )
            os.makedirs(output_folder, exist_ok=True)

            source_name = os.path.splitext(os.path.basename(source_image))[0]
            output_path = os.path.join(output_folder, f"{source_name}_reenacted")

            # Perform reenactment
            result_path = reenactor.reenact_from_video(
                source_path=source_image,
                driving_video_path=driving_video,
                output_path=output_path,
                save_comparison=save_comparison,
                resize_to_image_resolution=resize_to_image_resolution,
            )

            # Find comparison video if it was created
            comparison_path = None
            if save_comparison:
                # Extract driving video name for pattern matching
                driving_name = os.path.splitext(os.path.basename(driving_video))[0]

                # The comparison video should be in the same directory as output_path
                # which is the directory passed to the reenactor
                search_dir = output_path  # This is the directory passed to reenactor

                # Look for comparison video with the exact naming pattern used by TPS
                comparison_patterns = [
                    f"comparison_{source_name}_by_{driving_name}.mp4",  # Exact pattern from TPS
                    f"comparison_{source_name}_by_*.mp4",  # Pattern with wildcard
                    f"comparison_*.mp4",  # Generic comparison pattern
                ]

                import glob

                for pattern in comparison_patterns:
                    comparison_files = glob.glob(os.path.join(search_dir, pattern))
                    if comparison_files:
                        comparison_path = comparison_files[0]  # Take the first match
                        break

                # If still not found, check the parent directory of result_path
                if not comparison_path:
                    result_dir = os.path.dirname(result_path)
                    for pattern in comparison_patterns:
                        comparison_files = glob.glob(os.path.join(result_dir, pattern))
                        if comparison_files:
                            comparison_path = comparison_files[0]
                            break

            return result_path, comparison_path

        except Exception as e:
            return None, None

    def detect_deepfakes(
        self,
        media_file: str,
        models_to_use: List[str],
        confidence_threshold: float = 0.5,
        num_frames: int = 11,
        save_csv: bool = True,
        save_annotated: bool = True,
        save_individual_results: bool = True,
    ) -> Tuple[str, str]:
        """Detect deepfakes using multiple models in a pipeline.

        Args:
            media_file: Path to media file (image or video)
            models_to_use: List of models to use in pipeline
            confidence_threshold: Confidence threshold for final decision
            num_frames: Number of frames to analyze for videos
            save_csv: Whether to save CSV results
            save_annotated: Whether to save annotated media
            save_individual_results: Whether to save individual model results

        Returns:
            Tuple of (annotated_media_path, results_text)
        """
        try:
            # Configure models based on selection
            model_configs = []
            model_weights = {}

            if "resnet_inception" in models_to_use:
                model_configs.append(
                    {
                        "name": "resnet_inception",
                        "confidence_threshold": 0.4,
                    }
                )
                model_weights["resnet_inception"] = 0.4

            if "resnext" in models_to_use:
                model_configs.append(
                    {
                        "name": "resnext",
                        "model_variant": "resnext",
                        "confidence_threshold": 0.5,
                    }
                )
                model_weights["resnext"] = 0.3

            if "efficientnet" in models_to_use:
                model_configs.append(
                    {
                        "name": "efficientnet",
                        "net_model": "EfficientNetB4",
                        "confidence_threshold": 0.6,
                    }
                )
                model_weights["efficientnet"] = 0.3

            # Normalize weights
            total_weight = sum(model_weights.values())
            if total_weight > 0:
                model_weights = {k: v / total_weight for k, v in model_weights.items()}

            # Create pipeline
            pipeline = DeepfakeDetectionPipeline(
                model_configs=model_configs,
                model_weights=model_weights,
                confidence_threshold=confidence_threshold,
            )

            # Create output paths
            output_folder = os.path.join(self.output_dir, "deepfake_detection")

            # Detect deepfakes
            result = pipeline.detect(
                media_path=media_file,
                save_csv=save_csv,
                save_annotated=save_annotated,
                output_folder=output_folder,
                num_frames=num_frames,
                save_individual_results=save_individual_results,
            )

            # Find annotated media
            annotated_media_path = None
            if save_annotated:
                media_name = os.path.basename(media_file)
                name, ext = os.path.splitext(media_name)

                # Check for video or image
                if ext.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                    annotated_media_path = os.path.join(
                        output_folder, f"{name}_pipeline_annotated.mp4"
                    )
                else:
                    annotated_media_path = os.path.join(
                        output_folder, f"{name}_pipeline_annotated{ext}"
                    )

            # Create results text
            if isinstance(result, list):  # Video results
                deepfake_count = sum(1 for r in result if r.is_deepfake)
                total_frames = len(result)
                avg_confidence = sum(r.confidence for r in result) / len(result)

                results_text = f"Video Analysis Results:\n"
                results_text += f"Total frames analyzed: {total_frames}\n"
                results_text += f"Deepfake frames detected: {deepfake_count}\n"
                results_text += (
                    f"Deepfake percentage: {(deepfake_count/total_frames)*100:.1f}%\n"
                )
                results_text += f"Average confidence: {avg_confidence:.3f}\n"

                # Overall verdict
                if deepfake_count > total_frames * 0.5:
                    results_text += f"\n🚨 VERDICT: DEEPFAKE"
                else:
                    results_text += f"\n✅ VERDICT: REAL"

            else:  # Single image result
                status = "DEEPFAKE" if result.is_deepfake else "REAL"
                results_text = f"Image Analysis Results:\n"
                results_text += f"Status: {status}\n"
                results_text += f"Confidence: {result.confidence:.3f}\n"
                results_text += f"Model: {result.model_name}\n"

            return (
                (
                    annotated_media_path
                    if annotated_media_path and os.path.exists(annotated_media_path)
                    else None
                ),
                results_text,
            )

        except Exception as e:
            return None, f"Error: {str(e)}"

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface with all three tasks."""

        with gr.Blocks(
            title="Mukh: Face Processing Suite", theme=gr.themes.Soft()
        ) as interface:
            gr.Markdown(
                """
            # 🎭 Mukh: Face Processing Suite
            
            A comprehensive toolkit for face detection, reenactment, and deepfake detection.
            
            ---
            ### 📝 Instructions:
            
            **Face Detection**: Upload an image and select a detection model to identify faces.
            
            **Face Reenactment**: Upload a source image and driving video to animate the face.
            
            **Deepfake Detection**: Upload an image or video to detect if it contains deepfakes using multiple AI models.
            
            All results are saved to the `gradio_output` directory.
            ---
            """
            )

            with gr.Tabs():
                # Face Detection Tab
                with gr.TabItem("👤 Face Detection"):
                    gr.Markdown(
                        "### Detect faces in images using state-of-the-art models"
                    )

                    with gr.Row():
                        with gr.Column():
                            face_input_image = gr.Image(
                                type="filepath", label="Upload Image"
                            )
                            face_detection_model = gr.Dropdown(
                                choices=["blazeface", "mediapipe", "ultralight"],
                                value="mediapipe",
                                label="Detection Model",
                            )
                            face_save_csv = gr.Checkbox(
                                value=True, label="Save CSV Results"
                            )
                            face_save_annotated = gr.Checkbox(
                                value=True, label="Save Annotated Image"
                            )
                            face_detect_btn = gr.Button(
                                "🔍 Detect Faces", variant="primary"
                            )

                        with gr.Column():
                            face_output_image = gr.Image(label="Annotated Image")
                            face_results_text = gr.Textbox(
                                label="Detection Results", lines=10
                            )
                            face_csv_file = gr.File(label="CSV Results", visible=False)

                    face_detect_btn.click(
                        fn=self.detect_faces,
                        inputs=[
                            face_input_image,
                            face_detection_model,
                            face_save_csv,
                            face_save_annotated,
                        ],
                        outputs=[face_output_image, face_csv_file, face_results_text],
                    )

                # Face Reenactment Tab
                with gr.TabItem("🎬 Face Reenactment"):
                    gr.Markdown("### Animate faces using driving videos")

                    with gr.Row():
                        with gr.Column():
                            reenact_source_image = gr.Image(
                                type="filepath", label="Source Image"
                            )
                            reenact_driving_video = gr.Video(label="Driving Video")
                            reenact_model = gr.Dropdown(
                                choices=["tps"], value="tps", label="Reenactment Model"
                            )
                            reenact_save_comparison = gr.Checkbox(
                                value=True, label="Save Comparison Video"
                            )
                            reenact_resize_to_image = gr.Checkbox(
                                value=False, label="Resize to Image Resolution"
                            )
                            reenact_btn = gr.Button(
                                "🎬 Generate Reenactment", variant="primary"
                            )

                        with gr.Column():
                            reenact_output_video = gr.Video(
                                label="Reenacted Video", visible=False
                            )
                            reenact_comparison_video = gr.Video(
                                label="Comparison Video",
                            )

                    reenact_btn.click(
                        fn=self.reenact_face,
                        inputs=[
                            reenact_source_image,
                            reenact_driving_video,
                            reenact_model,
                            reenact_save_comparison,
                            reenact_resize_to_image,
                        ],
                        outputs=[reenact_output_video, reenact_comparison_video],
                    )

                # Deepfake Detection Tab
                with gr.TabItem("🕵️ Deepfake Detection"):
                    gr.Markdown("### Detect deepfakes using multiple AI models")

                    with gr.Row():
                        with gr.Column():
                            deepfake_input_media = gr.File(
                                label="Upload Image or Video",
                                file_types=["image", "video"],
                            )
                            deepfake_models = gr.CheckboxGroup(
                                choices=["resnet_inception", "resnext", "efficientnet"],
                                value=["resnet_inception", "resnext", "efficientnet"],
                                label="Models to Use",
                            )
                            deepfake_confidence_threshold = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.5,
                                step=0.1,
                                label="Confidence Threshold",
                            )
                            deepfake_num_frames = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=11,
                                step=1,
                                label="Number of Frames (for videos)",
                            )
                            deepfake_save_csv = gr.Checkbox(
                                value=True, label="Save CSV Results"
                            )
                            deepfake_save_annotated = gr.Checkbox(
                                value=True, label="Save Annotated Media"
                            )
                            deepfake_save_individual = gr.Checkbox(
                                value=True, label="Save Individual Model Results"
                            )
                            deepfake_detect_btn = gr.Button(
                                "🕵️ Detect Deepfakes", variant="primary"
                            )

                        with gr.Column():
                            deepfake_output_media = gr.File(label="Annotated Media")
                            deepfake_results_text = gr.Textbox(
                                label="Detection Results", lines=10
                            )

                    deepfake_detect_btn.click(
                        fn=self.detect_deepfakes,
                        inputs=[
                            deepfake_input_media,
                            deepfake_models,
                            deepfake_confidence_threshold,
                            deepfake_num_frames,
                            deepfake_save_csv,
                            deepfake_save_annotated,
                            deepfake_save_individual,
                        ],
                        outputs=[deepfake_output_media, deepfake_results_text],
                    )

        return interface

    def launch(self, **kwargs) -> None:
        """Launch the Gradio interface.

        Args:
            **kwargs: Additional arguments to pass to gr.Interface.launch()
        """
        interface = self.create_interface()
        interface.launch(**kwargs)


def main():
    """Main function to run the Gradio application."""
    app = GradioApp()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
    )


if __name__ == "__main__":
    main()
