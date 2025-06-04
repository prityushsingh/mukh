# Pipeline Examples

This page demonstrates how to combine multiple Mukh features to create powerful processing pipelines for complex face analysis workflows.

## Face Analysis Pipeline

Combine face detection, landmark detection, and deepfake detection:

```python
"""
Comprehensive face analysis pipeline combining multiple Mukh features.
"""
from mukh.face_detection import FaceDetector
from mukh.deepfake_detection import DeepfakeDetector
import os
import pandas as pd

class FaceAnalysisPipeline:
    """
    A comprehensive pipeline for face analysis.
    """
    
    def __init__(self, detection_model="mediapipe", deepfake_model="resnet_inception"):
        """
        Initialize the face analysis pipeline.
        
        Args:
            detection_model (str): Model for face detection
            deepfake_model (str): Model for deepfake detection
        """
        self.face_detector = FaceDetector.create(detection_model)
        self.deepfake_detector = DeepfakeDetector(
            model_name=deepfake_model,
            confidence_threshold=0.5
        )
        self.detection_model = detection_model
        self.deepfake_model = deepfake_model
    
    def analyze_image(self, image_path, output_folder=None):
        """
        Perform comprehensive analysis of a single image.
        
        Args:
            image_path (str): Path to the image file
            output_folder (str): Optional output folder for results
        
        Returns:
            dict: Comprehensive analysis results
        """
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
        
        # Step 1: Face Detection
        print(f"Detecting faces in {image_path}...")
        face_detections = self.face_detector.detect(
            image_path=image_path,
            save_csv=output_folder is not None,
            csv_path=os.path.join(output_folder, "face_detections.csv") if output_folder else None,
            save_annotated=output_folder is not None,
            output_folder=output_folder
        )
        
        # Step 2: Deepfake Detection
        print(f"Analyzing for deepfake content...")
        deepfake_result = self.deepfake_detector.detect(
            media_path=image_path,
            save_csv=output_folder is not None,
            csv_path=os.path.join(output_folder, "deepfake_analysis.csv") if output_folder else None,
            save_annotated=output_folder is not None,
            output_folder=output_folder
        )
        
        # Compile results
        analysis_result = {
            'image_path': image_path,
            'face_detection': {
                'num_faces': len(face_detections),
                'faces': [
                    {
                        'bbox': det.bbox,
                        'confidence': det.confidence,
                        'landmarks': det.landmarks
                    }
                    for det in face_detections
                ]
            },
            'deepfake_analysis': {
                'is_deepfake': deepfake_result.is_deepfake,
                'confidence': deepfake_result.confidence
            },
            'summary': {
                'has_faces': len(face_detections) > 0,
                'multiple_faces': len(face_detections) > 1,
                'likely_deepfake': deepfake_result.is_deepfake,
                'risk_score': self._calculate_risk_score(face_detections, deepfake_result)
            }
        }
        
        return analysis_result
    
    def analyze_batch(self, input_folder, output_folder):
        """
        Perform batch analysis of all images in a folder.
        
        Args:
            input_folder (str): Folder containing images
            output_folder (str): Output folder for results
        
        Returns:
            list: List of analysis results for all images
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # Supported image formats
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Find all images
        image_files = [
            f for f in os.listdir(input_folder)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        
        print(f"Found {len(image_files)} images to analyze")
        
        batch_results = []
        
        for i, filename in enumerate(image_files, 1):
            print(f"\nProcessing {i}/{len(image_files)}: {filename}")
            
            image_path = os.path.join(input_folder, filename)
            image_output_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
            
            try:
                result = self.analyze_image(image_path, image_output_folder)
                result['status'] = 'success'
                batch_results.append(result)
                
                # Print summary for this image
                summary = result['summary']
                print(f"  Faces: {result['face_detection']['num_faces']}")
                print(f"  Deepfake: {'Yes' if summary['likely_deepfake'] else 'No'} "
                      f"(conf: {result['deepfake_analysis']['confidence']:.3f})")
                print(f"  Risk Score: {summary['risk_score']:.2f}")
                
            except Exception as e:
                print(f"  Error processing {filename}: {str(e)}")
                batch_results.append({
                    'image_path': image_path,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Save batch summary
        self._save_batch_summary(batch_results, output_folder)
        
        return batch_results
    
    def _calculate_risk_score(self, face_detections, deepfake_result):
        """
        Calculate a risk score based on analysis results.
        
        Args:
            face_detections: List of face detections
            deepfake_result: Deepfake detection result
        
        Returns:
            float: Risk score between 0 and 1
        """
        risk_score = 0.0
        
        # Base risk from deepfake detection
        if deepfake_result.is_deepfake:
            risk_score += deepfake_result.confidence * 0.7
        
        # Risk from number of faces
        num_faces = len(face_detections)
        if num_faces == 0:
            risk_score += 0.1  # No faces detected
        elif num_faces > 3:
            risk_score += 0.2  # Many faces might indicate group manipulation
        
        # Risk from face detection confidence
        if face_detections:
            avg_face_confidence = sum(det.confidence for det in face_detections) / len(face_detections)
            if avg_face_confidence < 0.5:
                risk_score += 0.1  # Low confidence faces
        
        return min(1.0, risk_score)
    
    def _save_batch_summary(self, batch_results, output_folder):
        """
        Save a summary of batch processing results.
        """
        successful_results = [r for r in batch_results if r.get('status') == 'success']
        
        if not successful_results:
            return
        
        # Create summary DataFrame
        summary_data = []
        for result in successful_results:
            summary_data.append({
                'image_path': os.path.basename(result['image_path']),
                'num_faces': result['face_detection']['num_faces'],
                'has_faces': result['summary']['has_faces'],
                'multiple_faces': result['summary']['multiple_faces'],
                'is_deepfake': result['deepfake_analysis']['is_deepfake'],
                'deepfake_confidence': result['deepfake_analysis']['confidence'],
                'risk_score': result['summary']['risk_score']
            })
        
        df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_folder, "batch_analysis_summary.csv")
        df.to_csv(summary_path, index=False)
        
        # Print batch statistics
        print(f"\n--- Batch Analysis Summary ---")
        print(f"Total images processed: {len(successful_results)}")
        print(f"Images with faces: {df['has_faces'].sum()}")
        print(f"Images with multiple faces: {df['multiple_faces'].sum()}")
        print(f"Images detected as deepfakes: {df['is_deepfake'].sum()}")
        print(f"Average risk score: {df['risk_score'].mean():.3f}")
        print(f"High risk images (score > 0.7): {(df['risk_score'] > 0.7).sum()}")
        print(f"Summary saved to: {summary_path}")

# Usage example
pipeline = FaceAnalysisPipeline()

# Single image analysis
result = pipeline.analyze_image(
    image_path="assets/images/test_image.jpg",
    output_folder="output/single_analysis"
)

print("Single Image Analysis Result:")
print(f"  Faces detected: {result['face_detection']['num_faces']}")
print(f"  Deepfake: {result['summary']['likely_deepfake']}")
print(f"  Risk score: {result['summary']['risk_score']:.3f}")

# Batch analysis
batch_results = pipeline.analyze_batch(
    input_folder="assets/images",
    output_folder="output/batch_analysis"
)
```

## Content Verification Pipeline

Verify the authenticity of media content:

```python
"""
Content verification pipeline for detecting manipulated media.
"""
from mukh.face_detection import FaceDetector
from mukh.deepfake_detection import DeepfakeDetector
import cv2
import os
import json

class ContentVerificationPipeline:
    """
    Pipeline for comprehensive content verification.
    """
    
    def __init__(self):
        """Initialize the verification pipeline."""
        self.face_detector = FaceDetector.create("mediapipe")
        self.deepfake_detectors = {
            'resnet_inception': DeepfakeDetector("resnet_inception", 0.5),
            'resnext': DeepfakeDetector("resnext", 0.5),
            'efficientnet': DeepfakeDetector("efficientnet", 0.5)
        }
    
    def verify_image(self, image_path):
        """
        Verify the authenticity of an image.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            dict: Verification results
        """
        print(f"Verifying image: {image_path}")
        
        # Face detection
        face_detections = self.face_detector.detect(
            image_path=image_path,
            save_csv=False,
            save_annotated=False
        )
        
        # Multi-model deepfake detection
        deepfake_results = {}
        for model_name, detector in self.deepfake_detectors.items():
            result = detector.detect(
                media_path=image_path,
                save_csv=False,
                save_annotated=False
            )
            deepfake_results[model_name] = {
                'is_deepfake': result.is_deepfake,
                'confidence': result.confidence
            }
        
        # Calculate consensus
        deepfake_votes = sum(1 for r in deepfake_results.values() if r['is_deepfake'])
        total_models = len(deepfake_results)
        consensus_score = deepfake_votes / total_models
        
        # Calculate average confidence
        avg_confidence = sum(r['confidence'] for r in deepfake_results.values()) / total_models
        
        # Determine verification status
        if consensus_score >= 0.67:  # 2/3 majority
            verification_status = "LIKELY_MANIPULATED"
        elif consensus_score <= 0.33:  # 2/3 majority for real
            verification_status = "LIKELY_AUTHENTIC"
        else:
            verification_status = "UNCERTAIN"
        
        return {
            'image_path': image_path,
            'verification_status': verification_status,
            'face_analysis': {
                'faces_detected': len(face_detections),
                'faces': [{'bbox': det.bbox, 'confidence': det.confidence} for det in face_detections]
            },
            'deepfake_analysis': deepfake_results,
            'consensus': {
                'deepfake_votes': deepfake_votes,
                'total_models': total_models,
                'consensus_score': consensus_score,
                'average_confidence': avg_confidence
            },
            'confidence_level': self._calculate_confidence_level(consensus_score, avg_confidence)
        }
    
    def verify_video(self, video_path, num_frames=15):
        """
        Verify the authenticity of a video.
        
        Args:
            video_path (str): Path to the video file
            num_frames (int): Number of frames to analyze
        
        Returns:
            dict: Verification results
        """
        print(f"Verifying video: {video_path}")
        
        # Analyze video properties
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        # Multi-model deepfake detection on video
        frame_results = {}
        for model_name, detector in self.deepfake_detectors.items():
            detections = detector.detect(
                media_path=video_path,
                save_csv=False,
                save_annotated=False,
                num_frames=num_frames
            )
            
            deepfake_frames = sum(1 for det in detections if det.is_deepfake)
            avg_confidence = sum(det.confidence for det in detections) / len(detections)
            
            frame_results[model_name] = {
                'total_frames_analyzed': len(detections),
                'deepfake_frames': deepfake_frames,
                'deepfake_ratio': deepfake_frames / len(detections),
                'average_confidence': avg_confidence,
                'frame_results': [
                    {
                        'frame_idx': i,
                        'is_deepfake': det.is_deepfake,
                        'confidence': det.confidence
                    }
                    for i, det in enumerate(detections)
                ]
            }
        
        # Calculate consensus across models
        model_deepfake_ratios = [r['deepfake_ratio'] for r in frame_results.values()]
        avg_deepfake_ratio = sum(model_deepfake_ratios) / len(model_deepfake_ratios)
        
        # Determine verification status
        if avg_deepfake_ratio >= 0.6:
            verification_status = "LIKELY_MANIPULATED"
        elif avg_deepfake_ratio <= 0.3:
            verification_status = "LIKELY_AUTHENTIC"
        else:
            verification_status = "UNCERTAIN"
        
        return {
            'video_path': video_path,
            'verification_status': verification_status,
            'video_properties': {
                'total_frames': total_frames,
                'fps': fps,
                'duration': duration
            },
            'analysis_results': frame_results,
            'consensus': {
                'average_deepfake_ratio': avg_deepfake_ratio,
                'model_agreement': self._calculate_model_agreement(model_deepfake_ratios)
            },
            'confidence_level': self._calculate_confidence_level(avg_deepfake_ratio, 
                                                               sum(r['average_confidence'] for r in frame_results.values()) / len(frame_results))
        }
    
    def _calculate_confidence_level(self, consensus_score, avg_confidence):
        """Calculate overall confidence level in the verification."""
        # Combine consensus and confidence to determine overall confidence
        confidence_factor = min(abs(consensus_score - 0.5) * 2, 1.0)  # Higher when away from 0.5
        confidence_level = (confidence_factor + avg_confidence) / 2
        
        if confidence_level >= 0.8:
            return "HIGH"
        elif confidence_level >= 0.6:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_model_agreement(self, ratios):
        """Calculate agreement between different models."""
        if len(ratios) < 2:
            return 1.0
        
        # Calculate variance of ratios
        mean_ratio = sum(ratios) / len(ratios)
        variance = sum((r - mean_ratio) ** 2 for r in ratios) / len(ratios)
        
        # Convert variance to agreement score (lower variance = higher agreement)
        agreement = max(0, 1 - variance * 4)  # Scale factor to make it meaningful
        return agreement

# Usage example
verifier = ContentVerificationPipeline()

# Verify images
image_results = []
test_images = [
    "assets/images/real_photo.jpg",
    "assets/images/suspicious_photo.jpg",
    "assets/images/synthetic_face.jpg"
]

for image_path in test_images:
    if os.path.exists(image_path):
        result = verifier.verify_image(image_path)
        image_results.append(result)
        
        print(f"\nImage: {os.path.basename(image_path)}")
        print(f"Status: {result['verification_status']}")
        print(f"Confidence: {result['confidence_level']}")
        print(f"Faces detected: {result['face_analysis']['faces_detected']}")
        print(f"Consensus score: {result['consensus']['consensus_score']:.2f}")

# Verify videos
video_results = []
test_videos = [
    "assets/videos/authentic_video.mp4",
    "assets/videos/reenacted_video.mp4"
]

for video_path in test_videos:
    if os.path.exists(video_path):
        result = verifier.verify_video(video_path)
        video_results.append(result)
        
        print(f"\nVideo: {os.path.basename(video_path)}")
        print(f"Status: {result['verification_status']}")
        print(f"Confidence: {result['confidence_level']}")
        print(f"Duration: {result['video_properties']['duration']:.1f}s")
        print(f"Avg deepfake ratio: {result['consensus']['average_deepfake_ratio']:.2f}")

# Save detailed results
with open('output/verification_results.json', 'w') as f:
    json.dump({
        'image_results': image_results,
        'video_results': video_results
    }, f, indent=2, default=str)

print(f"\nDetailed results saved to: output/verification_results.json")
```

## Reenactment Quality Assessment Pipeline

Assess the quality of generated reenactment videos:

```python
"""
Pipeline for assessing reenactment quality and detecting artifacts.
"""
from mukh.face_detection import FaceDetector
from mukh.deepfake_detection import DeepfakeDetector
from mukh.reenactment import FaceReenactor
import cv2
import numpy as np
import os

class ReenactmentQualityPipeline:
    """
    Pipeline for comprehensive reenactment quality assessment.
    """
    
    def __init__(self):
        """Initialize the quality assessment pipeline."""
        self.face_detector = FaceDetector.create("mediapipe")
        self.deepfake_detector = DeepfakeDetector("efficientnet", 0.5)
        self.reenactor = FaceReenactor.create("tps")
    
    def assess_reenactment_quality(self, source_image, driving_video, output_folder):
        """
        Generate reenactment and assess its quality.
        
        Args:
            source_image (str): Path to source image
            driving_video (str): Path to driving video
            output_folder (str): Output folder for results
        
        Returns:
            dict: Quality assessment results
        """
        os.makedirs(output_folder, exist_ok=True)
        
        print("Generating reenactment...")
        
        # Generate reenactment
        reenacted_video_path = self.reenactor.reenact_from_video(
            source_path=source_image,
            driving_video_path=driving_video,
            output_path=output_folder,
            save_comparison=True,
            resize_to_image_resolution=False
        )
        
        print("Assessing quality...")
        
        # Assess various quality metrics
        quality_metrics = {
            'face_consistency': self._assess_face_consistency(reenacted_video_path),
            'temporal_stability': self._assess_temporal_stability(reenacted_video_path),
            'deepfake_detectability': self._assess_deepfake_detectability(reenacted_video_path),
            'visual_artifacts': self._assess_visual_artifacts(reenacted_video_path),
            'motion_preservation': self._assess_motion_preservation(driving_video, reenacted_video_path)
        }
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_quality(quality_metrics)
        
        # Generate quality report
        quality_report = {
            'source_image': source_image,
            'driving_video': driving_video,
            'reenacted_video': reenacted_video_path,
            'quality_metrics': quality_metrics,
            'overall_quality': overall_score,
            'quality_rating': self._get_quality_rating(overall_score),
            'recommendations': self._generate_recommendations(quality_metrics)
        }
        
        # Save quality report
        import json
        report_path = os.path.join(output_folder, "quality_report.json")
        with open(report_path, 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
        
        return quality_report
    
    def _assess_face_consistency(self, video_path):
        """Assess face detection consistency across frames."""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames for analysis
        sample_size = min(20, frame_count)
        frame_indices = np.linspace(0, frame_count - 1, sample_size, dtype=int)
        
        face_counts = []
        face_confidences = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Save frame temporarily
            temp_path = "temp_consistency_frame.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Detect faces
            detections = self.face_detector.detect(
                image_path=temp_path,
                save_csv=False,
                save_annotated=False
            )
            
            face_counts.append(len(detections))
            if detections:
                avg_confidence = sum(det.confidence for det in detections) / len(detections)
                face_confidences.append(avg_confidence)
        
        cap.release()
        
        # Clean up
        if os.path.exists("temp_consistency_frame.jpg"):
            os.remove("temp_consistency_frame.jpg")
        
        # Calculate consistency metrics
        face_consistency = len([c for c in face_counts if c == 1]) / len(face_counts) if face_counts else 0
        avg_confidence = sum(face_confidences) / len(face_confidences) if face_confidences else 0
        confidence_stability = 1 - (np.std(face_confidences) if len(face_confidences) > 1 else 0)
        
        return {
            'face_presence_consistency': face_consistency,
            'average_face_confidence': avg_confidence,
            'confidence_stability': confidence_stability,
            'overall_score': (face_consistency + avg_confidence + confidence_stability) / 3
        }
    
    def _assess_temporal_stability(self, video_path):
        """Assess temporal stability by analyzing frame-to-frame changes."""
        cap = cv2.VideoCapture(video_path)
        
        prev_frame = None
        frame_diffs = []
        
        frame_count = 0
        while frame_count < 50:  # Analyze first 50 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, gray)
                frame_diffs.append(np.mean(diff))
            
            prev_frame = gray
            frame_count += 1
        
        cap.release()
        
        if not frame_diffs:
            return {'overall_score': 0}
        
        # Calculate stability metrics
        avg_diff = np.mean(frame_diffs)
        std_diff = np.std(frame_diffs)
        
        # Normalize scores (lower differences = higher stability)
        stability_score = max(0, 1 - (avg_diff / 255))
        consistency_score = max(0, 1 - (std_diff / 50))
        
        return {
            'average_frame_difference': avg_diff,
            'difference_stability': consistency_score,
            'temporal_stability_score': stability_score,
            'overall_score': (stability_score + consistency_score) / 2
        }
    
    def _assess_deepfake_detectability(self, video_path):
        """Assess how easily the reenactment can be detected as synthetic."""
        # Analyze video with deepfake detector
        detections = self.deepfake_detector.detect(
            media_path=video_path,
            save_csv=False,
            save_annotated=False,
            num_frames=15
        )
        
        if not detections:
            return {'overall_score': 0}
        
        # Calculate detectability metrics
        deepfake_frames = sum(1 for det in detections if det.is_deepfake)
        deepfake_ratio = deepfake_frames / len(detections)
        avg_confidence = sum(det.confidence for det in detections) / len(detections)
        
        # Lower detectability = higher quality
        detectability_score = 1 - deepfake_ratio
        confidence_score = 1 - avg_confidence if deepfake_ratio > 0.5 else avg_confidence
        
        return {
            'deepfake_detection_ratio': deepfake_ratio,
            'average_detection_confidence': avg_confidence,
            'stealth_score': detectability_score,
            'confidence_score': confidence_score,
            'overall_score': (detectability_score + confidence_score) / 2
        }
    
    def _assess_visual_artifacts(self, video_path):
        """Assess visual artifacts in the reenacted video."""
        cap = cv2.VideoCapture(video_path)
        
        # Sample frames for artifact analysis
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_size = min(10, frame_count)
        frame_indices = np.linspace(0, frame_count - 1, sample_size, dtype=int)
        
        sharpness_scores = []
        blur_scores = []
        noise_scores = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Assess sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_scores.append(laplacian_var)
            
            # Assess blur using Gaussian blur difference
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            blur_diff = np.mean(np.abs(gray.astype(float) - blurred.astype(float)))
            blur_scores.append(blur_diff)
            
            # Assess noise using high-frequency content
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            noise_level = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
            noise_scores.append(noise_level)
        
        cap.release()
        
        if not sharpness_scores:
            return {'overall_score': 0}
        
        # Normalize scores
        avg_sharpness = np.mean(sharpness_scores)
        avg_blur = np.mean(blur_scores)
        avg_noise = np.mean(noise_scores)
        
        # Convert to quality scores (0-1)
        sharpness_quality = min(1, avg_sharpness / 1000)  # Normalize based on typical values
        blur_quality = max(0, 1 - (avg_blur / 50))
        noise_quality = max(0, 1 - (avg_noise / 100))
        
        return {
            'sharpness_score': sharpness_quality,
            'blur_score': blur_quality,
            'noise_score': noise_quality,
            'overall_score': (sharpness_quality + blur_quality + noise_quality) / 3
        }
    
    def _assess_motion_preservation(self, original_video, reenacted_video):
        """Assess how well motion is preserved from original to reenacted video."""
        # This is a simplified motion assessment
        # In practice, you might use optical flow or landmark tracking
        
        # For now, return a placeholder score
        # This could be enhanced with proper motion analysis
        return {
            'motion_similarity': 0.8,  # Placeholder
            'motion_smoothness': 0.75,  # Placeholder
            'overall_score': 0.77
        }
    
    def _calculate_overall_quality(self, metrics):
        """Calculate overall quality score from individual metrics."""
        weights = {
            'face_consistency': 0.25,
            'temporal_stability': 0.20,
            'deepfake_detectability': 0.15,
            'visual_artifacts': 0.25,
            'motion_preservation': 0.15
        }
        
        weighted_score = sum(
            metrics[metric]['overall_score'] * weight
            for metric, weight in weights.items()
            if metric in metrics
        )
        
        return weighted_score
    
    def _get_quality_rating(self, score):
        """Convert quality score to rating."""
        if score >= 0.8:
            return "EXCELLENT"
        elif score >= 0.65:
            return "GOOD"
        elif score >= 0.5:
            return "FAIR"
        else:
            return "POOR"
    
    def _generate_recommendations(self, metrics):
        """Generate recommendations based on quality metrics."""
        recommendations = []
        
        if metrics['face_consistency']['overall_score'] < 0.7:
            recommendations.append("Consider using higher resolution source image for better face consistency")
        
        if metrics['temporal_stability']['overall_score'] < 0.6:
            recommendations.append("Video shows temporal instability; try preprocessing the driving video")
        
        if metrics['deepfake_detectability']['overall_score'] < 0.5:
            recommendations.append("Reenactment is easily detectable; consider adjusting model parameters")
        
        if metrics['visual_artifacts']['overall_score'] < 0.6:
            recommendations.append("Visual artifacts detected; consider post-processing or higher quality inputs")
        
        if not recommendations:
            recommendations.append("Quality looks good! No specific improvements needed.")
        
        return recommendations

# Usage example
quality_pipeline = ReenactmentQualityPipeline()

# Assess quality of a reenactment
source_image = "assets/images/portrait.jpg"
driving_video = "assets/videos/expression_video.mp4"
output_folder = "output/quality_assessment"

if os.path.exists(source_image) and os.path.exists(driving_video):
    quality_report = quality_pipeline.assess_reenactment_quality(
        source_image, driving_video, output_folder
    )
    
    print("\n--- Reenactment Quality Report ---")
    print(f"Overall Quality: {quality_report['overall_quality']:.3f} ({quality_report['quality_rating']})")
    print(f"\nDetailed Metrics:")
    for metric_name, metric_data in quality_report['quality_metrics'].items():
        print(f"  {metric_name}: {metric_data['overall_score']:.3f}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(quality_report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print(f"\nDetailed report saved to: {output_folder}/quality_report.json")
```

## Next Steps

- Combine these pipelines with other Mukh features for even more sophisticated workflows
- Check the [API Reference](../api/core.md) for detailed documentation of all classes and methods
- Explore individual feature documentation for deeper understanding:
  - [Face Detection](face-detection.md)
  - [Face Reenactment](face-reenactment.md)
  - [Deepfake Detection](deepfake-detection.md) 