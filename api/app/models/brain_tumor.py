"""
Brain Tumor Classifier
Multi-class classification: glioma, meningioma, pituitary, notumor
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import io
from typing import Dict, Any, Optional

from .base import BaseClassifier
from ..utils.gradcam import GradCAMVisualizer, image_to_base64


class BrainTumorClassifier(BaseClassifier):
    """
    Brain tumor MRI classifier using EfficientNet-V2-S.
    Classifies into: glioma, meningioma, pituitary, notumor
    """
    
    def __init__(self):
        super().__init__()
        self.model_name = "brain_tumor"
        self.image_size = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transform = None
        self.config = None
    
    def _build_model(self, num_classes: int) -> nn.Module:
        """Build EfficientNet-V2-S with custom classifier head."""
        model = models.efficientnet_v2_s(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(512, num_classes)
        )
        return model
    
    def load_model(self, weights_path: str, config_path: str) -> None:
        """
        Load model weights and configuration.
        
        Args:
            weights_path: Path to .pth weights file
            config_path: Path to model_config.json
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.class_names = self.config['class_names']
        self.image_size = self.config.get('image_size', 224)
        self.mean = self.config['normalization']['mean']
        self.std = self.config['normalization']['std']
        
        # Build and load model
        self.model = self._build_model(len(self.class_names))
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        # Setup Grad-CAM visualizer (target: last conv layer)
        target_layer = self.model.features[-1]
        self.gradcam_visualizer = GradCAMVisualizer(
            model=self.model,
            target_layer=target_layer,
            image_size=self.image_size,
            mean=self.mean,
            std=self.std
        )
        
        print(f"✓ Brain tumor model loaded on {self.device}")
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL Image for model input."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)
    
    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Run prediction on image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Prediction results with confidence scores
        """
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = self.preprocess(image)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        # Build response
        probs_dict = {
            name: float(prob) 
            for name, prob in zip(self.class_names, probabilities[0].tolist())
        }
        
        return {
            "model": self.model_name,
            "prediction": self.class_names[predicted.item()],
            "confidence": float(confidence.item()),
            "probabilities": probs_dict
        }
    
    def get_gradcam(
        self, 
        image_bytes: bytes, 
        target_class: Optional[int] = None,
        output_type: str = "all"
    ) -> Dict[str, Any]:
        """
        Generate Grad-CAM visualization.
        
        Args:
            image_bytes: Raw image bytes
            target_class: Class index to visualize (None = predicted class)
            output_type: "heatmap", "overlay", or "all"
            
        Returns:
            Dictionary with prediction info and base64-encoded visualizations
        """
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = self.preprocess(image)
        
        # Get prediction first
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        # Use predicted class if not specified
        if target_class is None:
            target_class = predicted.item()
        
        # Generate Grad-CAM visualizations
        # Need to enable gradients for Grad-CAM
        input_tensor.requires_grad_(True)
        visualizations = self.gradcam_visualizer.generate_visualization(
            input_tensor,
            target_class=target_class,
            output_type=output_type
        )
        
        # Convert images to base64
        images_b64 = {}
        for name, img in visualizations.items():
            images_b64[name] = image_to_base64(img)
        
        # Build response
        probs_dict = {
            name: float(prob) 
            for name, prob in zip(self.class_names, probabilities[0].tolist())
        }
        
        return {
            "model": self.model_name,
            "prediction": self.class_names[predicted.item()],
            "confidence": float(confidence.item()),
            "probabilities": probs_dict,
            "visualized_class": self.class_names[target_class],
            "visualized_class_index": target_class,
            "images": images_b64
        }
    
    def predict_with_gradcam(
        self, 
        image_bytes: bytes,
        include_gradcam: bool = True,
        gradcam_type: str = "overlay"
    ) -> Dict[str, Any]:
        """
        Convenience method: predict with optional Grad-CAM in one call.
        
        Args:
            image_bytes: Raw image bytes
            include_gradcam: Whether to include Grad-CAM visualization
            gradcam_type: Type of Grad-CAM output
            
        Returns:
            Combined prediction and visualization results
        """
        if include_gradcam:
            return self.get_gradcam(image_bytes, output_type=gradcam_type)
        else:
            return self.predict(image_bytes)
