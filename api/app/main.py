"""
Medical Image Analysis API
Extensible API for medical image classification with Grad-CAM visualization.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
import base64
import os

from app.models import BrainTumorClassifier, PneumoniaClassifier, BoneFractureClassifier, RetinalOCTClassifier

# ============================================================================
# Model Registry - Add new models here
# ============================================================================

MODELS: Dict[str, Any] = {}

def load_models():
    """Load all available models at startup."""
    weights_dir = os.getenv("WEIGHTS_DIR", "./weights")
    
    # Brain Tumor Model
    brain_tumor_weights = os.path.join(weights_dir, "brain_tumor_model.pth")
    brain_tumor_config = os.path.join(weights_dir, "brain_tumor_config.json")
    
    if os.path.exists(brain_tumor_weights) and os.path.exists(brain_tumor_config):
        try:
            classifier = BrainTumorClassifier()
            classifier.load_model(brain_tumor_weights, brain_tumor_config)
            MODELS["brain_tumor"] = classifier
            print("✓ Brain tumor model loaded")
        except Exception as e:
            print(f"✗ Failed to load brain tumor model: {e}")
    else:
        print(f"✗ Brain tumor model files not found in {weights_dir}")
    
    # Pneumonia Model
    pneumonia_weights = os.path.join(weights_dir, "pneumonia_model.pth")
    pneumonia_config = os.path.join(weights_dir, "pneumonia_config.json")

    if os.path.exists(pneumonia_weights) and os.path.exists(pneumonia_config):
        try:
            classifier = PneumoniaClassifier()
            classifier.load_model(pneumonia_weights, pneumonia_config)
            MODELS["pneumonia"] = classifier
            print("✓ Pneumonia model loaded")
        except Exception as e:
            print(f"✗ Failed to load pneumonia model: {e}")
    else:
        print(f"✗ Pneumonia model files not found in {weights_dir}")

    # Bone fracture Model
    bone_fracture_weights = os.path.join(weights_dir, "bone_fracture_model.pth")
    bone_fracture_config = os.path.join(weights_dir, "bone_fracture_config.json")

    if os.path.exists(bone_fracture_weights) and os.path.exists(bone_fracture_config):
        try:
            classifier = BoneFractureClassifier()
            classifier.load_model(bone_fracture_weights, bone_fracture_config)
            MODELS["bone_fracture"] = classifier
            print("✓ Bone fracture model loaded")
        except Exception as e:
            print(f"✗ Failed to load bone fracture model: {e}")
    else:
        print(f"✗ Bone fracture model files not found in {weights_dir}")

    # Retinal OCT Model
    retinal_oct_weights = os.path.join(weights_dir, "retinal_oct_model.pth")
    retinal_oct_config = os.path.join(weights_dir, "retinal_oct_config.json")

    if os.path.exists(retinal_oct_weights) and os.path.exists(retinal_oct_config):
        try:
            classifier = RetinalOCTClassifier()
            classifier.load_model(retinal_oct_weights, retinal_oct_config)
            MODELS["retinal_oct"] = classifier
            print("✓ Retinal OCT model loaded")
        except Exception as e:
            print(f"✗ Failed to load retinal OCT model: {e}")
    else:
        print(f"✗ Retinal OCT model files not found in {weights_dir}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: Load models
    print("=" * 50)
    print("Loading models...")
    load_models()
    print(f"Loaded {len(MODELS)} model(s): {list(MODELS.keys())}")
    print("=" * 50)
    yield
    # Shutdown: Cleanup if needed
    print("Shutting down...")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Medical Image Analysis API",
    description="""
    API for medical image classification with Grad-CAM visualization.
    
    ## Available Models
    - **brain_tumor**: MRI brain tumor classification (glioma, meningioma, pituitary, notumor)
    
    ## Features
    - Image classification with confidence scores
    - Grad-CAM visualization for model interpretability
    - Support for multiple model types
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """API root - health check."""
    return {
        "status": "healthy",
        "service": "Medical Image Analysis API",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "models_loaded": len(MODELS),
        "available_models": list(MODELS.keys())
    }


@app.get("/models", tags=["Info"])
async def list_models():
    """List all available models and their details."""
    models_info = {}
    for name, model in MODELS.items():
        models_info[name] = model.get_model_info()
    return {"models": models_info}


@app.get("/models/{model_name}", tags=["Info"])
async def get_model_info(model_name: str):
    """Get detailed information about a specific model."""
    if model_name not in MODELS:
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_name}' not found. Available: {list(MODELS.keys())}"
        )
    return MODELS[model_name].get_model_info()


# ============================================================================
# Prediction Endpoints
# ============================================================================

@app.post("/predict/{model_name}", tags=["Prediction"])
async def predict(
    model_name: str,
    file: UploadFile = File(..., description="Image file to classify")
):
    """
    Run classification on an uploaded image.
    
    Returns prediction with confidence scores for all classes.
    """
    # Validate model
    if model_name not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {list(MODELS.keys())}"
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Run prediction
        result = MODELS[model_name].predict(image_bytes)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/{model_name}/gradcam", tags=["Prediction"])
async def predict_with_gradcam(
    model_name: str,
    file: UploadFile = File(..., description="Image file to classify"),
    output_type: str = Query(
        default="all",
        description="Visualization type: 'heatmap', 'overlay', or 'all'"
    ),
    target_class: Optional[int] = Query(
        default=None,
        description="Class index to visualize. If not provided, uses predicted class."
    )
):
    """
    Run classification with Grad-CAM visualization.
    
    Returns prediction with confidence scores and base64-encoded visualization images.
    
    **Output types:**
    - `heatmap`: Just the Grad-CAM heatmap
    - `overlay`: Heatmap overlaid on original image
    - `all`: Original, heatmap, overlay, and side-by-side comparison
    """
    # Validate model
    if model_name not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {list(MODELS.keys())}"
        )
    
    # Validate output type
    if output_type not in ["heatmap", "overlay", "all"]:
        raise HTTPException(
            status_code=400,
            detail="output_type must be 'heatmap', 'overlay', or 'all'"
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Run prediction with Grad-CAM
        result = MODELS[model_name].get_gradcam(
            image_bytes,
            target_class=target_class,
            output_type=output_type
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/{model_name}/gradcam/image", tags=["Prediction"])
async def predict_gradcam_image(
    model_name: str,
    file: UploadFile = File(..., description="Image file to classify"),
    image_type: str = Query(
        default="comparison",
        description="Image to return: 'original', 'heatmap', 'overlay', or 'comparison'"
    ),
    target_class: Optional[int] = Query(
        default=None,
        description="Class index to visualize. If not provided, uses predicted class."
    )
):
    """
    Run classification and return Grad-CAM visualization as an image file.
    
    Use this endpoint when you want to display the visualization directly
    (e.g., in an <img> tag) rather than receiving base64 data.
    """
    # Validate model
    if model_name not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {list(MODELS.keys())}"
        )
    
    # Map image_type to output_type
    type_map = {
        "original": "all",
        "heatmap": "heatmap", 
        "overlay": "overlay",
        "comparison": "all"
    }
    
    if image_type not in type_map:
        raise HTTPException(
            status_code=400,
            detail="image_type must be 'original', 'heatmap', 'overlay', or 'comparison'"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Run prediction with Grad-CAM
        result = MODELS[model_name].get_gradcam(
            image_bytes,
            target_class=target_class,
            output_type=type_map[image_type]
        )
        
        # Get requested image
        image_key = image_type if image_type != "comparison" else "comparison"
        if image_key not in result["images"]:
            # Fallback for heatmap/overlay only modes
            image_key = list(result["images"].keys())[0]
        
        image_b64 = result["images"][image_key]
        image_data = base64.b64decode(image_b64)
        
        return Response(
            content=image_data,
            media_type="image/png",
            headers={
                "X-Prediction": result["prediction"],
                "X-Confidence": str(result["confidence"]),
                "X-Model": model_name
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# ============================================================================
# Run with: uvicorn app.main:app --reload
# ============================================================================
