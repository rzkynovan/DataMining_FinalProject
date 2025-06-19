from flask import Flask, request, jsonify, render_template, send_from_directory
from model_utils import load_model
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import logging
import os
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create necessary directories
os.makedirs('templates', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

# Load models
try:
    classification_model = load_model('models/best_densenet121.pth', model_type='classification', model_name='densenet121')
    segmentation_model = load_model('models/severstal_unet_pytorch_best_augmented.pth', model_type='segmentation')
    logger.info("✅ Models loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load models: {e}")
    classification_model = None
    segmentation_model = None

# Check input channels for models
def get_input_channels(model):
    if model is None:
        return 3
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            return module.in_channels
    return 3

classification_channels = get_input_channels(classification_model)
segmentation_channels = get_input_channels(segmentation_model)

logger.info(f"Classification model expects {classification_channels} channels")
logger.info(f"Segmentation model expects {segmentation_channels} channels")

# Define preprocessing transformations
if classification_channels == 3:
    classification_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
else:
    classification_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

# IMPROVED: Fixed resolution to match training data (1600x256)
if segmentation_channels == 3:
    segmentation_transform = transforms.Compose([
        transforms.Resize((1600, 256)),  # FIXED: width=1600, height=256
        transforms.ToTensor(),
    ])
else:
    segmentation_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((1600, 256)),  # FIXED: width=1600, height=256
        transforms.ToTensor(),
    ])

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'ok',
        'models_loaded': {
            'classification': classification_model is not None,
            'segmentation': segmentation_model is not None
        }
    })

@app.route('/predict/classification', methods=['POST'])
def predict_classification():
    logger.info("📥 Classification request received")
    
    if classification_model is None:
        return jsonify({'error': 'Classification model not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read and process image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_tensor = classification_transform(img).unsqueeze(0)
        
        logger.info(f"📊 Processing image with shape: {img_tensor.shape}")
        
        with torch.no_grad():
            output = classification_model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            # Class names for steel defects
            class_names = ['No Defect', 'Crazing', 'Inclusion', 'Patches', 'Pitted Surface']
            prediction_class = predicted.item()
            confidence = probabilities[0][prediction_class].item()
            
        result = {
            'prediction': prediction_class + 1,
            'class_name': class_names[prediction_class] if prediction_class < len(class_names) else f'Class {prediction_class + 1}',
            'confidence': round(confidence * 100, 2)
        }
        
        logger.info(f"✅ Classification result: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Classification error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/segmentation', methods=['POST'])
def predict_segmentation():
    logger.info("📥 Segmentation request received")
    
    if segmentation_model is None:
        return jsonify({'error': 'Segmentation model not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read and process image
        img_bytes = file.read()
        original_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Prepare image for model with FIXED resolution
        img_tensor = segmentation_transform(original_image).unsqueeze(0)
        
        logger.info(f"📊 Processing image with shape: {img_tensor.shape}")
        
        with torch.no_grad():
            output = segmentation_model(img_tensor)
            prediction = output.squeeze().cpu().numpy()
        
        # IMPROVED: Try multiple thresholds for better detection
        thresholds = [0.2, 0.3, 0.5, 0.7]
        best_result = None
        max_defects = 0
        
        for thresh in thresholds:
            logger.info(f"🔍 Testing threshold: {thresh}")
            
            try:
                # Create visualization
                from model_utils import create_segmentation_visualization
                visualization_base64 = create_segmentation_visualization(
                    original_image, 
                    prediction, 
                    threshold=thresh
                )
                
                # Count defects with this threshold
                total_defects = 0
                defect_counts = {}
                class_names = ["Crazing", "Inclusion", "Patches", "Pitted Surface"]
                
                for i, class_name in enumerate(class_names):
                    mask = prediction[i] > thresh
                    
                    # Resize mask to match original resolution for better contour detection
                    mask_resized = cv2.resize(
                        mask.astype(np.uint8), 
                        (original_image.size[0], original_image.size[1]),
                        interpolation=cv2.INTER_NEAREST
                    )
                    
                    contours, _ = cv2.findContours(
                        (mask_resized * 255).astype(np.uint8), 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    # IMPROVED: Lower area threshold for better small defect detection
                    count = len([c for c in contours if cv2.contourArea(c) > 20])
                    defect_counts[class_name] = count
                    total_defects += count
                
                # Keep result with most detected defects (but reasonable threshold)
                if total_defects > max_defects and thresh >= 0.2:
                    max_defects = total_defects
                    best_result = {
                        'success': True,
                        'visualization': visualization_base64,
                        'defect_counts': defect_counts,
                        'threshold_used': thresh,
                        'total_defects': total_defects,
                        'shape': list(prediction.shape)
                    }
                    
                logger.info(f"📊 Threshold {thresh}: {total_defects} total defects")
                
            except Exception as viz_error:
                logger.warning(f"⚠️ Visualization failed for threshold {thresh}: {viz_error}")
                continue
        
        # If no defects found with any threshold, use default threshold 0.3
        if best_result is None:
            logger.info("🔄 No defects detected, using default threshold 0.3")
            
            from model_utils import create_segmentation_visualization
            visualization_base64 = create_segmentation_visualization(
                original_image, 
                prediction, 
                threshold=0.3
            )
            
            best_result = {
                'success': True,
                'visualization': visualization_base64,
                'defect_counts': {"Crazing": 0, "Inclusion": 0, "Patches": 0, "Pitted Surface": 0},
                'threshold_used': 0.3,
                'total_defects': 0,
                'shape': list(prediction.shape),
                'message': 'No significant defects detected'
            }
        
        logger.info(f"✅ Segmentation completed: {best_result['total_defects']} defects at threshold {best_result['threshold_used']}")
        return jsonify(best_result)
        
    except Exception as e:
        logger.error(f"❌ Segmentation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8899)
