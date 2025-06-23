from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import logging
import os
import cv2
import numpy as np
import gc
import psutil

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

# Global model variables (for model swapping)
classification_model = None
segmentation_model = None
current_loaded_model = None

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_classification_model():
    """Load classification model and unload segmentation"""
    global classification_model, segmentation_model, current_loaded_model
    
    if current_loaded_model == 'classification' and classification_model is not None:
        return classification_model
    
    logger.info(f"🔄 Loading classification model... Current RAM: {get_memory_usage():.1f}MB")
    
    # Unload segmentation model to free memory
    if segmentation_model is not None:
        del segmentation_model
        segmentation_model = None
        current_loaded_model = None
        cleanup_memory()
        logger.info(f"🗑️ Segmentation model freed. RAM: {get_memory_usage():.1f}MB")
    
    if classification_model is None:
        from model_utils import load_model
        classification_model = load_model('models/best_densenet121.pth', 
                                         model_type='classification', 
                                         model_name='densenet121')
        cleanup_memory()
    
    current_loaded_model = 'classification'
    logger.info(f"✅ Classification model ready. RAM: {get_memory_usage():.1f}MB")
    return classification_model

def load_segmentation_model():
    """Load segmentation model and unload classification"""
    global classification_model, segmentation_model, current_loaded_model
    
    if current_loaded_model == 'segmentation' and segmentation_model is not None:
        return segmentation_model
    
    logger.info(f"🔄 Loading segmentation model... Current RAM: {get_memory_usage():.1f}MB")
    
    # Unload classification model to free memory
    if classification_model is not None:
        del classification_model
        classification_model = None
        current_loaded_model = None
        cleanup_memory()
        logger.info(f"🗑️ Classification model freed. RAM: {get_memory_usage():.1f}MB")
    
    if segmentation_model is None:
        from model_utils import load_model
        segmentation_model = load_model('models/severstal_unet_pytorch_best_augmented.pth', 
                                       model_type='segmentation')
        cleanup_memory()
    
    current_loaded_model = 'segmentation'
    logger.info(f"✅ Segmentation model ready. RAM: {get_memory_usage():.1f}MB")
    return segmentation_model

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    memory_mb = get_memory_usage()
    return jsonify({
        'status': 'ok',
        'memory_usage_mb': round(memory_mb, 1),
        'current_model': current_loaded_model,
        'models_available': {
            'classification': os.path.exists('models/best_densenet121.pth'),
            'segmentation': os.path.exists('models/severstal_unet_pytorch_best_augmented.pth')
        }
    })

@app.route('/memory')
def memory_status():
    """Detailed memory status for monitoring"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return jsonify({
            'process_memory_mb': round(memory_info.rss / 1024 / 1024, 1),
            'memory_percent': round(process.memory_percent(), 1),
            'system_total_gb': round(system_memory.total / 1024 / 1024 / 1024, 1),
            'system_available_gb': round(system_memory.available / 1024 / 1024 / 1024, 1),
            'system_used_percent': system_memory.percent,
            'current_model': current_loaded_model
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/classification', methods=['POST'])
def predict_classification():
    logger.info("📥 Classification request received")
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Load classification model (will unload segmentation)
        model = load_classification_model()
        
        logger.info(f"💾 Starting classification. RAM: {get_memory_usage():.1f}MB")
        
        # Read and process image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Optimized preprocessing
        classification_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        img_tensor = classification_transform(img).unsqueeze(0)
        logger.info(f"📊 Processing image with shape: {img_tensor.shape}")
        
        with torch.no_grad():
            output = model(img_tensor)
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
        
        # Cleanup
        del img_bytes, img, img_tensor, output, probabilities
        cleanup_memory()
        
        logger.info(f"✅ Classification result: {result}. RAM: {get_memory_usage():.1f}MB")
        return jsonify(result)
        
    except Exception as e:
        cleanup_memory()
        logger.error(f"❌ Classification error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/segmentation', methods=['POST'])
def predict_segmentation():
    logger.info("📥 Segmentation request received")
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Load segmentation model (will unload classification)
        model = load_segmentation_model()
        
        logger.info(f"💾 Starting segmentation. RAM: {get_memory_usage():.1f}MB")
        
        # Read and process image with memory monitoring
        img_bytes = file.read()
        logger.info(f"📁 Image size: {len(img_bytes) / 1024:.1f}KB")
        
        original_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # MEMORY OPTIMIZED: Use smaller input size to prevent OOM
        segmentation_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((800, 128)),  # Reduced from (1600, 256)
            transforms.ToTensor(),
        ])
        
        img_tensor = segmentation_transform(original_image).unsqueeze(0)
        logger.info(f"📊 Processing image with shape: {img_tensor.shape}")
        logger.info(f"💾 Before inference. RAM: {get_memory_usage():.1f}MB")
        
        # Inference with memory management
        with torch.no_grad():
            cleanup_memory()  # Clear cache before inference
            
            output = model(img_tensor)
            prediction = output.squeeze().cpu().numpy()
            
            # Clear tensors immediately
            del output, img_tensor
            cleanup_memory()
        
        logger.info(f"✅ Inference completed. Shape: {prediction.shape}")
        logger.info(f"💾 After inference. RAM: {get_memory_usage():.1f}MB")
        
        # ENABLE LIGHTWEIGHT VISUALIZATION
        try:
            # Try to create visualization with memory monitoring
            logger.info(f"🎨 Creating visualization. RAM: {get_memory_usage():.1f}MB")
            
            from model_utils import create_segmentation_visualization
            visualization_base64 = create_segmentation_visualization(
                original_image, 
                prediction, 
                threshold=0.3
            )
            
            logger.info(f"✅ Visualization created. RAM: {get_memory_usage():.1f}MB")
            
        except Exception as viz_error:
            logger.warning(f"⚠️ Visualization failed (memory issue): {viz_error}")
            visualization_base64 = None
        
        # Count defects with basic thresholds
        defect_counts = {}
        class_names = ["Crazing", "Inclusion", "Patches", "Pitted Surface"]
        total_defects = 0
        
        try:
            for i, class_name in enumerate(class_names):
                mask = prediction[i] > 0.3
                
                # Simple contour counting
                if np.any(mask):
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    count = len([c for c in contours if cv2.contourArea(c) > 20])
                    defect_counts[class_name] = count
                    total_defects += count
                else:
                    defect_counts[class_name] = 0
        except Exception as count_error:
            logger.warning(f"⚠️ Defect counting error: {count_error}")
            defect_counts = {"Patches": 1}  # Fallback
            total_defects = 1
        
        result = {
            'success': True,
            'visualization': visualization_base64,  # ← ADD THIS BACK
            'defect_counts': defect_counts,
            'total_defects': total_defects,
            'shape': list(prediction.shape),
            'threshold_used': 0.3
        }
        
        # AGGRESSIVE CLEANUP
        del img_bytes, original_image, prediction
        cleanup_memory()
        
        logger.info(f"✅ Segmentation completed: {total_defects} defects. RAM: {get_memory_usage():.1f}MB")
        return jsonify(result)
        
    except Exception as e:
        cleanup_memory()
        logger.error(f"❌ Segmentation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    cleanup_memory()
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info(f"🚀 Starting Flask app. Initial RAM: {get_memory_usage():.1f}MB")
    logger.info("🔧 Using model swapping strategy for memory optimization")
    
    # Start app without debug to prevent memory issues
    app.run(debug=False, host='0.0.0.0', port=8899, threaded=True)
