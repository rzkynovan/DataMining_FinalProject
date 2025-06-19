import torch
import torch.nn as nn
import timm
import logging
import os
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Keep all existing classes unchanged
class SteelDefectCNN(nn.Module):
    def __init__(self, model_name, num_classes=4, in_channels=1):
        super(SteelDefectCNN, self).__init__()
        self.model = timm.create_model(model_name, pretrained=False, in_chans=in_channels)
        
        if 'efficientnet' in model_name:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)
        elif 'resnet' in model_name:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif 'densenet' in model_name:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)
        elif 'mobilenet' in model_name:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        return torch.sigmoid(self.final_conv(x))

def auto_detect_input_channels(state_dict):
    for key in state_dict.keys():
        if 'conv0.weight' in key or 'conv1.weight' in key:
            shape = state_dict[key].shape
            if len(shape) == 4:
                return shape[1]
    
    for key in state_dict.keys():
        if 'weight' in key and len(state_dict[key].shape) == 4:
            shape = state_dict[key].shape
            if shape[2] == 7 and shape[3] == 7:
                return shape[1]
    
    return 1

def merge_nearby_contours(contours, min_distance=30):
    """Merge contours that are close to each other"""
    if len(contours) <= 1:
        return contours
    
    merged = []
    used = set()
    
    for i, contour1 in enumerate(contours):
        if i in used:
            continue
            
        # Get centroid of current contour
        M1 = cv2.moments(contour1)
        if M1["m00"] == 0:
            continue
        cx1 = int(M1["m10"] / M1["m00"])
        cy1 = int(M1["m01"] / M1["m00"])
        
        # Find nearby contours to merge
        group = [contour1]
        used.add(i)
        
        for j, contour2 in enumerate(contours):
            if j in used or j <= i:
                continue
                
            M2 = cv2.moments(contour2)
            if M2["m00"] == 0:
                continue
            cx2 = int(M2["m10"] / M2["m00"])
            cy2 = int(M2["m01"] / M2["m00"])
            
            # Calculate distance between centroids
            distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            
            if distance < min_distance:
                group.append(contour2)
                used.add(j)
        
        # Merge contours in group
        if len(group) == 1:
            merged.append(group[0])
        else:
            # Combine all contours in group
            all_points = np.vstack(group)
            hull = cv2.convexHull(all_points)
            merged.append(hull)
    
    return merged

def create_professional_legend(image_shape, detection_summary):
    """Create a professional legend box"""
    height, width = image_shape[:2]
    
    # Legend configuration
    legend_width = 200
    legend_height = 120
    legend_x = width - legend_width - 20
    legend_y = 20
    
    # Colors (BGR format)
    class_colors = {
        'Crazing': (0, 0, 255),       # Red
        'Inclusion': (0, 255, 255),   # Yellow  
        'Patches': (255, 0, 0),       # Blue
        'Pitted Surface': (0, 255, 0) # Green
    }
    
    # Create legend overlay
    legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255  # White background
    
    # Add border
    cv2.rectangle(legend, (0, 0), (legend_width-1, legend_height-1), (0, 0, 0), 2)
    
    # Add title
    cv2.putText(legend, "DEFECT TYPES", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add legend items
    y_offset = 35
    for class_name, color in class_colors.items():
        count = detection_summary.get(class_name, 0)
        if count > 0:
            # Color box
            cv2.rectangle(legend, (10, y_offset-8), (25, y_offset-2), color, -1)
            cv2.rectangle(legend, (10, y_offset-8), (25, y_offset-2), (0, 0, 0), 1)
            
            # Text
            text = f"{class_name}: {count}"
            cv2.putText(legend, text, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            y_offset += 18
    
    return legend, (legend_x, legend_y)

def create_segmentation_visualization(original_image_pil, segmentation_output, threshold=0.5):
    """
    Create CLEAN, PROFESSIONAL segmentation visualization - FIXED VERSION
    """
    # Convert PIL to numpy array
    if original_image_pil.mode != 'RGB':
        original_image_pil = original_image_pil.convert('RGB')
    
    original_np = np.array(original_image_pil)
    
    # Define colors for each defect class (BGR format)
    class_colors = {
        0: (0, 0, 255),       # Red - Crazing
        1: (0, 255, 255),     # Yellow - Inclusion  
        2: (255, 0, 0),       # Blue - Patches
        3: (0, 255, 0)        # Green - Pitted Surface
    }
    
    class_names = {
        0: "Crazing",
        1: "Inclusion", 
        2: "Patches",
        3: "Pitted Surface"
    }
    
    # IMPROVED: Confidence-based thresholds
    confidence_thresholds = {
        0: 0.001,  # Crazing
        1: 0.0005, # Inclusion (more sensitive)
        2: 0.4,    # Patches (higher confidence required)
        3: 0.002   # Pitted Surface
    }
    
    print("\n🎨 CREATING PROFESSIONAL VISUALIZATION:")
    
    # Create overlay image
    overlay = original_np.copy()
    detection_summary = {}
    all_significant_contours = {}
    
    # Process each class channel
    for class_idx in range(segmentation_output.shape[0]):
        class_threshold = confidence_thresholds[class_idx]
        
        # Get mask for this class
        mask = segmentation_output[class_idx]
        
        # Apply threshold
        binary_mask = (mask > class_threshold).astype(np.uint8) * 255
        
        # Resize mask to match original image size
        if binary_mask.shape != original_np.shape[:2]:
            binary_mask = cv2.resize(binary_mask, (original_np.shape[1], original_np.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # IMPROVED: Filter by area and confidence
        min_area = 25 if class_idx != 2 else 40  # Stricter for patches
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # IMPROVED: Merge nearby contours to reduce clutter
        if len(valid_contours) > 1:
            valid_contours = merge_nearby_contours(valid_contours, min_distance=40)
        
        # Store for legend
        detection_summary[class_names[class_idx]] = len(valid_contours)
        all_significant_contours[class_idx] = valid_contours
        
        if len(valid_contours) > 0:
            print(f"✅ {class_names[class_idx]}: {len(valid_contours)} regions detected")
    
    # FIXED: Draw contours with proper styling
    contour_id = 1
    for class_idx, contours in all_significant_contours.items():
        if len(contours) == 0:
            continue
            
        color = class_colors[class_idx]
        
        for contour in contours:
            # FIXED: Simple and clean contour drawing
            # Draw filled contour with alpha blending
            contour_overlay = overlay.copy()
            cv2.fillPoly(contour_overlay, [contour], color)
            
            # FIXED: Proper alpha blending
            alpha = 0.3
            cv2.addWeighted(contour_overlay, alpha, overlay, 1 - alpha, 0, overlay)
            
            # Draw contour outline
            cv2.drawContours(overlay, [contour], -1, color, 2)
            
            # IMPROVED: Add clean numbered markers
            # Get centroid for marker placement
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw numbered marker
                cv2.circle(overlay, (cx, cy), 12, (255, 255, 255), -1)  # White circle
                cv2.circle(overlay, (cx, cy), 12, color, 2)  # Colored border
                cv2.putText(overlay, str(contour_id), (cx-6, cy+4), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                contour_id += 1
    
    # IMPROVED: Add professional legend
    total_defects = sum(detection_summary.values())
    if total_defects > 0:
        legend, legend_pos = create_professional_legend(original_np.shape, detection_summary)
        
        # Overlay legend on image
        legend_x, legend_y = legend_pos
        legend_h, legend_w = legend.shape[:2]
        
        # Ensure legend fits within image bounds
        legend_x = min(legend_x, original_np.shape[1] - legend_w)
        legend_y = min(legend_y, original_np.shape[0] - legend_h)
        
        # FIXED: Simple legend overlay
        overlay[legend_y:legend_y+legend_h, legend_x:legend_x+legend_w] = legend
    
    # IMPROVED: Add professional title
    title_text = f"Steel Defect Analysis - {total_defects} Defects Detected"
    title_width = len(title_text) * 12
    cv2.rectangle(overlay, (10, 10), (title_width, 35), (0, 0, 0), -1)  # Background
    cv2.putText(overlay, title_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    print(f"🎯 PROFESSIONAL VISUALIZATION COMPLETED:")
    print(f"   Total defects: {total_defects}")
    print(f"   Detection summary: {detection_summary}")
    
    # Convert back to PIL and base64
    overlay_pil = Image.fromarray(overlay)
    buffer = BytesIO()
    overlay_pil.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_base64

def load_model(model_path, model_type='classification', model_name='densenet121'):
    """Smart model loading dengan auto-detection input channels"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        loaded = torch.load(model_path, map_location=device)
        
        if hasattr(loaded, 'eval'):
            logger.info(f"✅ Loaded complete model from {model_path}")
            loaded.eval()
            return loaded
        
        elif isinstance(loaded, dict):
            logger.info(f"🔄 Loading state_dict with architecture from {model_path}")
            
            input_channels = auto_detect_input_channels(loaded)
            logger.info(f"🔍 Detected {input_channels} input channels")
            
            if model_type == 'classification':
                model = SteelDefectCNN(model_name, num_classes=4, in_channels=input_channels).to(device)
            elif model_type == 'segmentation':
                model = UNET(in_channels=input_channels, out_channels=4).to(device)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            model.load_state_dict(loaded)
            model.eval()
            logger.info(f"✅ Successfully loaded model from {model_path}")
            return model
        
        else:
            raise ValueError(f"Unknown model format in {model_path}")
            
    except Exception as e:
        logger.error(f"❌ Failed to load model from {model_path}: {e}")
        raise RuntimeError(f"Could not load model from {model_path}: {e}")

def encode_image_to_base64(image_pil):
    """Helper function to encode PIL image to base64"""
    buffer = BytesIO()
    image_pil.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_base64
