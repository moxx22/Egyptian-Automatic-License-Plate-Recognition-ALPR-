import os
import cv2
import torch
import numpy as np
from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
import shutil
import uuid

app = Flask(__name__)

# ----------------------------
# Configuration
# ----------------------------
UPLOAD_FOLDER = 'uploads'
TEMP_FOLDER = 'temp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# ----------------------------
# Load Models (Load once at startup)
# ----------------------------
print("Loading models...")

# Load YOLO models
plate_detection_model = YOLO('Predection------use_this/best1.pt')  # Plate detection model
character_detection_model = YOLO('Predection------use_this/best2.pt')  # Character detection model

# Load CNN model for character recognition
class CNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128*8*8, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load character classes and CNN model
dataset = ImageFolder("Predection------use_this/characters")
classes = dataset.classes
num_classes = len(classes)
character_recognition_model = CNN(num_classes)
character_recognition_model.load_state_dict(torch.load("Predection------use_this/best3.pth", map_location="cpu"))
character_recognition_model.eval()

print("All models loaded successfully!")

# ----------------------------
# Image preprocessing for character recognition
# ----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_character(img_path):
    """Predict character from image path"""
    try:
        img = Image.open(img_path)
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = character_recognition_model(img_tensor)
            _, predicted = torch.max(output, 1)
        return classes[predicted.item()]
    except Exception as e:
        print(f"Error predicting character: {e}")
        return "?"

# ----------------------------
# Helper Functions
# ----------------------------
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image(image, target_width=1020, target_height=600):
    """Resize image for optimal processing"""
    return cv2.resize(image, (target_width, target_height))

def clean_temp_files(session_id):
    """Clean up temporary files for a session"""
    temp_dir = os.path.join(TEMP_FOLDER, session_id)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def calculate_iou(box1, box2):
    """Calculate Intersection Over Union (IOU) between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    # Avoid division by zero
    if union_area == 0:
        return 0
    
    return inter_area / union_area

def non_max_suppression(detections, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to eliminate duplicate detections"""
    if not detections:
        return []
    
    # Sort detections by confidence score (descending)
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    filtered_detections = []
    
    while detections:
        # Take the detection with highest confidence
        current = detections.pop(0)
        filtered_detections.append(current)
        
        # Find and remove detections that overlap significantly with current
        detections = [
            det for det in detections 
            if calculate_iou(current['box'], det['box']) < iou_threshold
        ]
    
    return filtered_detections

# ----------------------------
# Pipeline Functions
# ----------------------------
def detect_plates(image_path, session_id):
    """Detect license plates in image"""
    print(f"Detecting plates in {image_path}")
    
    # Create session temp directory
    session_dir = os.path.join(TEMP_FOLDER, session_id)
    cropped_plates_dir = os.path.join(session_dir, "cropped_plates")
    os.makedirs(cropped_plates_dir, exist_ok=True)
    
    # Load and resize image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError("Could not load image")
    
    frame = resize_image(frame)
    
    # Detect plates using YOLO
    results = plate_detection_model(frame)
    
    detected_plates = []
    plate_counter = 1
    
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            class_ids = result.boxes.cls.int().cpu().tolist()
            
            for box, class_id in zip(boxes, class_ids):
                x1, y1, x2, y2 = box
                class_name = plate_detection_model.names[class_id]
                
                # Check if it's a numberplate
                if class_name.lower() == "numberplate":
                    cropped_plate = frame[y1:y2, x1:x2]
                    if cropped_plate.size == 0:
                        continue
                    
                    # Save cropped plate
                    filename = f"{cropped_plates_dir}/plate{plate_counter}.jpg"
                    cv2.imwrite(filename, cropped_plate)
                    detected_plates.append({
                        'path': filename,
                        'name': f'plate{plate_counter}',
                        'box': box
                    })
                    plate_counter += 1
    
    print(f"Detected {len(detected_plates)} plates")
    return detected_plates

def detect_characters(plate_path, plate_name, session_id):
    """Detect characters in a license plate"""
    print(f"Detecting characters in {plate_path}")
    
    session_dir = os.path.join(TEMP_FOLDER, session_id)
    char_output_dir = os.path.join(session_dir, "corepted_char", plate_name)
    os.makedirs(char_output_dir, exist_ok=True)
    
    # Load plate image
    frame = cv2.imread(plate_path)
    if frame is None:
        raise ValueError(f"Could not load plate image: {plate_path}")
    
    # Resize for character detection
    frame = cv2.resize(frame, (400, 200))
    
    # Detect characters
    results = character_detection_model(frame)
    
    detected_characters = []
    
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            class_ids = result.boxes.cls.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().numpy()
            
            for box, class_id, confidence in zip(boxes, class_ids, confidences):
                x1, y1, x2, y2 = box
                class_name = character_detection_model.names[class_id]
                
                # Store character data
                detected_characters.append({
                    'box': box,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'cropped_char': frame[y1:y2, x1:x2]
                })
    
    # Apply Non-Maximum Suppression to remove duplicate detections
    detected_characters = non_max_suppression(detected_characters, iou_threshold=0.5)
    
    # Sort characters from right to left (Arabic RTL)
    detected_characters.sort(key=lambda char: char['box'][0], reverse=True)
    
    # Save characters
    character_count = 0
    for char_data in detected_characters:
        if char_data['cropped_char'].size > 0:
            char_filename = f"{char_output_dir}/{character_count + 1}.jpg"
            cv2.imwrite(char_filename, char_data['cropped_char'])
            character_count += 1
    
    print(f"Detected {character_count} characters in {plate_name} (after NMS)")
    return character_count

def recognize_characters(session_id):
    """Recognize characters and generate final plate text"""
    print("Recognizing characters...")
    
    char_root = os.path.join(TEMP_FOLDER, session_id, "corepted_char")
    results = []
    
    if not os.path.exists(char_root):
        return results
    
    for plate_folder in sorted(os.listdir(char_root)):
        full_plate_path = os.path.join(char_root, plate_folder)
        if not os.path.isdir(full_plate_path):
            continue
        
        # Get character images
        files = [f for f in os.listdir(full_plate_path) if f.lower().endswith((".jpg", ".jpeg"))]
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        
        predictions = []
        for file in files:
            img_path = os.path.join(full_plate_path, file)
            pred = predict_character(img_path)
            predictions.append(pred)
        
        # For Arabic RTL, we keep the order as detected (already sorted RTL)
        plate_text = "".join(predictions)
        results.append({
            'plate_name': plate_folder,
            'plate_text': plate_text,
            'character_count': len(predictions)
        })
    
    return results

# ----------------------------
# Main Pipeline Function
# ----------------------------
def process_image_pipeline(image_path):
    """Main pipeline: Process image and return license plate recognition results"""
    session_id = str(uuid.uuid4())
    
    try:
        # Step 1: Detect license plates
        detected_plates = detect_plates(image_path, session_id)
        
        if not detected_plates:
            return {
                'success': False,
                'message': 'No license plates detected in the image',
                'results': []
            }
        
        # Step 2: Detect characters in each plate
        plate_results = []
        for plate in detected_plates:
            char_count = detect_characters(plate['path'], plate['name'], session_id)
            
            if char_count > 0:
                plate_results.append({
                    'plate_name': plate['name'],
                    'character_count': char_count
                })
        
        if not plate_results:
            return {
                'success': False,
                'message': 'No characters detected in the license plates',
                'results': []
            }
        
        # Step 3: Recognize characters and generate final text
        final_results = recognize_characters(session_id)
        
        return {
            'success': True,
            'message': f'Successfully processed {len(final_results)} license plate(s)',
            'results': final_results,
            'session_id': session_id
        }
        
    except Exception as e:
        # Clean up on error
        clean_temp_files(session_id)
        return {
            'success': False,
            'message': f'Error processing image: {str(e)}',
            'results': []
        }

# ----------------------------
# Flask Routes
# ----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the image through the pipeline
            result = process_image_pipeline(filepath)
            
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify(result)
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'success': False, 'message': f'Processing error: {str(e)}'})
    
    return jsonify({'success': False, 'message': 'Invalid file type'})

@app.route('/cleanup/<session_id>', methods=['POST'])
def cleanup_session(session_id):
    """Clean up temporary files for a session"""
    clean_temp_files(session_id)
    return jsonify({'success': True, 'message': 'Cleanup completed'})

if __name__ == '__main__':
    print("Starting License Plate Recognition Web Application...")
    print("Visit http://localhost:5000 to use the application")
    app.run(debug=True, host='0.0.0.0', port=5000)
