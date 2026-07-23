# Arabic License Plate Recognition System

A comprehensive, production-ready web application for automatic detection and recognition of Arabic license plates using a multi-stage deep learning pipeline.

## 🎯 Project Overview

The **Arabic License Plate Recognition System** is an end-to-end solution that combines computer vision and deep learning to automatically detect and recognize Arabic license plates from images. The system uses a three-stage pipeline: plate detection, character detection, and character recognition, specifically designed for Arabic RTL (Right-to-Left) text processing.

### Key Technologies
- **Backend**: Flask web framework
- **Computer Vision**: OpenCV, YOLO (Ultralytics)
- **Deep Learning**: PyTorch, CNN for character recognition
- **Frontend**: Modern HTML5, CSS3, JavaScript (ES6+)
- **Deployment**: Multiple production-ready options

## ✨ Features

### Core Features
- **Web Interface**: Modern, responsive web interface with drag & drop functionality
- **Complete Pipeline**: Automatic processing through plate detection → character detection → character recognition
- **Arabic RTL Support**: Proper Right-to-Left character recognition for Arabic license plates
- **Real-time Processing**: Live progress indicators and results display
- **Multi-Model Architecture**: YOLO for detection + CNN for character recognition
- **Error Handling**: Comprehensive error handling with user-friendly messages

### Advanced Features
- **Session Management**: Automatic cleanup of temporary files
- **File Upload Security**: Validation and security measures for uploaded files
- **Scalable Architecture**: Designed for both development and production environments
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## 🏗️ System Architecture

### Pipeline Overview
```
Image Upload → Plate Detection → Character Detection → Character Recognition → Results Display
      ↓              ↓                ↓                  ↓                  ↓
   Uploads      YOLO (best1.pt)  YOLO (best2.pt)    CNN (best3.pth)    Web Interface
```

### Detailed Workflow
1. **Image Upload**: User uploads image via web interface
2. **Preprocessing**: Image resizing and optimization
3. **Plate Detection**: YOLO model detects license plate regions
4. **Character Detection**: YOLO model detects individual characters within plates
5. **Character Recognition**: CNN model classifies Arabic characters
6. **RTL Processing**: Characters sorted Right-to-Left for Arabic text
7. **Result Assembly**: Final license plate text generation
8. **Display**: Results presented to user via web interface

## 📋 Prerequisites

### System Requirements
- **Python**: 3.7 or higher
- **Operating System**: Windows, Linux, or macOS
- **Memory**: Minimum 4GB RAM (8GB recommended for better performance)
- **Storage**: At least 2GB free space for models and dependencies
- **GPU** (Optional): CUDA-enabled GPU for faster processing

### Required Model Files
Ensure these files are present in the `Predection------use_this/` directory:
- `best1.pt` - YOLO plate detection model
- `best2.pt` - YOLO character detection model  
- `best3.pth` - CNN character recognition model
- `characters/` - Character dataset directory with Arabic character folders

## 🚀 Quick Start (Development)

### 1. Clone and Setup
```bash
# Navigate to your project directory
cd "c:/Users/M. ASHRAF/Desktop/رواد مصر/Project/Tests/Final_Car/Code"

# Or if using git
git clone <repository-url>
cd <repository-name>
```

### 2. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Project Structure
```bash
# Check critical files
ls -la app.py requirements.txt README.md templates/index.html

# Verify model files
ls -la Predection------use_this/best1.pt Predection------use_this/best2.pt Predection------use_this/best3.pth
```

### 4. Run Development Server
```bash
python app.py
```

### 5. Access Application
Open your browser and navigate to: `http://localhost:5000`

## 🏗️ Production Deployment Options

### Option 1: Gunicorn + Nginx (Linux/macOS - Recommended for Production)
1. Install Gunicorn: `pip install gunicorn`
2. Run with Gunicorn: `gunicorn -w 4 -b 0.0.0.0:8000 app:app`
3. Configure Nginx as reverse proxy

### Option 2: Waitress (Windows/Linux/macOS)
1. Install Waitress: `pip install waitress`
2. Run production server: `waitress-serve --port=8080 app:app`

### Option 3: Docker Deployment
1. Create a Dockerfile with Python 3.9 and required dependencies
2. Build image: `docker build -t license-plate-recognition .`
3. Run container: `docker run -p 5000:5000 license-plate-recognition`

## 📡 API Documentation

### Endpoints

#### `GET /`
- **Description**: Serve the main web interface
- **Response**: HTML page with upload interface
- **Content-Type**: `text/html`

#### `POST /upload`
- **Description**: Process uploaded image for license plate recognition
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` (required): Image file (PNG, JPG, JPEG, GIF)
- **Success Response**:
```json
{
  "success": true,
  "message": "Successfully processed 2 license plate(s)",
  "results": [
    {
      "plate_name": "plate1",
      "plate_text": "أ ب ج ١٢٣",
      "character_count": 6
    },
    {
      "plate_name": "plate2",
      "plate_text": "د ه و ٤٥٦",
      "character_count": 6
    }
  ],
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```
- **Error Response**:
```json
{
  "success": false,
  "message": "No license plates detected in the image",
  "results": []
}
```

#### `POST /cleanup/<session_id>`
- **Description**: Clean up temporary files for a session
- **Parameters**:
  - `session_id` (required): UUID from upload response
- **Response**:
```json
{
  "success": true,
  "message": "Cleanup completed"
}
```

## 📁 Project Structure

```
Arabic License Plate Recognition System/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This documentation
├── templates/
│   └── index.html        # Web interface
├── uploads/              # Temporary upload directory
├── temp/                 # Temporary processing files
├── 1_Training_Plates_detect/    # Plate detection training
│   ├── best1.pt          # Trained plate detection model
│   └── yolov12_ocr_dataset.ipynb  # Training notebook
├── 2_Training_Char_detect/      # Character detection training
│   ├── best.pt           # Trained character detection model
│   └── yolov12_ocr_dataset.ipynb  # Training notebook
├── 3_Training_CNN/              # Character recognition training
│   ├── CNN.py            # CNN model definition
│   ├── best3.pth         # Trained character recognition model
│   └── characters/       # Character dataset for training
└── Predection------use_this/    # Production inference directory
    ├── best1.pt         # Plate detection model (copy)
    ├── best2.pt         # Character detection model (copy)
    ├── best3.pth        # Character recognition model (copy)
    ├── characters/      # Character dataset (copy)
    ├── 1_plate_detect.py    # Original plate detection script
    ├── 2_char_detect.py     # Original character detection script
    ├── 3_predict_final_plates_txt.py  # Original recognition script
    ├── plates_predictions.txt  # Prediction results
    ├── vid1.mp4         # Sample video
    ├── cropped_plates/  # Detected plates (generated)
    ├── corepted_char/   # Detected characters (generated)
    └── corepted_plate_char/  # Processed characters (generated)
```

## 🤖 Model Information

### Model 1: Plate Detection (`best1.pt`)
- **Type**: YOLO (You Only Look Once) object detection
- **Purpose**: Detect license plate regions in images
- **Training Data**: Custom dataset of Arabic license plates
- **Input**: RGB image (resized to 640x640)
- **Output**: Bounding boxes for license plates

### Model 2: Character Detection (`best2.pt`)
- **Type**: YOLO object detection
- **Purpose**: Detect individual Arabic characters within license plates
- **Training Data**: Character-level annotations on license plates
- **Input**: Cropped license plate image
- **Output**: Bounding boxes for individual characters

### Model 3: Character Recognition (`best3.pth`)
- **Type**: Convolutional Neural Network (CNN)
- **Purpose**: Classify Arabic characters
- **Architecture**: Custom CNN with 3 convolutional layers
- **Training Data**: `characters/` directory with Arabic character images
- **Classes**: Arabic characters (أ, ب, ج, د, etc.) and numerals (١, ٢, ٣, etc.)
- **Input**: 64x64 grayscale character image
- **Output**: Character classification

## 🎓 Training Pipeline

### 1. Plate Detection Training
- **Location**: `1_Training_Plates_detect/`
- **Notebook**: `yolov12_ocr_dataset.ipynb`
- **Process**: YOLO training on license plate detection dataset
- **Output**: `best1.pt` model file

### 2. Character Detection Training
- **Location**: `2_Training_Char_detect/`
- **Notebook**: `yolov12_ocr_dataset.ipynb`
- **Process**: YOLO training on character-level annotations
- **Output**: `best.pt` model file (renamed to `best2.pt` for production)

### 3. Character Recognition Training
- **Location**: `3_Training_CNN/`
- **Script**: `CNN.py`
- **Dataset**: `characters/` directory with organized character images
- **Process**: CNN training on Arabic character classification
- **Output**: `best3.pth` model file

## 🔧 Configuration

### Environment Variables
```bash
# Flask Configuration
export FLASK_ENV=production
export FLASK_DEBUG=0

# Application Configuration  
export MAX_CONTENT_LENGTH=16777216  # 16MB file size limit

# Model Paths (if different from default)
export PLATE_MODEL_PATH="Predection------use_this/best1.pt"
export CHAR_MODEL_PATH="Predection------use_this/best2.pt"
export RECOGNITION_MODEL_PATH="Predection------use_this/best3.pth"
```

### Application Configuration in `app.py`
- **Upload Folder**: `uploads/` (temporary storage)
- **Temp Folder**: `temp/` (processing files)
- **Allowed Extensions**: PNG, JPG, JPEG, GIF
- **Max File Size**: 16MB
- **Image Resizing**: 1020x600 for processing

## 🔍 Troubleshooting

### Common Issues

#### 1. Model Loading Errors
```bash
# Check if model files exist
ls -la Predection------use_this/best1.pt Predection------use_this/best2.pt Predection------use_this/best3.pth

# Verify file permissions
chmod 644 Predection------use_this/best1.pt Predection------use_this/best2.pt Predection------use_this/best3.pth

# Check model paths in app.py
grep -n "best1.pt\|best2.pt\|best3.pth" app.py
```

#### 2. Dependency Issues
```bash
# Verify Python version
python --version

# Check installed packages
pip list | grep -E "flask|opencv|ultralytics|torch|pillow|numpy"

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

#### 3. File Upload Issues
```bash
# Check upload directory permissions
ls -la uploads/
chmod 755 uploads
chmod 755 temp

# Verify disk space
df -h

# Check application logs for errors
tail -f /var/log/your-app.log  # Linux/macOS
# Or check console output for Flask errors
```

#### 4. Performance Issues
- **Slow Processing**: Consider using GPU acceleration
- **Memory Issues**: Reduce image size or batch processing
- **High CPU Usage**: Optimize worker count in production

### Error Codes and Solutions
- **400 Bad Request**: Invalid file type or no file selected
- **413 Payload Too Large**: File exceeds 16MB limit
- **500 Internal Server Error**: Check application logs for details
- **ModuleNotFoundError**: Install missing dependencies from requirements.txt

## 🔒 Security Considerations

### Production Security
1. **Use HTTPS**: Always deploy behind HTTPS in production
2. **File Upload Security**: 
   - Validate file types server-side
   - Scan uploaded files for malware
   - Limit file size (currently 16MB)
   - Store uploads outside web root
3. **Environment Security**:
   - Don't run as root user
   - Use virtual environments
   - Keep dependencies updated
   - Use environment variables for secrets

### Network Security
```bash
# Configure firewall (Linux)
ufw allow 80
ufw allow 443
ufw enable

# Use reverse proxy (Nginx) for additional security
```

## 📊 Performance & Scaling

### Expected Performance
- **Processing Time**: 2-10 seconds per image (depending on hardware)
- **Concurrent Users**: 10-50 users (depending on server resources)
- **Memory Usage**: 1-2GB per worker process
- **CPU Usage**: Moderate to high during image processing

### Hardware Recommendations
- **Development**: 4GB RAM, 2-core CPU
- **Production**: 8GB+ RAM, 4-core CPU, SSD storage
- **High Traffic**: 16GB+ RAM, 8-core CPU, GPU acceleration

### Scaling Strategies
1. **Vertical Scaling**: Increase server resources (RAM, CPU)
2. **Horizontal Scaling**: Deploy multiple instances behind load balancer
3. **GPU Acceleration**: Use CUDA-enabled PyTorch for faster inference
4. **Caching**: Implement Redis for session management

## 🤝 Contributing & Development

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Testing
```bash
# Run basic functionality test
python -c "from app import app; print('App imports successfully')"

# Test model loading
python -c "
from ultralytics import YOLO
import torch
print('YOLO import successful')
print('PyTorch version:', torch.__version__)
"

# Test web server
python app.py &
sleep 2
curl http://localhost:5000
pkill -f "python app.py"
```

### Code Structure Guidelines
- Keep business logic in `app.py`
- Maintain separate training and inference directories
- Use descriptive variable and function names
- Add comments for complex logic
- Follow PEP 8 style guidelines

## 📄 License

This project is for educational and demonstration purposes. Please ensure you have appropriate licenses for commercial use of the models and datasets.

## 📞 Support

For deployment issues:
1. Check the troubleshooting section above
2. Verify all model files are present
3. Ensure Python version compatibility
4. Check system resource availability

---
## 📞 Team Memmbers
1. Mohammed Momen
2. Ashraf Mohammed
3. Zead Ahmed
4. Omar Shaban


**Ready to deploy?** Start with the [Quick Start](#-quick-start-development) section for development or choose a [Production Deployment](#-production-deployment-options) option for live environments.
