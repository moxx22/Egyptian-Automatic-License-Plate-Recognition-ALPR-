# Egyptian Automatic License Plate Recognition (ALPR)

Welcome to the Egyptian ALPR System! This project provides a complete end-to-end pipeline for detecting and recognizing Egyptian license plates from images. It uses state-of-the-art YOLO models for detection and a custom Convolutional Neural Network (CNN) for Arabic character recognition, all wrapped in a user-friendly Flask web interface.

## 🚀 Features

*   **License Plate Detection:** Accurately locates license plates within an image using a trained YOLO model.
*   **Character Segmentation:** Isolates individual Arabic characters on the detected license plate using a secondary YOLO model.
*   **Character Recognition (OCR):** Predicts the specific Arabic letter or number using a custom-trained PyTorch CNN model.
*   **Right-to-Left Support:** Automatically sorts and reads detected Arabic characters correctly from right to left.
*   **Web Interface:** Easy-to-use Flask web application to upload images and instantly view the ALPR results.

## 🛠️ Technology Stack

*   **Backend:** Python, Flask
*   **Object Detection:** Ultralytics YOLO (v12/v8 format)
*   **Deep Learning (OCR):** PyTorch, Torchvision
*   **Image Processing:** OpenCV (cv2), Pillow (PIL)
*   **Frontend:** HTML/CSS (Jinja2 Templates)

## 🧠 How the Pipeline Works

1.  **Upload:** The user uploads a vehicle image via the web interface.
2.  **Plate Detection (`best1.pt`):** The first YOLO model scans the image and extracts the cropped license plate.
3.  **Character Detection (`best2.pt`):** The second YOLO model analyzes the cropped plate to locate individual characters (letters and numbers) and draws bounding boxes around them. Non-Maximum Suppression (NMS) is applied to remove overlapping duplicate boxes.
4.  **Character Recognition (`best3.pth`):** Each cropped character is converted to grayscale, resized, and passed through the custom PyTorch CNN model to predict the exact character.
5.  **Result Compilation:** The recognized characters are sorted from right-to-left (standard for Arabic) and displayed back to the user on the web interface.

## 💻 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/moxx22/Egyptian-Automatic-License-Plate-Recognition-ALPR-.git
    cd Egyptian-Automatic-License-Plate-Recognition-ALPR-
    ```

2.  **Install dependencies:**
    Ensure you have Python installed, then install the required packages.
    ```bash
    pip install -r Code/requirements.txt
    ```
    *(Note: You may need to install specific PyTorch versions depending on your hardware/CUDA setup).*

3.  **Run the application:**
    Navigate to the `Code` directory and start the Flask server:
    ```bash
    cd Code
    python app.py
    ```

4.  **Access the Web App:**
    Open your web browser and go to `http://localhost:5000`

## 📂 Project Structure

*   `Code/app.py`: The main Flask application and ALPR pipeline logic.
*   `Code/templates/index.html`: The frontend web interface.
*   `Code/Predection------use_this/`: Contains the trained models and character class definitions.
    *   `best1.pt`: YOLO model for Plate Detection.
    *   `best2.pt`: YOLO model for Character Detection.
    *   `best3.pth`: PyTorch CNN model weights for Character Recognition.
*   `Presentation/`: Documentation, diagrams, and evaluation metrics for the models.

## ⚠️ Notes
* Large model files (`*.pt`, `*.pth`) and video demos (`*.mp4`) are excluded from this Git repository due to size limits. You must place your trained weights in the `Code/Predection------use_this/` directory for the application to function.
