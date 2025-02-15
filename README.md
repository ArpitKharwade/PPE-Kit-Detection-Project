# 🦺 PPE Kit Detection using Deep Learning

A deep learning-based project that detects **Personal Protective Equipment (PPE)** in real-time using **YOLO**, **OpenCV**, and **Python**. This model helps ensure workplace safety by identifying whether individuals are wearing necessary safety gear.

## 🚀 Features
- 🎯 **Real-time PPE Detection**: Identifies safety gear such as helmets, vests, gloves, and masks.
- 📷 **Works on Live Video & Images**: Supports both video stream and image processing.
- ⚡ **High-Speed & Accuracy**: Uses **YOLO (You Only Look Once)** for efficient object detection.
- 🔍 **Customizable Model**: Train on additional PPE items or improve accuracy with more data.
- 📊 **Bounding Boxes & Labels**: Visualizes detected PPE components in real-time.

## 🛠️ Technologies Used
- **Python** - Core programming language
- **YOLO (You Only Look Once)** - Object detection algorithm
- **OpenCV** - Image and video processing
- **Deep Learning Framework** - TensorFlow / PyTorch (depending on implementation)
- **VS Code** - Development environment

## 📂 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ArpitKharwade/ppe-detection.git
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Download YOLO Weights
Download pre-trained YOLO weights from the [official YOLO repository](https://pjreddie.com/darknet/yolo/).

### 4️⃣ Run the Application
To detect PPE in an image:
```bash
python detect_ppe.py --image path/to/image.jpg
```
To detect PPE in a video:
```bash
python detect_ppe.py --video path/to/video.mp4
```
To run on a live webcam:
```bash
python detect_ppe.py --webcam
```

## 🛠️ Customization
- Modify `config.py` to change YOLO parameters and thresholds.
- Train your own YOLO model on custom PPE datasets.
- Update `detect_ppe.py` to enhance detection accuracy.


<!-- ## Project Structure 
 
```
/c:/jupyter/Infosys Intern Project/Detect-the-PPE-Kit-in-Automobile-Manufacturing-Project/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│
├── README.md
└── requirements.txt
``` -->


## 📜 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🤝 Contributing
Contributions are welcome! Feel free to fork this repository and submit a pull request.

## 📧 Contact
For any questions or feedback, reach out via [LinkedIn](https://www.linkedin.com/in/arpit-kharwade/) or email at `arpitkharwade2004@gmail.com`.

---
🚀 Ensure workplace safety with real-time PPE detection! 🦺


