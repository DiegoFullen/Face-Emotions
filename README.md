# Facial Emotion Recognition using MediaPipe and KNN
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-yellowgreen)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-red)

A computer vision system that detects emotions in real-time using facial landmarks analysis with MediaPipe and K-Nearest Neighbors classification.

## Key Features
- **Real-time emotion detection** from webcam feed
- **Modular architecture** for easy maintenance
- **Custom-trained KNN model** with 5 emotion classes
- **Lightweight solution** (under 50MB model size)
- **Privacy-focused** - no data storage

## Project Structure

/emotion-recognition-mediapipe-knn/  
  │  
  ├── datasets/               # CSV with data of emotions and users  
  ├── feature_extraction/     # Landmark processing and feature generation  
  ├── training/               # KNN model training  
  ├── prediction/             # Emotion prediction logic  
  ├── main.py                 # Entry point (or individual scripts)  
  ├── requirements.txt        # Dependencies  
  └── README.md  


## How to Run
### 1. Clone repository
```bash
git clone https://github.com/DiegoFullen/Face-Emotions.git
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the system
```bash
python main.py
```
(For separate execution: python training/train_knn.py then python prediction/predict_emotion.py)


## Emotions Supported
- Happy 😄
- Sad 😢
- Angry 😠
- Surprised 😯
- Neutral 😐


## Technical Stack
Core libraries used:
- Machine Learning: scikit-learn, joblib, mediapipe
- Computer Vision: OpenCV
- Data Processing: pandas, numpy
- Visualization: matplotlib, seaborn


## Technologies used
- Python
- MediaPipe
- scikit-learn
- OpenCV


## Notes
- No personal data is used or stored.
- No GUI implemented yet – console interface only.
*Specific versions pinned for reproducibility (see requirements.txt)*


## Project Context
Schoolar project developed for Machine Learning - CETI Colomos  
*Not intended for commercial use* 


## Future Roadmap
- Implement GUI interface.

## Author
Diego Salvador Candia Fullen
