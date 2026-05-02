import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os
import pickle
from skimage.feature import local_binary_pattern
from skimage import img_as_ubyte

# Function to detect face using Haar Cascade
def detect_face(image):
    # Load Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    # Return the first face (largest)
    x, y, w, h = faces[0]
    face_roi = image[y:y+h, x:x+w]
    return face_roi, (x, y, w, h)

# Function to extract features from face
def extract_features(face_roi):
    # Convert to grayscale for some features
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # 1. Texture Features using LBP
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalize
    
    # 2. Image Sharpness using Laplacian Variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 3. Color Features (RGB mean and variance)
    r, g, b = cv2.split(face_roi)
    rgb_mean = [np.mean(r), np.mean(g), np.mean(b)]
    rgb_var = [np.var(r), np.var(g), np.var(b)]
    
    # 4. Edge Density using Canny
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Combine all features
    features = np.concatenate([lbp_hist, [laplacian_var], rgb_mean, rgb_var, [edge_density]])
    return features

# Function to load dataset
def load_dataset(dataset_path):
    features = []
    labels = []
    for label, folder in enumerate(['fake', 'real']):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(folder_path, filename)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                face_roi = detect_face(image)
                if face_roi is None:
                    continue
                face_roi = face_roi[0]  # Get the ROI
                feat = extract_features(face_roi)
                features.append(feat)
                labels.append(label)  # 0 for fake, 1 for real
    return np.array(features), np.array(labels)

# Function to train model
def train_model(dataset_path):
    features, labels = load_dataset(dataset_path)
    if len(features) == 0:
        print("No data found. Please check dataset structure.")
        return None, None
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)
    
    # Train SVM
    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(X_train, y_train)
    
    # Predict and print accuracy
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Save model and scaler
    with open('antispoof_model.pkl', 'wb') as f:
        pickle.dump((svm, scaler), f)
    
    return svm, scaler

# Function to predict on a single image
def predict_image(image_path, model, scaler):
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Unable to load image."
    
    result = detect_face(image)
    if result is None:
        return "No face detected."
    
    face_roi, (x, y, w, h) = result
    features = extract_features(face_roi)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    
    # Draw bounding box and text
    if prediction == 1:
        text = "REAL FACE"
        color = (0, 255, 0)  # Green
    else:
        text = "SPOOF ATTACK"
        color = (0, 0, 255)  # Red
    
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Display image
    cv2.imshow('Face Anti-Spoofing', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return text

# Main function
if __name__ == "__main__":
    dataset_path = 'dataset'  # Path to dataset folder
    
    # Train model
    print("Training model...")
    model, scaler = train_model(dataset_path)
    if model is None:
        exit()
    
    # Example prediction
    test_image_path = 'test_image.jpg'  # Replace with actual test image path
    print("Predicting on test image...")
    result = predict_image(test_image_path, model, scaler)
    print(f"Prediction: {result}")