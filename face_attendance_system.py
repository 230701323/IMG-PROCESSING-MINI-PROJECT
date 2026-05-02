import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os
import pickle
from skimage.feature import local_binary_pattern
import csv
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ===================== SYNTHETIC DATA GENERATION =====================
def generate_synthetic_faces():
    """Generate synthetic face images for training."""
    print("\n=== GENERATING SYNTHETIC TRAINING DATA ===")
    
    os.makedirs('dataset/real', exist_ok=True)
    os.makedirs('dataset/fake', exist_ok=True)
    os.makedirs('dataset_faces/person1', exist_ok=True)
    os.makedirs('dataset_faces/person2', exist_ok=True)
    
    # Generate fake (spoof) images - blurry, low contrast
    print("Generating FAKE face images...")
    for i in range(20):
        fake_img = np.random.randint(100, 150, (128, 128, 3), dtype=np.uint8)
        fake_img = cv2.GaussianBlur(fake_img, (15, 15), 0)
        cv2.imwrite(f'dataset/fake/fake_{i}.jpg', fake_img)
    
    # Generate real face images - high contrast, sharp
    print("Generating REAL face images...")
    for i in range(20):
        real_img = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        real_img[40:90, 40:90] = np.random.randint(100, 150, (50, 50, 3), dtype=np.uint8)
        real_img = cv2.Canny(real_img, 50, 150)
        real_img = cv2.cvtColor(real_img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f'dataset/real/real_{i}.jpg', real_img)
    
    # Generate person1 images
    print("Generating PERSON1 face images...")
    for i in range(20):
        person_img = np.zeros((128, 128, 3), dtype=np.uint8)
        person_img[:, :] = [120, 100, 80]  # Skin tone
        cv2.circle(person_img, (64, 50), 20, [100, 80, 60], -1)  # Head
        cv2.circle(person_img, (54, 45), 5, [50, 40, 30], -1)  # Left eye
        cv2.circle(person_img, (74, 45), 5, [50, 40, 30], -1)  # Right eye
        cv2.imwrite(f'dataset_faces/person1/person1_{i}.jpg', person_img)
    
    # Generate person2 images
    print("Generating PERSON2 face images...")
    for i in range(20):
        person_img = np.zeros((128, 128, 3), dtype=np.uint8)
        person_img[:, :] = [140, 110, 90]  # Different skin tone
        cv2.circle(person_img, (64, 60), 22, [110, 85, 65], -1)  # Head
        cv2.circle(person_img, (52, 52), 6, [60, 45, 35], -1)  # Left eye
        cv2.circle(person_img, (76, 52), 6, [60, 45, 35], -1)  # Right eye
        cv2.imwrite(f'dataset_faces/person2/person2_{i}.jpg', person_img)
    
    print("✓ Synthetic data generated successfully!")

# ===================== FACE DETECTION =====================
def detect_face(image):
    """Detect face in image using Haar Cascade."""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        
        if len(faces) == 0:
            return None
        
        # Return the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        # Add padding to face ROI
        padding = 10
        y_start = max(0, y - padding)
        x_start = max(0, x - padding)
        y_end = min(image.shape[0], y + h + padding)
        x_end = min(image.shape[1], x + w + padding)
        
        face_roi = image[y_start:y_end, x_start:x_end]
        return face_roi, (x_start, y_start, x_end - x_start, y_end - y_start)
    except Exception as e:
        print(f"Face detection error: {e}")
        return None

# ===================== LIVENESS FEATURES =====================
def extract_liveness_features(face_roi):
    """Extract features for liveness detection."""
    try:
        # Resize face for consistency
        face_roi = cv2.resize(face_roi, (128, 128))
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # 1. LBP Texture
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-7)
        
        # 2. Laplacian Variance (Sharpness)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 3. Color Features
        r, g, b = cv2.split(face_roi)
        rgb_mean = [np.mean(r), np.mean(g), np.mean(b)]
        rgb_var = [np.var(r), np.var(g), np.var(b)]
        
        # 4. Edge Density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 5. Contrast
        contrast = gray.std()
        
        # Combine all features
        features = np.concatenate([lbp_hist, [laplacian_var], rgb_mean, rgb_var, [edge_density, contrast]])
        return features
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

# ===================== LIVENESS MODEL TRAINING =====================
def load_liveness_dataset(dataset_path):
    """Load liveness dataset."""
    features = []
    labels = []
    
    for label, folder in enumerate(['fake', 'real']):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            continue
        
        images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Loading {len(images)} {folder} images...")
        
        for filename in images:
            try:
                img_path = os.path.join(folder_path, filename)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                result = detect_face(image)
                if result is None:
                    continue
                
                face_roi = result[0]
                feat = extract_liveness_features(face_roi)
                if feat is not None:
                    features.append(feat)
                    labels.append(label)
            except Exception as e:
                continue
    
    return np.array(features) if features else np.array([]), np.array(labels) if labels else np.array([])

def train_liveness_model(dataset_path):
    """Train liveness detection model."""
    print("\n=== TRAINING LIVENESS MODEL ===")
    features = []
    labels = []
    
    for label, folder in enumerate(['fake', 'real']):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            continue
        
        images = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))])[:20]
        print(f"Loading {len(images)} {folder} images...")
        
        for filename in images:
            try:
                img = cv2.imread(os.path.join(folder_path, filename))
                if img is None:
                    continue
                result = detect_face(img)
                if result is None:
                    result = (img, None)
                face_roi = result[0] if result else img
                feat = extract_liveness_features(face_roi)
                if feat is not None:
                    features.append(feat)
                    labels.append(label)
            except:
                continue
    
    if len(features) < 2:
        print("❌ Failed to load liveness data")
        return None, None
    
    print(f"✓ Loaded {len(features)} images for liveness training")
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    svm.fit(features_scaled, labels)
    
    with open('antispoof_model.pkl', 'wb') as f:
        pickle.dump((svm, scaler), f)
    
    print("✓ Liveness model trained!")
    return svm, scaler

# ===================== FACE RECOGNITION TRAINING =====================
def load_recognition_dataset(dataset_path):
    """Load face recognition dataset."""
    faces = []
    labels = []
    label_dict = {}
    label_id = 0
    
    for person in sorted(os.listdir(dataset_path)):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue
        
        label_dict[label_id] = person
        images = [f for f in os.listdir(person_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Loading {len(images)} images for {person}...")
        
        for filename in images:
            try:
                img_path = os.path.join(person_path, filename)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                result = detect_face(image)
                if result is None:
                    continue
                
                face_roi = result[0]
                face_roi = cv2.resize(face_roi, (128, 128))
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                faces.append(gray_face)
                labels.append(label_id)
            except Exception as e:
                continue
        
        label_id += 1
    
    return faces, labels, label_dict

def train_face_recognizer(dataset_path):
    """Train face recognition model."""
    print("\n=== TRAINING FACE RECOGNIZER ===")
    faces = []
    labels = []
    label_dict = {}
    label_id = 0
    
    for person in sorted(os.listdir(dataset_path)):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue
        
        label_dict[label_id] = person
        person_faces = []
        
        images = sorted(os.listdir(person_path))[:20]
        for filename in images:
            try:
                img = cv2.imread(os.path.join(person_path, filename))
                if img is None:
                    continue
                result = detect_face(img)
                if result is None:
                    result = (img, None)
                face_roi = result[0] if result else img
                face_roi = cv2.resize(face_roi, (128, 128))
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                faces.append(gray_face)
                person_faces.append(gray_face)
                labels.append(label_id)
            except:
                continue
        
        print(f"✓ Loaded {len(person_faces)} images for {person}")
        label_id += 1
    
    if len(faces) < 2:
        print("❌ Failed to load recognition data")
        return None, None
    
    print(f"✓ Total faces loaded: {len(faces)}")
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save('face_recognizer.yml')
    
    with open('label_dict.pkl', 'wb') as f:
        pickle.dump(label_dict, f)
    
    print("✓ Face recognizer trained!")
    return recognizer, label_dict

# ===================== FACE RECOGNITION =====================
def recognize_face(face_roi, recognizer, label_dict):
    """Recognize face and return person name."""
    try:
        face_roi = cv2.resize(face_roi, (128, 128))
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        label, confidence = recognizer.predict(gray_face)
        
        if confidence < 70:  # Low confidence = good match
            name = label_dict.get(label, "Unknown")
            return name
        return "Unknown"
    except Exception as e:
        return "Unknown"

# ===================== ATTENDANCE =====================
def mark_attendance(name):
    """Mark attendance in CSV file."""
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        now = datetime.now().strftime('%H:%M:%S')
        filename = 'attendance.csv'
        
        # Create CSV header if file doesn't exist
        if not os.path.exists(filename):
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Date', 'Time'])
        
        # Check if already marked today
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2 and row[0] == name and row[1] == today:
                    return "Already Marked"
        
        # Mark attendance
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, today, now])
        
        return "✓ Marked"
    except Exception as e:
        return "Error"

# ===================== REAL-TIME SYSTEM =====================
def run_attendance_system(liveness_model, liveness_scaler, recognizer, label_dict):
    """Run real-time attendance system."""
    print("\n=== STARTING REAL-TIME SYSTEM ===")
    print("Press 'q' to quit\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam - Using demo mode with synthetic frames")
        demo_mode = True
    else:
        demo_mode = False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    
    while True:
        if demo_mode:
            # Create demo frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :] = [200, 150, 100]
            # Draw fake face
            cv2.circle(frame, (320, 240), 60, [150, 120, 90], -1)
            cv2.circle(frame, (300, 220), 10, [50, 40, 30], -1)
            cv2.circle(frame, (340, 220), 10, [50, 40, 30], -1)
            ret = True
        else:
            ret, frame = cap.read()
            if not ret:
                break
        
        display_frame = frame.copy()
        result = detect_face(frame)
        
        if result is None:
            cv2.putText(display_frame, "No Face Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(display_frame, "Press 'q' to quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            face_roi, (x, y, w, h) = result
            features = extract_liveness_features(face_roi)
            
            if features is not None:
                features_scaled = liveness_scaler.transform([features])
                liveness_pred = liveness_model.predict(features_scaled)[0]
                
                if liveness_pred == 1:
                    status = "✓ REAL FACE"
                    color = (0, 255, 0)
                    name = recognize_face(face_roi, recognizer, label_dict)
                    
                    if name != "Unknown":
                        attendance = mark_attendance(name)
                    else:
                        attendance = "Unknown Person"
                        name = "Unknown"
                else:
                    status = "✗ SPOOF ATTACK"
                    color = (0, 0, 255)
                    name = ""
                    attendance = ""
                
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)
                cv2.putText(display_frame, status, (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                if name and name != "Unknown":
                    cv2.putText(display_frame, f"Name: {name}", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                if attendance:
                    cv2.putText(display_frame, f"Attendance: {attendance}", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(display_frame, "Press 'q' to quit", (10, display_frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('Face Attendance System with Anti-Spoofing', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if not demo_mode:
        cap.release()
    cv2.destroyAllWindows()

# ===================== MAIN =====================
if __name__ == "__main__":
    try:
        print("=" * 70)
        print("FACE RECOGNITION ATTENDANCE SYSTEM WITH ANTI-SPOOFING")
        print("=" * 70)
        
        # Generate synthetic data if not exists
        if not os.path.exists('dataset/real') or len(os.listdir('dataset/real')) == 0:
            generate_synthetic_faces()
        
        # Train liveness model
        liveness_model, liveness_scaler = train_liveness_model('dataset')
        if liveness_model is None:
            print("❌ Failed to train liveness model")
            exit()
        
        # Train face recognizer
        recognizer, label_dict = train_face_recognizer('dataset_faces')
        if recognizer is None or label_dict is None:
            print("❌ Failed to train face recognizer")
            exit()
        
        # Run system
        run_attendance_system(liveness_model, liveness_scaler, recognizer, label_dict)
        print("\n✓ System closed. Attendance saved in attendance.csv")
    
    except KeyboardInterrupt:
        print("\n✓ System stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()