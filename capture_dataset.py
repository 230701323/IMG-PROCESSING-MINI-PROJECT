import cv2
import os
from datetime import datetime

def capture_dataset():
    """Capture faces from webcam to create dataset."""
    
    # Create directories if they don't exist
    os.makedirs('dataset/real', exist_ok=True)
    os.makedirs('dataset/fake', exist_ok=True)
    os.makedirs('dataset_faces/person1', exist_ok=True)
    os.makedirs('dataset_faces/person2', exist_ok=True)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    print("=" * 60)
    print("DATASET CAPTURE UTILITY")
    print("=" * 60)
    
    # Capture real faces
    print("\n📸 CAPTURING REAL FACES for dataset/real/")
    print("Press SPACE to capture, 'c' when done (capture 10-15 images)")
    count_real = 0
    while count_real < 15:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.putText(frame, f"Real Faces: {count_real}/15", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE: capture | 'c': done", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        cv2.imshow('Capture Dataset', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and len(faces) > 0:
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]
            filename = f'dataset/real/real_{count_real}_{datetime.now().strftime("%H%M%S")}.jpg'
            cv2.imwrite(filename, face)
            print(f"✓ Saved: {filename}")
            count_real += 1
        elif key == ord('c'):
            break
    
    print("\n" + "="*60)
    print("📸 CAPTURING PERSON 1 FACE for dataset_faces/person1/")
    print("Press SPACE to capture, 'c' when done (capture 10-15 images)")
    count_person1 = 0
    while count_person1 < 15:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.putText(frame, f"Person1 Faces: {count_person1}/15", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE: capture | 'c': done", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        cv2.imshow('Capture Dataset', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and len(faces) > 0:
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]
            filename = f'dataset_faces/person1/person1_{count_person1}_{datetime.now().strftime("%H%M%S")}.jpg'
            cv2.imwrite(filename, face)
            print(f"✓ Saved: {filename}")
            count_person1 += 1
        elif key == ord('c'):
            break
    
    print("\n" + "="*60)
    print("📸 CAPTURING PERSON 2 FACE for dataset_faces/person2/")
    print("Press SPACE to capture, 'c' when done (capture 10-15 images)")
    count_person2 = 0
    while count_person2 < 15:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.putText(frame, f"Person2 Faces: {count_person2}/15", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE: capture | 'c': done", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        cv2.imshow('Capture Dataset', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and len(faces) > 0:
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]
            filename = f'dataset_faces/person2/person2_{count_person2}_{datetime.now().strftime("%H%M%S")}.jpg'
            cv2.imwrite(filename, face)
            print(f"✓ Saved: {filename}")
            count_person2 += 1
        elif key == ord('c'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("✓ CAPTURE COMPLETE!")
    print(f"✓ Real faces saved: {count_real}")
    print(f"✓ Person1 faces saved: {count_person1}")
    print(f"✓ Person2 faces saved: {count_person2}")
    print("\nNow run: python face_attendance_system.py")
    print("="*60)

if __name__ == "__main__":
    capture_dataset()
