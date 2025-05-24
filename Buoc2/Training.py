import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

class IdentityMetadata:
    def __init__(self, base, name, file):
        self.base = base
        self.name = name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        full_path = os.path.normpath(os.path.join(self.base, self.name, self.file))
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Image file not found: {full_path}")
        return full_path

def load_metadata(path):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")

    metadata = []
    identity_counts = {}
    for name in sorted(os.listdir(path)):
        dir_path = os.path.join(path, name)
        if not os.path.isdir(dir_path):
            continue
        image_files = [f for f in sorted(os.listdir(dir_path)) if f.lower().endswith(('.jpg', '.jpeg', '.bmp'))]
        if image_files:  # Only add identity if it has valid images
            identity_counts[name] = len(image_files)
            for file in image_files:
                metadata.append(IdentityMetadata(path, name, file))
    
    print(f"Found {len(identity_counts)} identities: {identity_counts}")
    return np.array(metadata)

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img[..., ::-1]

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def show_pair(idx1, idx2, metadata, embedded):
    try:
        assert 0 <= idx1 < len(metadata) and 0 <= idx2 < len(metadata)
        path1 = metadata[idx1].image_path()
        path2 = metadata[idx2].image_path()

        img1 = load_image(path1)
        img2 = load_image(path2)

        plt.figure(figsize=(8, 3))
        plt.suptitle(f'Distance = {distance(embedded[idx1], embedded[idx2]):.2f}')
        plt.subplot(121)
        plt.imshow(img1)
        plt.title(metadata[idx1].name)
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(img2)
        plt.title(metadata[idx2].name)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error in show_pair: {e}")

def extract_embeddings(metadata, recognizer):
    embedded = np.zeros((len(metadata), 128))
    valid_indices = []
    for i, m in enumerate(metadata):
        try:
            print(f"[{i+1}/{len(metadata)}] Loading: {m.image_path()}")
            img = load_image(m.image_path())
            face_feature = recognizer.feature(img)
            embedded[i] = face_feature
            valid_indices.append(i)
        except Exception as e:
            print(f"Skipping {m}: {e}")
    if not valid_indices:
        raise ValueError("No valid embeddings extracted. Check image files or face recognition model.")
    return embedded[valid_indices], valid_indices

def main():
    # Load models
    try:
        detector = cv2.FaceDetectorYN.create(
            "C:\\Users\\trung\\OneDrive\\Documents\\project\\train_face\\model\\face_detection_yunet_2023mar.onnx", "", (320, 320), 0.9, 0.3, 5000
        )
        recognizer = cv2.FaceRecognizerSF.create(
            "C:\\Users\\trung\\OneDrive\\Documents\\project\\train_face\\model\\face_recognition_sface_2021dec.onnx", ""
        )
        detector.setInputSize((320, 320))
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Load metadata
    try:
        metadata = load_metadata('C:\\Users\\trung\\OneDrive\\Documents\\project\\train_face\\image')
        if len(metadata) == 0:
            print("Error: No valid images found in the specified directory.")
            print("Please ensure the directory contains subdirectories with valid .jpg, .jpeg, or .bmp images.")
            return
        print(f"Loaded {len(metadata)} images.")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    # Check identities
    targets = np.array([m.name for m in metadata])
    unique_identities = np.unique(targets)
    if len(unique_identities) < 2:
        print(f"Error: Found only {len(unique_identities)} unique identity(ies): {unique_identities}")
        print("At least two unique identities are required for training. Please add more identity folders with images.")
        return
    print(f"Found {len(unique_identities)} unique identities: {unique_identities}")

    # Extract embeddings
    try:
        embedded, valid_indices = extract_embeddings(metadata, recognizer)
        if len(valid_indices) == 0:
            print("Error: No valid embeddings generated.")
            return
        print(f"Extracted embeddings for {len(valid_indices)} images.")
        metadata = metadata[valid_indices]  # Update metadata to include only valid samples
        targets = np.array([m.name for m in metadata])  # Update targets
    except Exception as e:
        print(f"Error extracting embeddings: {e}")
        return

    # Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(targets)

    # Split data
    idx = np.arange(len(metadata))
    train_idx = idx % 5 != 0
    test_idx = idx % 5 == 0
    X_train, X_test = embedded[train_idx], embedded[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Check if training data is sufficient
    if len(X_train) == 0:
        print("Error: No training samples available after split. Ensure there are enough valid images.")
        return
    if len(X_test) == 0:
        print("Warning: No test samples available. Evaluation will be skipped.")

    # Train classifier
    print("Training Linear SVM...")
    svc = LinearSVC()
    svc.fit(X_train, y_train)

    # Evaluate
    if len(X_test) > 0:
        acc = accuracy_score(y_test, svc.predict(X_test))
        print(f"SVM accuracy: {acc:.6f}")
    else:
        print("Skipping evaluation due to insufficient test samples.")

    # Save model
    try:
        joblib.dump(svc, 'C:\\Users\\trung\\OneDrive\\Documents\\project\\train_face\\model\\svc.pkl')
        print("Model saved to ../model/svc.pkl")
    except Exception as e:
        print(f"Error saving model: {e}")

    # Optionally: Show a test pair
    if len(metadata) >= 2:
        show_pair(0, 1, metadata, embedded)
    else:
        print("Not enough valid images to display a pair.")

if __name__ == "__main__":
    main()