import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt 
def create_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

try:
    model = load_model("model.h5")
    print("Model loaded successfully.")
except:
    print("No pre-trained model found. Training a new model...")
    model = create_model()
def train_model():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_data = datagen.flow_from_directory('train/', target_size=(64, 64), batch_size=32, class_mode='binary', subset='training')
    val_data = datagen.flow_from_directory('train/', target_size=(64, 64), batch_size=32, class_mode='binary', subset='validation')
    
    history = model.fit(train_data, validation_data=val_data, epochs=10)
    model.save("gender_model.h5")
    return history
def webcam_gender_detection():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gender_labels = ["Male", "Female"]
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (64,64))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)
            
            prediction = model.predict(face)[0]
            gender = gender_labels[int(prediction > 0.5)]
            color = (0, 255, 0) if gender == "Male" else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow("Gender Classification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
webcam_gender_detection()
# Display the live video feed and the classification results
def app():
    webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

    if webrtc_ctx.video_processor:
        class_label = st.empty()
        elapsed_time = st.empty()

        while True:
            class_label.write(f"Predicted class: {webrtc_ctx.video_processor.class_label}")
            elapsed_time.write(f"Time taken: {webrtc_ctx.video_processor.elapsed_time:.4f} seconds")
            time.sleep(0.1)

app()