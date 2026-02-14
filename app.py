import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = os.path.join("models", "final_model.keras")

EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral',
    7: 'Contempt'
}

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_face(face_img):
    try:
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
            
        resized = cv2.resize(gray, (48, 48))
        normalized = resized / 255.0
        expanded = np.expand_dims(normalized, axis=0)
        final_input = np.expand_dims(expanded, axis=-1)
        
        return final_input
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

def predict_emotion(image):
    if model is None:
        return image, "Model not loaded!"
    
    if image is None:
        return None, "No image"

    output_image = image.copy()
    gray_full = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray_full, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        return output_image, "No face detected"

    results = {}
    
    for (x, y, w, h) in faces:
        roi_face = output_image[y:y+h, x:x+w]
        model_input = preprocess_face(roi_face)
        
        if model_input is not None:
            prediction = model.predict(model_input)
            label_index = np.argmax(prediction)
            confidence = np.max(prediction)
            label_text = EMOTION_LABELS.get(label_index, "Unknown")
            
            results[label_text] = float(confidence)

            color = (0, 255, 0) if label_index == 3 else (255, 0, 0)
            cv2.rectangle(output_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(output_image, f"{label_text} ({confidence:.2f})", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return output_image, results

def create_demo():
    with gr.Blocks(title="Facial Expression Recognition") as demo:
        gr.Markdown("# AI Emotion Recognition Demo")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=["webcam", "upload"], label="Input Image")
                submit_btn = gr.Button("Analyze Emotion", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Detected Face")
                output_label = gr.Label(num_top_classes=3, label="Probabilities")
        
        submit_btn.click(
            fn=predict_emotion, 
            inputs=input_image, 
            outputs=[output_image, output_label]
        )
        
    return demo

if __name__ == "__main__":
    print("Starting Gradio in Standalone Mode...")
    demo_app = create_demo()
    demo_app.launch(share=True) 

def run_from_main():
    print("Starting Gradio from Main Script...")
    demo_app = create_demo()
    demo_app.launch()