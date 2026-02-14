import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = os.path.join("models", "final_model.keras")

EMOTION_LABELS = {
    0: 'Angry',
    1: 'Contempt',   
    2: 'Disgust',    
    3: 'Fear',         
    4: 'Happy',     
    5: 'Neutral',     
    6: 'Sad',        
    7: 'Surprise'    
}

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    print("Model & Cascade loaded successfully.")
except Exception as e:
    print(f"Error loading system: {e}")
    model = None

def preprocess_for_model(roi_gray):
    try:
        if roi_gray is None or roi_gray.size == 0:
            return None
        resized = cv2.resize(roi_gray, (48, 48))
        normalized = resized / 255.0
        final_input = np.expand_dims(normalized, axis=0)
        final_input = np.expand_dims(final_input, axis=-1)
        return final_input
    except Exception as e:
        print(f"Preprocessing Error: {e}")
        return None

def predict_emotion(image):
    if model is None:
        return image, {"Error: Model not loaded": 0.0}
    
    if image is None:
        return None, None

    try:
        output_image = image.copy()
        
        if len(image.shape) == 3 and image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            output_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        
        results = {}
        processed_flag = False

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                model_input = preprocess_for_model(roi_gray)
                if model_input is not None:
                    prediction = model.predict(model_input, verbose=0)
                    processed_flag = True
                    
                    best_idx = np.argmax(prediction)
                    confidence = np.max(prediction)
                    best_label = EMOTION_LABELS.get(best_idx, "Unknown")

                    if not results:
                        for i, prob in enumerate(prediction[0]):
                            lbl = EMOTION_LABELS.get(i, f"Class {i}")
                            results[lbl] = float(prob)

                    color = (0, 255, 0) if best_idx == 4 else (255, 0, 0)
                    cv2.rectangle(output_image, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(output_image, f"{best_label} ({confidence:.2f})", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        if len(faces) == 0 or not processed_flag:
            cv2.putText(output_image, "Full Frame Analysis", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            model_input = preprocess_for_model(gray)
            if model_input is not None:
                prediction = model.predict(model_input, verbose=0)
                for i, prob in enumerate(prediction[0]):
                    lbl = EMOTION_LABELS.get(i, f"Class {i}")
                    results[lbl] = float(prob)

        return output_image, results

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return image, {f"Error: {str(e)}": 0.0}

def create_demo():
    with gr.Blocks(title="Facial Expression Recognition") as demo:
        gr.Markdown("# AI Emotion Recognition Demo")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=["webcam", "upload"], label="Input", type="numpy")
                analyze_btn = gr.Button("Analyze Emotion", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Processed Result")
                output_label = gr.Label(num_top_classes=3, label="Probabilities")
        
        analyze_btn.click(
            fn=predict_emotion, 
            inputs=input_image, 
            outputs=[output_image, output_label]
        )
        
    return demo

if __name__ == "__main__":
    demo_app = create_demo()
    demo_app.launch(share=True) 

def run_from_main():
    demo_app = create_demo()
    demo_app.launch()