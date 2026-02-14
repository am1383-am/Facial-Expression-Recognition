import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
from PIL import Image
import pandas as pd
import plotly.express as px
import time

st.set_page_config(
    page_title="FER Project",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: bold; text-align: center; color: #31333F; margin-bottom: 20px;}
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #4F8BF9; }
    .stSelectbox { margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

EMOTION_LABELS = {
    0: 'Angry', 1: 'Contempt', 2: 'Disgust',
    3: 'Fear', 4: 'Happy', 5: 'Neutral',
    6: 'Sad', 7: 'Surprise'
}

@st.cache_resource
def load_resources(model_name):
    model_path = os.path.join("models", model_name)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    try:
        model = tf.keras.models.load_model(model_path)
        face_cascade = cv2.CascadeClassifier(cascade_path)
        return model, face_cascade, True
    except Exception as e:
        return None, None, str(e)

def preprocess_image(roi_gray):
    try:
        resized = cv2.resize(roi_gray, (48, 48))
        normalized = resized / 255.0
        return np.expand_dims(np.expand_dims(normalized, axis=0), axis=-1)
    except: return None

def create_chart(predictions):
    if not predictions: return None
    df = pd.DataFrame(list(predictions.items()), columns=['Emotion', 'Probability'])
    df = df.sort_values(by='Probability', ascending=True)
    fig = px.bar(
        df, x='Probability', y='Emotion', orientation='h',
        text_auto='.0%', color='Probability', color_continuous_scale='Blues'
    )
    fig.update_layout(
        xaxis_title="", yaxis_title="", showlegend=False,
        height=300, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_xaxes(showticklabels=False)
    return fig

def main():
    st.markdown('<div class="main-header">Facial Expression Recognition System</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Settings")
        
        if not os.path.exists("models"):
            os.makedirs("models")
            
        model_files = [f for f in os.listdir("models") if f.endswith(".keras")]
        
        default_idx = 0
        if "final_model.keras" in model_files:
            default_idx = model_files.index("final_model.keras")
        
        selected_model_file = st.selectbox(
            "Select Model Architecture:",
            options=model_files,
            index=default_idx,
            format_func=lambda x: x.replace(".keras", "") 
        )

        interval_options = [10, 5, 4, 3, 2, 1, 0.5]
        analysis_interval = st.selectbox(
            "Analysis Interval (Seconds):",
            options=interval_options,
            index=2
        )
        
        st.divider()
        st.info(f"Active Model: {selected_model_file.replace('.keras', '')}\n\nInterval: Every {analysis_interval}s")

    if selected_model_file:
        model, face_cascade, status = load_resources(selected_model_file)
        if status is not True:
            st.error(status)
            return
    else:
        st.error("No models found in the models directory.")
        return

    input_method = st.radio("Select Input Method:", ["Live Webcam", "Upload Image"], horizontal=True)
    st.divider()

    if "Upload" in input_method:
        uploaded_file = st.file_uploader("Choose an image (JPG, PNG)", type=['jpg', 'png'])
        
        if uploaded_file:
            input_image = np.array(Image.open(uploaded_file))
            col_img, col_res = st.columns([1, 1.2])
            
            gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY) if len(input_image.shape) == 3 else input_image
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            
            final_img = input_image.copy()
            preds = {}
            
            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                inp = preprocess_image(roi)
                if inp is not None:
                    p = model.predict(inp, verbose=0)[0]
                    if not preds:
                         for i, val in enumerate(p): preds[EMOTION_LABELS[i]] = float(val)
                    
                    best = np.argmax(p)
                    color = (0, 255, 0) if best == 4 else (255, 0, 0)
                    cv2.rectangle(final_img, (x, y), (x+w, y+h), color, 3)
                    cv2.putText(final_img, EMOTION_LABELS[best], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            if len(faces) == 0:
                inp = preprocess_image(gray)
                if inp is not None:
                    p = model.predict(inp, verbose=0)[0]
                    for i, val in enumerate(p): preds[EMOTION_LABELS[i]] = float(val)

            with col_img:
                st.image(final_img, caption="Analyzed Image", use_container_width=True)
            with col_res:
                if preds:
                    best_lbl = max(preds, key=preds.get)
                    st.metric("Dominant Emotion", best_lbl, f"{preds[best_lbl]*100:.1f}%")
                    st.plotly_chart(create_chart(preds), use_container_width=True)
                else:
                    st.warning("Analysis failed.")

    else:
        col_cam, col_stats = st.columns([1.5, 1])
        
        with col_cam:
            run = st.checkbox("Start Camera", value=False)
            image_placeholder = st.empty()
            
        with col_stats:
            timer_bar = st.progress(0, text="Waiting to start...")
            metric_ph = st.empty()
            chart_ph = st.empty()

        if run:
            cap = cv2.VideoCapture(0)
            last_analysis_time = time.time() - (analysis_interval + 1) 
            cached_preds = {}
            cached_faces = []
            
            while run:
                ret, frame = cap.read()
                if not ret: break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                current_time = time.time()
                time_diff = current_time - last_analysis_time
                
                progress_val = min(time_diff / analysis_interval, 1.0)
                if progress_val < 1.0:
                     timer_bar.progress(progress_val, text=f"Next Analysis in {analysis_interval - time_diff:.1f}s...")
                else:
                     timer_bar.progress(1.0, text="Analyzing...")

                if time_diff > analysis_interval:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
                    
                    new_preds = {}
                    found_face = False
                    
                    cached_faces = faces
                    for (x, y, w, h) in faces:
                        roi = gray[y:y+h, x:x+w]
                        inp = preprocess_image(roi)
                        if inp is not None:
                            p = model.predict(inp, verbose=0)[0]
                            for i, val in enumerate(p): new_preds[EMOTION_LABELS[i]] = float(val)
                            cached_preds = new_preds
                            found_face = True
                            break 
                    
                    if not found_face:
                        cached_faces = []
                        inp = preprocess_image(gray)
                        if inp is not None:
                            p = model.predict(inp, verbose=0)[0]
                            for i, val in enumerate(p): new_preds[EMOTION_LABELS[i]] = float(val)
                            cached_preds = new_preds
                    
                    last_analysis_time = time.time()
                    
                    if cached_preds:
                        fig = create_chart(cached_preds)
                        chart_ph.plotly_chart(fig, use_container_width=True)

                draw_img = frame_rgb.copy()
                for (x, y, w, h) in cached_faces:
                    color = (0, 255, 0)
                    if cached_preds:
                        best = max(cached_preds, key=cached_preds.get)
                        if "Happy" not in best: color = (255, 0, 0)
                    cv2.rectangle(draw_img, (x, y), (x+w, y+h), color, 3)

                image_placeholder.image(draw_img, channels="RGB", use_container_width=True)
                
                if cached_preds:
                    best = max(cached_preds, key=cached_preds.get)
                    metric_ph.metric("Last Status", best, f"{cached_preds[best]*100:.1f}%")
                else:
                    metric_ph.info("Searching for face...")

            cap.release()

    st.divider()
    with st.expander("System Guide"):
        st.markdown(f"""
        ### Instructions:
        1. **Select Model:** Choose the desired model (e.g., `final_model`) from the sidebar.
        2. **Set Interval:** Adjust the time between analyses (Default: 4s).
        3. **Webcam Mode:**
           - Check "Start Camera".
           - The system will capture and analyze a frame every **{analysis_interval} seconds**.
        """)

if __name__ == "__main__":
    main()