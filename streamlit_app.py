import streamlit as st
import cv2
import tempfile
import os
from pathlib import Path
from ultralytics import YOLO
import time

st.set_page_config(
    page_title="Crowd Management Demo",
    page_icon="👥",
    layout="wide"
)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

def process_video(input_path, output_path, progress_bar=None):
    model = load_model()
    
    cap = cv2.VideoCapture(input_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_persons = 0
    max_persons = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")[0]
        boxes = results.boxes
        
        person_count = 0
        
        if boxes.id is not None:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                if cls_id == 0:  # Person class
                    person_count += 1
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                    track_id = int(boxes.id[i])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        total_persons += person_count
        max_persons = max(max_persons, person_count)
        
        cv2.putText(frame, f"Persons: {person_count}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        out.write(frame)
        
        frame_count += 1
        if progress_bar is not None:
            progress_bar.progress(frame_count / total_frames)
    
    cap.release()
    out.release()
    
    avg_persons = total_persons / frame_count if frame_count > 0 else 0
    return {
        "total_frames": frame_count,
        "max_persons": max_persons,
        "avg_persons": avg_persons
    }

def main():
    st.title("👥 Crowd Management Demo")
    st.markdown("""
    This application uses YOLOv8 for real-time person detection and tracking in videos.
    Upload a video to see the system in action!
    """)
    
    st.sidebar.title("About")
    st.sidebar.info("""
    This demo uses:
    - YOLOv8 for person detection
    - ByteTrack for person tracking
    - OpenCV for video processing
    
    Built by [Mercity AI](https://mercity.ai)
    """)
    
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        temp_dir = tempfile.TemporaryDirectory()
        input_path = os.path.join(temp_dir.name, "input_video.mp4")
        output_path = os.path.join(temp_dir.name, "output_video.mp4")
        
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
        cap.release()
        
        st.subheader("Video Information")
        col1, col2, col3 = st.columns(3)
        col1.metric("Resolution", f"{width}x{height}")
        col2.metric("FPS", f"{fps:.2f}")
        col3.metric("Duration", f"{duration} seconds")
        
        if st.button("Process Video"):
            progress_text = "Processing video..."
            progress_bar = st.progress(0)
            
            with st.spinner("Processing video..."):
                start_time = time.time()
                stats = process_video(input_path, output_path, progress_bar)
                processing_time = time.time() - start_time
            
            st.success(f"Video processed successfully in {processing_time:.2f} seconds!")
            
            st.subheader("Processing Statistics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Frames", stats["total_frames"])
            col2.metric("Max Persons", stats["max_persons"])
            col3.metric("Avg Persons", f"{stats['avg_persons']:.2f}")
            
            st.subheader("Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Video**")
                st.video(input_path)
            
            with col2:
                st.markdown("**Processed Video with Person Tracking**")
                st.video(output_path)
            
            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download Processed Video",
                    data=file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
        
        st.session_state["temp_dir"] = temp_dir

if __name__ == "__main__":
    main()
