import cv2
from ultralytics import YOLO
from pathlib import Path

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use your custom path if needed

# Get all video files
video_files = list(Path("videos").glob("*.mp4"))

for video_path in video_files:
    print(f"Processing {video_path.name}...")
    
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    
    # Get video properties for VideoWriter
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or "XVID"
    out = cv2.VideoWriter(f"results/tracked_{video_path.name}", fourcc, fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Perform tracking with ByteTrack
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")[0]
        boxes = results.boxes

        person_count = 0

        if boxes.id is not None:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                if cls_id == 0:
                    person_count += 1
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                    track_id = int(boxes.id[i])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display total people tracked
        cv2.putText(frame, f"Persons: {person_count}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Write the frame with drawings to the output file
        out.write(frame)

        # Show the frame
        cv2.imshow(f"Tracking {video_path.name}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

print("Processing complete!") 