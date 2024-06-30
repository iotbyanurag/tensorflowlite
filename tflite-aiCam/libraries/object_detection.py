import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import logging
import multiprocessing
import time
import os

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# Define the absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../modelResources/ssd_mobilenet_v2_coco_quant_postprocess.tflite')
LABEL_MAP_PATH = os.path.join(BASE_DIR, '../modelResources/labelmap.txt')
RECORDINGS_DIR = os.path.join(BASE_DIR, '../../recordings')

# Create the recordings directory if it doesn't exist
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load label map
with open(LABEL_MAP_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
logging.debug(f"Loaded labels: {labels}")

# Global variable to control recording
recording = multiprocessing.Value('b', False)
recording_start_time = multiprocessing.Value('d', 0.0)

# Function to perform object detection on a frame
def detect_objects(frame, recording, recording_start_time):
    height, width, _ = frame.shape
    input_shape = input_details[0]['shape']
    image_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(image_resized, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    person_detected = False

    for i in range(len(scores)):
        if scores[i] > 0.5 and int(classes[i]) == 0:  # Confidence threshold and class index for "person"
            person_detected = True
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height))
            label = f"Person: {int(scores[i] * 100)}%"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if person_detected and not recording.value:
        with recording.get_lock():
            recording.value = True
            recording_start_time.value = time.time()
        p = multiprocessing.Process(target=record_clip, args=(frame,))
        p.start()

    return frame

# Function to record 15-second clip
def record_clip(initial_frame):
    video_path = 'rtsp://localhost:8554/cam1'
    cap = cv2.VideoCapture(video_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_file = os.path.join(RECORDINGS_DIR, f'person_detected_{int(time.time())}.mp4')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (initial_frame.shape[1], initial_frame.shape[0]))
    
    start_time = time.time()
    while time.time() - start_time < 15:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to grab frame")
            break

        # Add timestamp overlay
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    with recording.get_lock():
        recording.value = False

def main():
    video_path = 'rtsp://localhost:8554/cam1'
    cap = cv2.VideoCapture(video_path)
    
    frame_skip = 5  # Process every 5th frame
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to grab frame")
            time.sleep(0.1)
            continue

        frame_counter += 1
        if frame_counter % frame_skip == 0:
            # Perform object detection on the frame
            frame = detect_objects(frame, recording, recording_start_time)

    cap.release()

if __name__ == '__main__':
    main()
