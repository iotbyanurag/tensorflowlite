from flask import Flask, render_template, Response
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import logging
import multiprocessing
import time

app = Flask(__name__)

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# Load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path="ssd_mobilenet_v2_coco_quant_postprocess.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load label map
with open('labelmap.txt', 'r') as f:
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
    out = cv2.VideoWriter(f'person_detected_{int(time.time())}.mp4', fourcc, 20.0, (initial_frame.shape[1], initial_frame.shape[0]))
    
    start_time = time.time()
    while time.time() - start_time < 15:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

    with recording.get_lock():
        recording.value = False

# Generator function to stream video frames
def gen_frames(recording, recording_start_time):
    video_path = 'rtsp://localhost:8554/cam1'
    cap = cv2.VideoCapture(video_path)
    frame_skip = 5  # Process every 5th frame
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to grab frame")
            break

        frame_counter += 1
        if frame_counter % frame_skip == 0:
            # Perform object detection on the frame
            frame = detect_objects(frame, recording, recording_start_time)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(recording, recording_start_time), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
