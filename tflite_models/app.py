from flask import Flask, render_template, Response
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import logging

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

# Function to perform object detection on a frame
def detect_objects(frame):
    height, width, _ = frame.shape
    input_shape = input_details[0]['shape']
    image_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(image_resized, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if scores[i] > 0.5:  # Confidence threshold
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height))
            class_id = int(classes[i])
            if class_id < len(labels):
                label = labels[class_id]
                label = f"{label}: {int(scores[i] * 100)}%"
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Generator function to stream video frames
def gen_frames():
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
            frame = detect_objects(frame)

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
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

