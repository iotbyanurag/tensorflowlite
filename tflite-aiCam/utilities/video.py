import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os
import logging

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

# Open the RTSP stream
video_path = 'rtsp://localhost:8554/cam1'
cap = cv2.VideoCapture(video_path)

# Skip frame counter
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

    # Display the frame
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()

print("Inference completed. Live display ended.")

