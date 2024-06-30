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

# Open the video file or capture from camera
video_path = 'input_video.mp4'  # Change this to the path of your video file
cap = cv2.VideoCapture(video_path)

# Get video details
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a video writer to save the output video
output_video_path = 'output_video.mp4'
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    frame = detect_objects(frame)

    # Write the frame to the output video
    out.write(frame)

# Release video capture and writer
cap.release()
out.release()

print(f"Inference completed. Check the output video saved as {output_video_path}.")

