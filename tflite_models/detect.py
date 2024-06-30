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

# Load an image
image = cv2.imread('new.jpg')
height, width, _ = image.shape
logging.debug(f"Loaded image with shape: {image.shape}")

# Resize the image
input_shape = input_details[0]['shape']
image_resized = cv2.resize(image, (input_shape[1], input_shape[2]))
logging.debug(f"Resized image to: {image_resized.shape}")

# Convert the image to UINT8 format
input_data = np.expand_dims(image_resized, axis=0).astype(np.uint8)
logging.debug(f"Converted image to UINT8 format with shape: {input_data.shape}")

# Perform inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get the results
boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
logging.debug(f"Boxes: {boxes}")
logging.debug(f"Classes: {classes}")
logging.debug(f"Scores: {scores}")

# Load label map
with open('labelmap.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]
logging.debug(f"Loaded labels: {labels}")

# Draw the results on the image
for i in range(len(scores)):
    if scores[i] > 0.5:  # Confidence threshold
        ymin, xmin, ymax, xmax = boxes[i]
        (left, right, top, bottom) = (int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height))
        class_id = int(classes[i])
        logging.debug(f"Processing detection {i}: class_id={class_id}, score={scores[i]}, box=({left}, {top}, {right}, {bottom})")
        
        if class_id < len(labels):
            label = labels[class_id]
            label = f"{label}: {int(scores[i] * 100)}%"
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            logging.debug(f"Added label: {label}")
        else:
            logging.debug(f"Class ID {class_id} is out of range of the labels")

# Save the output image
output_image_path = 'output.jpg'
cv2.imwrite(output_image_path, image)
logging.debug(f"Saved output image to: {output_image_path}")

# Open the output image using the default image viewer
os.system(f'xdg-open {output_image_path}')

print("Inference completed. Check the output image saved as output.jpg.")

