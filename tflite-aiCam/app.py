from flask import Flask, render_template, Response, send_from_directory
import logging
import os
from multiprocessing import Value
from libraries.object_detection import gen_frames, recording_start_time, recording

app = Flask(__name__)

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# Path for recordings
RECORDINGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'recordings'))

# Create the recordings directory if it doesn't exist
os.makedirs(RECORDINGS_DIR, exist_ok=True)

logging.debug(f"Recordings directory: {RECORDINGS_DIR}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recordings')
def list_recordings():
    files = os.listdir(RECORDINGS_DIR)
    logging.debug(f"Files in recordings directory: {files}")
    files = [f for f in files if f.endswith('.mp4')]

    # Prepare file details for the template
    file_details = []
    for file in files:
        parts = file.replace('.mp4', '').split('_')
        logging.debug(f"Processing file: {file}, parts: {parts}")
        if len(parts) == 3:
            file_date = parts[1]
            file_time = parts[2]
        else:
            file_date = 'Unknown Date'
            file_time = 'Unknown Time'
        file_details.append({
            'file': file,
            'file_date': file_date,
            'file_time': file_time,
        })

    logging.debug(f"File details: {file_details}")
    return render_template('recordings.html', files=file_details)

@app.route('/recordings/<filename>')
def get_recording(filename):
    return send_from_directory(RECORDINGS_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
