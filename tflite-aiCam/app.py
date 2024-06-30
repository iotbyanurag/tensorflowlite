from flask import Flask, render_template, Response, send_from_directory
import logging
import os
from multiprocessing import Value
from libraries.object_detection import gen_frames, recording_start_time, recording

app = Flask(__name__)

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# Path for recordings
RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), '../recordings')

# Create the recordings directory if it doesn't exist
os.makedirs(RECORDINGS_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recordings')
def list_recordings():
    files = os.listdir(RECORDINGS_DIR)
    files = [f for f in files if f.endswith('.mp4')]
    return render_template('recordings.html', files=files)

@app.route('/recordings/<filename>')
def get_recording(filename):
    return send_from_directory(RECORDINGS_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
