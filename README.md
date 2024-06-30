To create services for the three applications you need to run, you can use `systemd` to manage them. `systemd` is a system and service manager for Linux operating systems. Here’s how you can create the necessary services:

1. **Create the service file for `mediamtx`:**

Create a new service file called `mediamtx.service`.

```bash
sudo nano /etc/systemd/system/mediamtx.service
```

Add the following content to the file:

```ini
[Unit]
Description=MediaMTX Service
After=network.target

[Service]
ExecStart=/home/admin/tensorflow/CameraStreaming/mediamtx
WorkingDirectory=/home/admin/tensorflow/CameraStreaming
Restart=always
User=admin
Group=admin

[Install]
WantedBy=multi-user.target
```

2. **Create the service file for the object detection and Flask app:**

Create a new service file called `object_detection.service`.

```bash
sudo nano /etc/systemd/system/object_detection.service
```

Add the following content to the file:

```ini
[Unit]
Description=Object Detection and Flask App Service
After=network.target mediamtx.service

[Service]
ExecStart=/bin/bash -c 'source /home/admin/tensorflow/camera-env/bin/activate && python /home/admin/tensorflow/tflite-aiCam/libraries/object_detection.py && python /home/admin/tensorflow/tflite-aiCam/app.py'
WorkingDirectory=/home/admin/tensorflow/tflite-aiCam
Restart=always
User=admin
Group=admin

[Install]
WantedBy=multi-user.target
```

3. **Reload systemd and enable the services:**

After creating the service files, you need to reload `systemd` to recognize the new services and then enable and start them.

```bash
sudo systemctl daemon-reload
sudo systemctl enable mediamtx.service
sudo systemctl enable object_detection.service
sudo systemctl start mediamtx.service
sudo systemctl start object_detection.service
```

4. **Check the status of the services:**

You can check the status of your services to ensure they are running correctly:

```bash
sudo systemctl status mediamtx.service
sudo systemctl status object_detection.service
```

By following these steps, you create two separate services that will ensure `mediamtx` and the object detection along with the Flask app are started automatically on boot and managed by `systemd`. If any of these services fail, `systemd` will attempt to restart them automatically.