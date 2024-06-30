
First, open the release page on the GitHub.
Second, copy a link to an ARM64 archive. It’s essential to choose a proper version here.


Third, create a folder for it on the Raspberry PI.

mkdir mediamtx && cd mediamtx

Fourth, download it using the WGET command (find the latest link on GitHub).

wget https://github.com/bluenviron/mediamtx/releases/download/v1.7.0/mediamtx_v1.7.0_linux_arm64v8.tar.gz

Fifth, unarchive it to the same folder.

tar -xvzf mediamtx_v1.7.0_linux_arm64v8.tar.gz

Sixth, open the YML configuration file for editing.

nano mediamtx.yml

Seventh, scroll down and paste these configurations.

cam1:
    runOnInit: bash -c 'rpicam-vid -t 0 --camera 0 --nopreview --codec yuv420 --width 1280 --height 720 --inline --listen -o - | ffmpeg -f rawvideo -pix_fmt yuv420p -s:v 1280x720 -i /dev/stdin -c:v libx264 -preset ultrafast -tune zerolatency -f rtsp rtsp://localhost:$RTSP_PORT/$MTX_PATH'
    runOnInitRestart: yes

In a nutshell, the MediaMTX software will run this command in a bash.

Next, it asks the RPICAM-VID command (the same as we dealt with before) to send a stream to the FFMPEG.

And FFMPEG will send it to the MediaMTX via RTSP protocol, but locally.

We can save the configuration and run MediaMTX.

./mediamtx

The very first prints of it will have helpful information about protocol and ports that we can use.


