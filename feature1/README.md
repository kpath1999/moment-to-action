# Moment to Action repository

Thank you for contributing to this repository. This project is optimized for **source-code–only collaboration**, primarily for experiments and development on the Thundercomm Rubik Pi platform.

Please read this document carefully before adding or modifying code.

This is the current state of the vigilante/trigger branch.

This branch is focused on implementing an efficient trigger mechanism that can estimate the threat level of the environment.

Currently, only audio mode is used. We plan on adding motion (IMU) as a modality.

The script realtime_monitor.py is used to run YAMNet + MoViNet

Please use the following instruction to run the code:

```bash
python3 -m venv <venv_name>
source <venv_name>/bin/activate
pip install -r requirements.txt
```

To run realtime_monitor.py (connect a microphone and camera to the rubik pi)

Usually, on the pi (if microphone is connected to the AUX port)
camera_id   :2
mic_id      :3

yamnet-threshold: This threshold (between 0.0 and 1.0) determines when YAMNet triggers (threat_score > threshold)

window-sec: Mentions how many seconds on video data MoViNet uses to make an inference
```bash
python3 realtime_monitor.py --camera <camera_id> --mic-device <mic_device> --yamnet-threshold <yth> --window-sec <time> --output <output_file_name.mp4> --no-display
```
---
