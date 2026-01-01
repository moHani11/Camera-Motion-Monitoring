# Camera-Motion-Monitoring

Webcam motion detection using OpenCV optical flow.

## Install

```bash
pip install opencv-python numpy
```

## Run

```bash
python motion_detector.py
```

## Controls

- `q` - quit
- `s` - save frame
- `r` - record video
- `+/-` - adjust sensitivity
- `c` - clear trail

## What you see

- Green circle where motion is strongest
- Trail showing recent motion
- Stats box (top left)
- Motion graph (bottom right)

## Settings

```python
detector = MotionDetector(
    camera_id=0,        # which camera
    sensitivity=70,     # lower = more sensitive
    motion_threshold=10 # min motion to detect
)
```

## Output files

- `motion_recording_TIMESTAMP.avi` - recordings
- `motion_frame_TIMESTAMP.png` - screenshots
- `optical_flow_TIMESTAMP.png` - flow viz

---
# Demo

- No optical flow is detected in a **static frame**
- Relatively zero motion is detected 
![alt text](image.png)

---

- By moving an object we can see it appear in the optical flow frame
- A spike is introduced in the monitor
- Max motion rises as well 
![alt text](<Screenshot 2026-01-01 025251.png>)

---

- by vibrating the camera itself, oscillations are introduced
- **objects** in the frame starts to **stand out**
![alt text](image-1.png)
> This can be used for multiple other applications