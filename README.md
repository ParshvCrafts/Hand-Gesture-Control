# Hand Gesture Control

Control your computer with bare hands: a webcam, MediaPipe hand tracking and a few finger poses replace the mouse, the volume keys and the paintbrush.

![Python](https://img.shields.io/badge/python-3.11-blue)
![MediaPipe](https://img.shields.io/badge/mediapipe-0.10.8-orange)
![OpenCV](https://img.shields.io/badge/opencv-4.11-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## What's inside

Every app builds on one shared module, [`hand_tracking_module.py`](hand_tracking_module.py), which runs MediaPipe Hands once per frame and caches each hand's 21 landmarks, handedness and finger states.

| App | Gesture | Result |
| --- | --- | --- |
| [`virtual_painter.py`](virtual_painter.py) | Index + middle up: select a colour or the eraser in the toolbar. Index only: draw. Other hand in the toolbar: clear | Air-drawing canvas blended over the live video |
| [`virtual_mouse.py`](virtual_mouse.py) | Index only: move the cursor (smoothed). Index + middle pinched together: click | System mouse control |
| [`volume_control.py`](volume_control.py) | Thumb-to-index distance (30 to 250 px) | System volume, with an on-screen volume bar |
| [`mouse_and_volume_control.py`](mouse_and_volume_control.py) | One hand drives the mouse; the other changes volume only while held in a thumb + index "pinch lock" pose | Both at once, two hands |
| [`finger_counter.py`](finger_counter.py) | Hold up 0 to 10 fingers on one or two hands | Live count with a matching overlay image |
| [`hand_tracking_basics.py`](hand_tracking_basics.py) | None | Minimal MediaPipe Hands example: landmarks + FPS |

## Quick start

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python virtual_painter.py      # press q to quit
```

Needs a webcam. Mouse and volume control use `autopy` and `pycaw`, so those three apps run on **Windows only**; the painter, finger counter and basics run anywhere OpenCV and MediaPipe do.

## How it works

1. OpenCV grabs a frame and mirrors it, so the display behaves like a mirror.
2. `HandDetector.findHands()` runs MediaPipe Hands and caches each hand as `{"type", "lmList", "handNo", "wrist_pos"}`, where `lmList` maps landmark id to pixel `(x, y)`.
3. `fingersUp()` turns landmarks into `[thumb, index, middle, ring, pinky]` flags: a finger is up when its tip sits above its middle joint; the thumb compares x positions, with the direction set by handedness.
4. Each app maps those flags and landmark distances to an action: cursor moves are interpolated from a reduced frame region to the full screen and smoothed; volume maps pinch distance to the system volume range.

## Project structure

```
hand_tracking_module.py   shared detector used by every app
virtual_painter.py        air canvas
virtual_mouse.py          cursor + click
volume_control.py         pinch-to-volume
mouse_and_volume_control.py
finger_counter.py
hand_tracking_basics.py
assets/fingers/           finger-counter overlay images
assets/toolbar/           painter toolbar images
tests/                    scripted-hand test for the painter
```

## Status

- All seven scripts run end to end in a headless harness (fake camera frames, stubbed display, mouse and audio).
- The painter's select, draw and clear logic is covered by a scripted-hand test: `python tests/test_virtual_painter.py`.
- Which physical hand MediaPipe calls "Left" or "Right" on the mirrored frame has not been re-checked on a live webcam since the shared-module refactor; if the painter's hands feel swapped, swap the two `hand["type"]` checks in `virtual_painter.py`.

## License

[MIT](LICENSE)
