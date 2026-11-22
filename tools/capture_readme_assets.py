"""Regenerate the README images in docs/assets/ from the apps' own render loops (no webcam needed).

Run from the repo root:  python tools/capture_readme_assets.py   (needs ffmpeg on PATH for the GIF)

- real-hand-strip.png: hand_tracking_module, finger_counter.py and volume_control.py each run their real loop with
  real MediaPipe on one photo (tests/fixtures/right_palm.jpg, CC BY-SA 4.0) instead of a camera frame.
- painter.gif: virtual_painter.py's real loop, but the hand is a scripted fingertip path (no MediaPipe), because
  a still photo cannot draw.
- social-preview.png: 1280x640 card built from the two images above.
Apps run unmodified: cv2.VideoCapture / imshow / waitKey are swapped for fakes, autopy and pycaw are stubbed.
"""
import math
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import types

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "docs", "assets")
sys.path.insert(0, REPO)
os.chdir(REPO)
import hand_tracking_module as htm

W, H = 1280, 720
BG = (238, 238, 238)


def stub_system_modules():
    autopy = types.ModuleType("autopy")
    autopy.screen = types.SimpleNamespace(size=lambda: (1920, 1080))
    autopy.mouse = types.SimpleNamespace(move=lambda *a: None, click=lambda *a: None)
    sys.modules["autopy"] = autopy

    class Volume:
        def GetVolumeRange(self): return (-65.25, 0.0, 0.03125)
        def SetMasterVolumeLevel(self, *a): pass
    pycaw, pp = types.ModuleType("pycaw"), types.ModuleType("pycaw.pycaw")
    pp.AudioUtilities = types.SimpleNamespace(GetSpeakers=lambda: types.SimpleNamespace(EndpointVolume=Volume()))
    pycaw.pycaw = pp
    sys.modules["pycaw"], sys.modules["pycaw.pycaw"] = pycaw, pp


def run_app(script, frames):
    """Run an app's real loop on the given frames; return every image it passed to cv2.imshow."""
    shown, it = [], iter(frames)
    saved = cv2.VideoCapture, cv2.imshow, cv2.waitKey, cv2.destroyAllWindows

    class Cap:
        def __init__(self, *a): pass
        def set(self, *a): return True
        def read(self):
            f = next(it, None)
            return (f is not None), (None if f is None else f.copy())
        def release(self): pass

    cv2.VideoCapture = Cap
    cv2.imshow = lambda name, img: shown.append(img.copy())
    cv2.waitKey = lambda *a: -1
    cv2.destroyAllWindows = lambda: None
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        cv2.VideoCapture, cv2.imshow, cv2.waitKey, cv2.destroyAllWindows = saved
    return shown


def photo_frame():
    """The fixture photo (unflipped, as a webcam delivers it) on a 1280x720 frame, just right of the app overlays."""
    hand = cv2.imread(os.path.join(REPO, "tests", "fixtures", "right_palm.jpg"))
    hand = cv2.resize(hand, (int(hand.shape[1] * (H - 40) / hand.shape[0]), H - 40), interpolation=cv2.INTER_AREA)
    frame = np.full((H, W, 3), BG, np.uint8)
    x = 230
    frame[20:20 + hand.shape[0], x:x + hand.shape[1]] = hand
    return frame


def landmark_view(frame):
    """Mirrored like the apps' cv2.flip(img, 1), so MediaPipe's label names the physical hand."""
    det = htm.HandDetector(mode=True, max_num_hands=1)
    img = det.findHands(cv2.flip(frame, 1), draw=True)
    hand = det.findAllHands(img)[0]
    det.findBoundingBox(img, handNo=0, draw=True, color=(255, 0, 0))
    x0 = hand["wrist_pos"][0] - 360
    return img[:, x0:x0 + 690].copy(), f'{hand["type"]} hand, fingersUp() = {det.fingersUp(0)}'


def label(img, text):
    out = cv2.copyMakeBorder(img, 0, 56, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(out, text, (16, img.shape[0] + 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2)
    return out


def painter_script():
    """(hand_type, index_tip, fingers) per frame: pick colours in the toolbar, draw a wave, a ring and a tick."""
    select, draw = [0, 1, 1, 0, 0], [0, 1, 0, 0, 0]
    bw = W // 4  # four toolbar buttons
    steps = []

    def pick(button):
        steps.extend([("Right", (button * bw + bw // 2, 60), select)] * 3)

    def lift(x, y):
        steps.extend([("Right", (x, y), select)] * 2)

    pick(0)
    pts = [(260 + i * 14, int(420 + 70 * math.sin(i / 4))) for i in range(36)]
    lift(*pts[0]); steps.extend(("Right", p, draw) for p in pts)
    pick(1)
    ring = [(int(880 + 100 * math.cos(t)), int(430 + 100 * math.sin(t))) for t in np.linspace(0, 2 * math.pi, 34)]
    lift(*ring[0]); steps.extend(("Right", p, draw) for p in ring)
    pick(2)
    tick = [(1040 + i * 6, 440 + i * 6) for i in range(12)] + [(1106 + i * 7, 506 - i * 14) for i in range(14)]
    lift(*tick[0]); steps.extend(("Right", p, draw) for p in tick)
    steps.extend([("Right", (1200, 300), select)] * 12)  # hold the finished picture
    return steps


def capture_painter():
    steps, frame_no = painter_script(), {"i": -1}

    class ScriptedDetector:
        def __init__(self, *a, **k): pass
        def findHands(self, img, draw=False):
            frame_no["i"] += 1
            return img
        def findAllHands(self, img, draw=False):
            kind, tip, _ = steps[frame_no["i"]]
            lm = {i: (0, 0) for i in range(21)}
            lm[8], lm[12] = tip, (tip[0] + 34, tip[1] + 4)
            return [{"type": kind, "lmList": lm}]
        def fingersUp(self, handNo=0):
            return steps[frame_no["i"]][2]

    real = htm.HandDetector
    htm.HandDetector = ScriptedDetector
    try:
        bg = np.full((H, W, 3), (70, 62, 58), np.uint8)
        return run_app("virtual_painter.py", [bg] * (len(steps) + 1))  # +1: the painter reads one probe frame
    finally:
        htm.HandDetector = real


def write_gif(frames, path, width=640, fps=12):
    tmp = tempfile.mkdtemp()
    try:
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(tmp, f"{i:04d}.png"), f)
        vf = f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                        os.path.join(tmp, "%04d.png"), "-vf", vf, "-loop", "0", path], check=True)
    finally:
        shutil.rmtree(tmp)


def social_preview(strip, painter_frame):
    card = np.full((640, 1280, 3), (40, 34, 30), np.uint8)
    cv2.putText(card, "Hand Gesture Control", (60, 120), cv2.FONT_HERSHEY_DUPLEX, 2.1, (255, 255, 255), 3)
    cv2.putText(card, "Webcam + MediaPipe: air painting, virtual mouse, pinch-to-volume",
                (62, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (200, 200, 200), 2)
    pf = cv2.resize(painter_frame, (640, 360), interpolation=cv2.INTER_AREA)
    st = cv2.resize(strip, (int(strip.shape[1] * 360 / strip.shape[0]), 360), interpolation=cv2.INTER_AREA)
    card[230:590, 60:700] = pf
    card[230:590, 720:1220] = st[:, :500]  # left 500 px of the finger-counter frame: overlay + hand
    return card


def main():
    os.makedirs(OUT, exist_ok=True)
    stub_system_modules()
    frame = photo_frame()

    lm, lm_text = landmark_view(frame)
    counter = run_app("finger_counter.py", [frame] * 3)[-1]
    volume = run_app("volume_control.py", [frame] * 3)[-1]
    panels = [label(lm, lm_text),
              label(counter[:, :690], "finger_counter.py"),
              label(volume[:, :690], "volume_control.py")]
    gap = np.full((panels[0].shape[0], 12, 3), 255, np.uint8)
    strip = np.hstack([panels[0], gap, panels[1], gap, panels[2]])
    strip = cv2.resize(strip, (1800, int(strip.shape[0] * 1800 / strip.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(OUT, "real-hand-strip.png"), strip)

    painter = capture_painter()
    write_gif(painter, os.path.join(OUT, "painter.gif"))
    cv2.imwrite(os.path.join(OUT, "social-preview.png"), social_preview(counter, painter[-1]))
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
