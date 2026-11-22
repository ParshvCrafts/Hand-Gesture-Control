"""Drive virtual_painter.py's real loop with scripted hands (no camera, no MediaPipe).

Run: pytest -q tests
Catches: wrong hand-label mapping (Right paints / Left clears), only the first listed hand being handled,
clearing overwriting the selected tool, and a crash on a frame with no hands.
"""
import os
import runpy
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import hand_tracking_module as htm

W, H = 1280, 720
SELECT, DRAW = [0, 1, 1, 0, 0], [0, 1, 0, 0, 0]


def hand(kind, tip, fingers):
    lm = {i: (0, 0) for i in range(21)}
    lm[8], lm[12] = tip, (tip[0] + 30, tip[1])
    return {"type": kind, "lmList": lm}, fingers


# One list of (hand, fingers) per frame. Toolbar = top strip (y < headerHeight), 4 buttons of W/4 px: Red, Blue,
# Green, Eraser.
SCRIPT = [
    [hand("Right", (100, 50), SELECT)],          # 0 select Red
    [hand("Right", (600, 400), DRAW)],           # 1 draw start
    [hand("Right", (700, 450), DRAW)],           # 2 draw: canvas must have ink
    [],                                          # 3 no hand: must not crash
    [hand("Right", (800, 450), DRAW),            # 4 drawing hand listed first, other hand clears
     hand("Left", (1100, 50), SELECT)],          #   over the Eraser button: canvas empty, tool stays Red
]


def test_select_draw_clear(monkeypatch):
    frame = {"i": -1}
    seen = {}

    class FakeDetector:
        def __init__(self, *a, **k): pass
        def findHands(self, img, draw=False):
            frame["i"] += 1
            return img
        def findAllHands(self, img, draw=False):
            return [h for h, _ in SCRIPT[frame["i"]]]
        def fingersUp(self, handNo=0):
            return SCRIPT[frame["i"]][handNo][1]

    class Cap:
        def __init__(self, *a): pass
        def set(self, *a): return True
        def read(self): return True, np.full((H, W, 3), 90, np.uint8)
        def release(self): pass

    def wait_key(*a):
        g = sys._getframe(1).f_globals
        if frame["i"] == 2:
            seen["ink"], seen["color"] = bool(g["imgCanvas"].any()), g["draw_color"]
        return ord("q") if frame["i"] >= len(SCRIPT) - 1 else -1

    monkeypatch.setattr(htm, "HandDetector", FakeDetector)
    monkeypatch.setattr(cv2, "VideoCapture", Cap)
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "waitKey", wait_key)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.chdir(REPO)

    g = runpy.run_path(os.path.join(REPO, "virtual_painter.py"), run_name="__main__")

    assert seen["color"] == (0, 0, 255), "Right hand in the toolbar should select Red"
    assert seen["ink"], "Right hand with only the index finger up should draw"
    assert not g["imgCanvas"].any(), "Left hand in the toolbar should clear, even when listed second"
    assert g["idx"] == 0, "clearing must not change the selected tool"
