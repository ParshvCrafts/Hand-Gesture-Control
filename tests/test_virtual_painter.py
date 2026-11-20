"""Run: python tests/test_virtual_painter.py
Drive virtual_painter.py with scripted hands; assert select -> draw -> clear behaviour of the port.
Catches: wrong hand label mapping (Right paints / Left clears), wrong lmList indexing, fingersUp handNo."""
import os, sys, runpy
import cv2, numpy as np

repo = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo); os.chdir(repo)
import hand_tracking_module as htm

W = 1280
def hand(kind, tip, mid):
    lm = {i: (0, 0) for i in range(21)}; lm[8] = tip; lm[12] = mid
    return {"type": kind, "lmList": lm}
# (hands, fingers) per frame. Toolbar header is y < headerHeight; button 0 (Red) is the leftmost quarter.
script = [
    ([hand("Right", (100, 50), (130, 50))], [0, 1, 1, 0, 0]),   # select Red in toolbar
    ([hand("Right", (600, 400), (630, 400))], [0, 1, 0, 0, 0]),  # draw start
    ([hand("Right", (700, 450), (730, 450))], [0, 1, 0, 0, 0]),  # draw line
    ([], None),                                                  # no hand: must not crash
    ([hand("Left", (100, 50), (130, 50))], [0, 1, 1, 0, 0]),     # left hand in toolbar clears
]
state = {"i": -1, "drawn": None}

class FakeDetector:
    def __init__(self, *a, **k): pass
    def findHands(self, img, draw=False):
        state["i"] += 1; return img
    def findAllHands(self, img, draw=False): return script[min(state["i"], len(script) - 1)][0]
    def fingersUp(self, handNo=0): return script[min(state["i"], len(script) - 1)][1]
htm.HandDetector = FakeDetector

class Cap:
    def __init__(self, *a): pass
    def set(self, *a): return True
    def read(self): return True, np.full((720, W, 3), 90, np.uint8)
    def release(self): pass
cv2.VideoCapture = Cap
cv2.imshow = lambda *a: None
def waitKey(*a):
    g = sys._getframe(1).f_globals
    if state["i"] == 2: state["drawn"] = int(g["imgCanvas"].any()), g["draw_color"]
    return ord("q") if state["i"] >= len(script) - 1 else -1
cv2.waitKey = waitKey
cv2.destroyAllWindows = lambda: None

g = runpy.run_path("virtual_painter.py", run_name="__main__")
drawn, color = state["drawn"]
assert color == (0, 0, 255), f"Right hand in toolbar should select Red, got {color}"
assert drawn == 1, "Right hand index-only should draw on the canvas"
assert not g["imgCanvas"].any(), "Left hand index+middle in toolbar should clear the canvas"
print("painter logic OK: select Red, draw, survive no-hand frame, clear")
