"""Shared-module tests: handedness labels, the thumb rule, and one real-photo MediaPipe run.

Run: pytest -q tests
"""
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hand_tracking_module as htm

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "right_palm.jpg")


def open_hand(thumb_dx):
    """Landmarks for an upright hand with all four fingers up; thumb tip sits thumb_dx px from its IP joint."""
    lm = {i: (300, 400) for i in range(21)}
    for tip in (8, 12, 16, 20):
        lm[tip], lm[tip - 2] = (300, 100), (300, 250)  # tip above PIP joint
    lm[3] = (200, 300)
    lm[4] = (200 + thumb_dx, 280)
    return lm


def mirror(lm, width=640):
    return {k: (width - x, y) for k, (x, y) in lm.items()}


@pytest.mark.parametrize("label, thumb_dx, thumb_up", [
    ("Right", -40, 1),  # "Right" hand: thumb points to image-left when extended
    ("Right", +40, 0),
    ("Left", +40, 1),
    ("Left", -40, 0),
])
def test_thumb_rule_follows_label(label, thumb_dx, thumb_up):
    # Catches: the pre-fix module had the thumb rule reversed for both labels.
    assert htm.fingers_up(open_hand(thumb_dx), label) == [thumb_up, 1, 1, 1, 1]


def test_thumb_rule_survives_mirroring():
    # A cv2.flip mirrors the geometry and MediaPipe swaps the label, so the answer must not change.
    lm = open_hand(-40)
    assert htm.fingers_up(lm, "Right") == htm.fingers_up(mirror(lm), "Left")


def test_folded_fingers_read_as_down():
    lm = open_hand(-40)
    lm[12], lm[16] = (300, 300), (300, 300)  # middle and ring tips below their PIP joints
    assert htm.fingers_up(lm, "Right") == [1, 1, 0, 0, 1]


def test_handedness_label_is_stripped():
    # Catches: the Windows mediapipe 0.10.8 wheel returns "Right\r" (CRLF label file), which made every
    # == "Right" check in the apps fail silently.
    lms = SimpleNamespace(landmark=[SimpleNamespace(x=0.5, y=0.5)] * 21)
    handed = SimpleNamespace(classification=[SimpleNamespace(label="Right\r")])
    results = SimpleNamespace(multi_hand_landmarks=[lms], multi_handedness=[handed])
    det = htm.HandDetector.__new__(htm.HandDetector)
    det.hands = SimpleNamespace(process=lambda img: results)
    det.findHands(np.zeros((480, 640, 3), np.uint8))
    assert det.findAllHands(None)[0]["type"] == "Right"


@pytest.fixture(scope="module")
def detector():
    return htm.HandDetector(mode=True, max_num_hands=1, detection_confidence=0.5)


@pytest.mark.parametrize("flip, label", [(False, "Left"), (True, "Right")])
def test_real_open_palm(detector, flip, label):
    # Real photo of a right palm with all five fingers extended (see fixtures/ATTRIBUTION.md).
    # Unflipped, MediaPipe (which assumes a mirrored frame) calls it "Left"; mirrored, it is the physical "Right".
    img = cv2.imread(FIXTURE)
    if flip:
        img = cv2.flip(img, 1)
    detector.findHands(img)
    hands = detector.findAllHands(img)
    assert len(hands) == 1
    assert hands[0]["type"] == label
    assert detector.fingersUp(0) == [1, 1, 1, 1, 1]
