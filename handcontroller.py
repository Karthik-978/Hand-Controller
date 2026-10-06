import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import math
import os
import time
import urllib.request
import ctypes
from ctypes import wintypes


# ============================================================
# JARVIS AIR CONTROL v2
# ============================================================
#
# POINTER
#   Index finger -> mouse
#
# MOUSE
#   Thumb + Index pinch -> Left click
#   Hold Thumb + Index -> Drag
#   Thumb + Middle pinch -> Right click
#
# TOUCHPAD-STYLE
#   Index + Middle -> Scroll
#
# WINDOWS
#   Closed fist -> Minimize all windows
#   Open palm  -> Restore all windows
#
# EXTRA
#   3 fingers + swipe left  -> Browser/app back
#   3 fingers + swipe right -> Browser/app forward
#
# SAFETY
#   There is NO Q key to close the program.
#   The camera window's close button is disabled.
#   Move the REAL mouse to the TOP-LEFT corner for PyAutoGUI
#   emergency fail-safe.
#
# IMPORTANT
#   The pointer is always tracked independently of gestures.
# ============================================================


# ============================================================
# SAFETY
# ============================================================

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


# ============================================================
# MODEL
# ============================================================

MODEL_FILE = "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model():

    if os.path.exists(MODEL_FILE):
        return

    print()
    print("Downloading MediaPipe hand model...")
    print("Please wait...")

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_FILE
    )

    print("Model downloaded successfully.")
    print()


def create_detector():

    with open(MODEL_FILE, "rb") as f:
        model_data = f.read()

    base_options = python.BaseOptions(
        model_asset_buffer=model_data
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.50,
        min_hand_presence_confidence=0.50,
        min_tracking_confidence=0.50,
    )

    return vision.HandLandmarker.create_from_options(
        options
    )


# ============================================================
# CAMERA
# ============================================================

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


# ============================================================
# POINTER
# ============================================================
#
# Large control area.
# The hand does not need to travel to the edge of the camera
# frame to reach the edge of the monitor.
# ============================================================

CONTROL_LEFT = 0.055
CONTROL_RIGHT = 0.945
CONTROL_TOP = 0.055
CONTROL_BOTTOM = 0.945


# ============================================================
# ONE EURO FILTER
# ============================================================
#
# This is much better for a hand mouse than simple averaging.
#
# Slow movement:
#   strong filtering -> stable cursor
#
# Fast movement:
#   filtering automatically relaxes -> quick cursor
#
# This directly addresses:
#   - shaking while moving long distances
#   - cursor feeling stuck
#   - cursor lagging behind the hand
# ============================================================

class LowPassFilter:

    def __init__(self):

        self.initialized = False
        self.value = 0.0

    def filter(self, value, alpha):

        if not self.initialized:

            self.value = value
            self.initialized = True

            return value

        self.value = (
            alpha * value
            +
            (1.0 - alpha) * self.value
        )

        return self.value

    def reset(self):

        self.initialized = False
        self.value = 0.0


class OneEuroFilter:

    def __init__(
        self,
        min_cutoff=1.15,
        beta=0.035,
        d_cutoff=1.0
    ):

        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()

        self.previous_raw = None
        self.previous_time = None

    @staticmethod
    def alpha(cutoff, dt):

        tau = 1.0 / (
            2.0 * math.pi * cutoff
        )

        return 1.0 / (
            1.0
            +
            tau / max(dt, 0.0001)
        )

    def reset(self):

        self.x_filter.reset()
        self.dx_filter.reset()

        self.previous_raw = None
        self.previous_time = None

    def filter(
        self,
        value,
        now
    ):

        if self.previous_time is None:

            self.previous_time = now
            self.previous_raw = value

            return self.x_filter.filter(
                value,
                1.0
            )

        dt = max(
            now - self.previous_time,
            0.001
        )

        raw_speed = (
            value
            -
            self.previous_raw
        ) / dt

        self.previous_raw = value
        self.previous_time = now

        derivative_alpha = self.alpha(
            self.d_cutoff,
            dt
        )

        filtered_speed = (
            self.dx_filter.filter(
                raw_speed,
                derivative_alpha
            )
        )

        cutoff = (
            self.min_cutoff
            +
            self.beta
            *
            abs(filtered_speed)
        )

        position_alpha = self.alpha(
            cutoff,
            dt
        )

        return self.x_filter.filter(
            value,
            position_alpha
        )


# ============================================================
# HAND GEOMETRY
# ============================================================

def clamp(
    value,
    low,
    high
):

    return max(
        low,
        min(high, value)
    )


def distance(a, b):

    return math.hypot(
        a.x - b.x,
        a.y - b.y
    )


def hand_size(landmarks):

    palm_length = distance(
        landmarks[0],
        landmarks[9]
    )

    palm_width = distance(
        landmarks[5],
        landmarks[17]
    )

    return max(
        (palm_length + palm_width) / 2,
        0.001
    )


def relative_distance(
    landmarks,
    a,
    b
):

    return (
        distance(
            landmarks[a],
            landmarks[b]
        )
        /
        hand_size(landmarks)
    )


# ============================================================
# FINGER STATE
# ============================================================

def finger_extended(
    landmarks,
    tip,
    pip
):

    wrist = landmarks[0]

    tip_distance = distance(
        landmarks[tip],
        wrist
    )

    pip_distance = distance(
        landmarks[pip],
        wrist
    )

    return (
        tip_distance
        >
        pip_distance * 1.10
    )


def get_fingers(landmarks):

    return {

        "thumb": finger_extended(
            landmarks,
            4,
            3
        ),

        "index": finger_extended(
            landmarks,
            8,
            6
        ),

        "middle": finger_extended(
            landmarks,
            12,
            10
        ),

        "ring": finger_extended(
            landmarks,
            16,
            14
        ),

        "pinky": finger_extended(
            landmarks,
            20,
            18
        ),
    }


def count_extended(fingers):

    return sum(
        1 for value in fingers.values()
        if value
    )


# ============================================================
# WINDOWS HOTKEYS
# ============================================================

def minimize_all_windows():

    # Windows Show Desktop.
    pyautogui.hotkey(
        "win",
        "d"
    )


def restore_all_windows():

    # Windows toggles Show Desktop back.
    pyautogui.hotkey(
        "win",
        "d"
    )


def browser_back():

    pyautogui.hotkey(
        "alt",
        "left"
    )


def browser_forward():

    pyautogui.hotkey(
        "alt",
        "right"
    )


# ============================================================
# CONTROLLER
# ============================================================

class JarvisController:

    def __init__(
        self,
        screen_width,
        screen_height
    ):

        self.screen_width = screen_width
        self.screen_height = screen_height

        # Separate filters for X/Y.
        self.filter_x = OneEuroFilter(
            min_cutoff=1.05,
            beta=0.045
        )

        self.filter_y = OneEuroFilter(
            min_cutoff=1.05,
            beta=0.045
        )

        # ----------------------------------------------------
        # Pinch state
        # ----------------------------------------------------

        self.left_pinching = False
        self.left_start = 0.0
        self.dragging = False

        self.right_pinching = False
        self.right_start = 0.0

        # ----------------------------------------------------
        # Scroll
        # ----------------------------------------------------

        self.scroll_y = None
        self.last_scroll = 0.0

        # ----------------------------------------------------
        # Gesture cooldown
        # ----------------------------------------------------

        self.last_action_time = 0.0
        self.last_click_time = 0.0

        self.gesture_cooldown = 0.65

        # ----------------------------------------------------
        # Fist / palm state
        # ----------------------------------------------------

        self.fist_since = None
        self.palm_since = None

        self.fist_triggered = False
        self.palm_triggered = False

        self.GESTURE_HOLD = 0.45

        # ----------------------------------------------------
        # Three-finger swipe
        # ----------------------------------------------------

        self.three_start_x = None
        self.three_start_time = None

        self.swipe_triggered = False

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        self.action = "SYSTEM READY"
        self.action_time = time.perf_counter()

        self.status = "TRACKING"

        # ----------------------------------------------------
        # Trail
        # ----------------------------------------------------

        self.trail = []


    # ========================================================
    # UI
    # ========================================================

    def set_action(self, text):

        self.action = text
        self.action_time = time.perf_counter()


    # ========================================================
    # POINTER
    # ========================================================

    def move_pointer(
        self,
        index_tip,
        now
    ):

        x = (
            index_tip.x
            -
            CONTROL_LEFT
        ) / (
            CONTROL_RIGHT
            -
            CONTROL_LEFT
        )

        y = (
            index_tip.y
            -
            CONTROL_TOP
        ) / (
            CONTROL_BOTTOM
            -
            CONTROL_TOP
        )

        x = clamp(
            x,
            0.0,
            1.0
        )

        y = clamp(
            y,
            0.0,
            1.0
        )

        target_x = (
            x
            *
            (self.screen_width - 1)
        )

        target_y = (
            y
            *
            (self.screen_height - 1)
        )

        smooth_x = self.filter_x.filter(
            target_x,
            now
        )

        smooth_y = self.filter_y.filter(
            target_y,
            now
        )

        smooth_x = clamp(
            smooth_x,
            0,
            self.screen_width - 1
        )

        smooth_y = clamp(
            smooth_y,
            0,
            self.screen_height - 1
        )

        pyautogui.moveTo(
            int(smooth_x),
            int(smooth_y)
        )


    # ========================================================
    # LEFT PINCH
    # ========================================================

    def update_left_pinch(
        self,
        pinch,
        now
    ):

        if self.left_pinching:

            active = (
                pinch
                <
                0.52
            )

        else:

            active = (
                pinch
                <
                0.40
            )


        if (
            active
            and
            not self.left_pinching
        ):

            self.left_pinching = True

            self.left_start = now

            return


        if (
            active
            and
            self.left_pinching
        ):

            duration = (
                now
                -
                self.left_start
            )

            if (
                duration >= 0.48
                and
                not self.dragging
            ):

                self.dragging = True

                pyautogui.mouseDown()

                self.set_action(
                    "DRAGGING"
                )

            return


        if (
            not active
            and
            self.left_pinching
        ):

            duration = (
                now
                -
                self.left_start
            )

            if self.dragging:

                pyautogui.mouseUp()

                self.dragging = False

                self.set_action(
                    "RELEASE"
                )

            elif (
                0.04
                <= duration
                <= 0.30
                and
                (
                    now
                    -
                    self.last_click_time
                )
                >= 0.16
            ):

                pyautogui.click()

                self.last_click_time = now

                self.set_action(
                    "LEFT CLICK"
                )

            self.left_pinching = False

            self.left_start = 0.0


    # ========================================================
    # RIGHT PINCH
    # ========================================================

    def update_right_pinch(
        self,
        pinch,
        now
    ):

        if self.right_pinching:

            active = (
                pinch
                <
                0.52
            )

        else:

            active = (
                pinch
                <
                0.40
            )


        if (
            active
            and
            not self.right_pinching
        ):

            self.right_pinching = True

            self.right_start = now

            return


        if (
            not active
            and
            self.right_pinching
        ):

            duration = (
                now
                -
                self.right_start
            )

            if (
                0.04
                <= duration
                <= 0.30
                and
                not self.left_pinching
                and
                not self.dragging
                and
                (
                    now
                    -
                    self.last_click_time
                )
                >= 0.16
            ):

                pyautogui.rightClick()

                self.last_click_time = now

                self.set_action(
                    "RIGHT CLICK"
                )

            self.right_pinching = False

            self.right_start = 0.0


    # ========================================================
    # SCROLL
    # ========================================================

    def update_scroll(
        self,
        index_y,
        now
    ):

        if self.scroll_y is None:

            self.scroll_y = index_y

            return


        dy = (
            index_y
            -
            self.scroll_y
        )

        self.scroll_y = index_y


        if abs(dy) < 0.0025:

            return


        if (
            now
            -
            self.last_scroll
            <
            0.015
        ):

            return


        amount = clamp(
            abs(dy) * 1000,
            1,
            7
        )


        if dy < 0:

            pyautogui.scroll(
                int(amount)
            )

        else:

            pyautogui.scroll(
                -int(amount)
            )


        self.last_scroll = now


    # ========================================================
    # FIST / OPEN PALM
    # ========================================================

    def update_hand_gestures(
        self,
        fingers,
        now
    ):

        extended = count_extended(
            fingers
        )

        # ----------------------------------------------------
        # CLOSED FIST
        # ----------------------------------------------------

        fist = (
            extended <= 1
            and
            not self.left_pinching
            and
            not self.right_pinching
            and
            not self.dragging
        )


        if fist:

            if self.fist_since is None:

                self.fist_since = now

            if (
                not self.fist_triggered
                and
                now
                -
                self.fist_since
                >=
                self.GESTURE_HOLD
                and
                now
                -
                self.last_action_time
                >=
                self.gesture_cooldown
            ):

                minimize_all_windows()

                self.fist_triggered = True

                self.last_action_time = now

                self.set_action(
                    "SHOW DESKTOP"
                )

        else:

            self.fist_since = None
            self.fist_triggered = False


        # ----------------------------------------------------
        # OPEN PALM
        # ----------------------------------------------------

        palm = (
            extended >= 5
            and
            not self.left_pinching
            and
            not self.right_pinching
            and
            not self.dragging
        )


        if palm:

            if self.palm_since is None:

                self.palm_since = now

            if (
                not self.palm_triggered
                and
                now
                -
                self.palm_since
                >=
                self.GESTURE_HOLD
                and
                now
                -
                self.last_action_time
                >=
                self.gesture_cooldown
            ):

                restore_all_windows()

                self.palm_triggered = True

                self.last_action_time = now

                self.set_action(
                    "RESTORE DESKTOP"
                )

        else:

            self.palm_since = None
            self.palm_triggered = False


    # ========================================================
    # THREE-FINGER SWIPE
    # ========================================================

    def update_three_finger_swipe(
        self,
        landmarks,
        fingers,
        now
    ):

        three = (
            fingers["index"]
            and
            fingers["middle"]
            and
            fingers["ring"]
            and
            not fingers["pinky"]
            and
            not self.left_pinching
            and
            not self.right_pinching
            and
            not self.dragging
        )


        if not three:

            self.three_start_x = None
            self.three_start_time = None
            self.swipe_triggered = False

            return


        center_x = (
            landmarks[8].x
            +
            landmarks[12].x
            +
            landmarks[16].x
        ) / 3


        if self.three_start_x is None:

            self.three_start_x = center_x

            self.three_start_time = now

            self.swipe_triggered = False

            return


        dx = (
            center_x
            -
            self.three_start_x
        )


        elapsed = (
            now
            -
            self.three_start_time
        )


        if (
            not self.swipe_triggered
            and
            elapsed < 0.70
            and
            abs(dx) > 0.18
            and
            now
            -
            self.last_action_time
            >=
            self.gesture_cooldown
        ):

            if dx < 0:

                browser_back()

                self.set_action(
                    "BACK"
                )

            else:

                browser_forward()

                self.set_action(
                    "FORWARD"
                )


            self.swipe_triggered = True

            self.last_action_time = now


    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process(
        self,
        landmarks,
        now
    ):

        # ----------------------------------------------------
        # POINTER ALWAYS FIRST
        # ----------------------------------------------------

        self.move_pointer(
            landmarks[8],
            now
        )


        fingers = get_fingers(
            landmarks
        )


        # ----------------------------------------------------
        # Pinches
        # ----------------------------------------------------

        left_pinch = relative_distance(
            landmarks,
            4,
            8
        )

        right_pinch = relative_distance(
            landmarks,
            4,
            12
        )


        self.update_left_pinch(
            left_pinch,
            now
        )

        self.update_right_pinch(
            right_pinch,
            now
        )


        # ----------------------------------------------------
        # Scroll
        # ----------------------------------------------------

        scroll = (
            fingers["index"]
            and
            fingers["middle"]
            and
            not fingers["ring"]
            and
            not fingers["pinky"]
            and
            not self.left_pinching
            and
            not self.right_pinching
            and
            not self.dragging
        )


        if scroll:

            self.update_scroll(
                landmarks[8].y,
                now
            )

        else:

            self.scroll_y = None


        # ----------------------------------------------------
        # Fist / palm
        # ----------------------------------------------------

        self.update_hand_gestures(
            fingers,
            now
        )


        # ----------------------------------------------------
        # Three-finger swipe
        # ----------------------------------------------------

        self.update_three_finger_swipe(
            landmarks,
            fingers,
            now
        )


        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if self.dragging:

            self.status = "DRAG"

        elif scroll:

            self.status = "SCROLL"

        elif self.left_pinching:

            self.status = "PINCH"

        elif self.right_pinching:

            self.status = "RIGHT"

        elif count_extended(fingers) == 0:

            self.status = "FIST"

        elif count_extended(fingers) >= 5:

            self.status = "OPEN PALM"

        else:

            self.status = "TRACKING"


        # ----------------------------------------------------
        # Trail
        # ----------------------------------------------------

        self.trail.append(
            (
                landmarks[8].x,
                landmarks[8].y
            )
        )

        if len(self.trail) > 18:

            self.trail.pop(0)


        return fingers, scroll


# ============================================================
# HUD
# ============================================================

def overlay_box(
    frame,
    x1,
    y1,
    x2,
    y2,
    opacity=0.82
):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (9, 14, 20),
        -1
    )

    cv2.addWeighted(
        overlay,
        opacity,
        frame,
        1.0 - opacity,
        0,
        frame
    )


def draw_hud(
    frame,
    controller,
    fingers,
    scroll,
    fps
):

    h, w, _ = frame.shape


    # ========================================================
    # TOP BAR
    # ========================================================

    overlay_box(
        frame,
        14,
        14,
        300,
        78,
        0.88
    )


    cv2.putText(
        frame,
        "JARVIS",
        (28, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (225, 245, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        "AIR INTERFACE",
        (29, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.29,
        (90, 205, 240),
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # STATUS
    # ========================================================

    status_colors = {

        "TRACKING":
            (90, 225, 255),

        "PINCH":
            (90, 240, 150),

        "RIGHT":
            (255, 180, 90),

        "DRAG":
            (80, 235, 255),

        "SCROLL":
            (205, 130, 255),

        "FIST":
            (255, 150, 90),

        "OPEN PALM":
            (100, 240, 160),
    }


    status_color = status_colors.get(
        controller.status,
        (100, 210, 240)
    )


    cv2.circle(
        frame,
        (326, 39),
        5,
        status_color,
        -1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        controller.status,
        (340, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        status_color,
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # RIGHT CONTROL PANEL
    # ========================================================

    panel_w = 265

    panel_x = (
        w
        -
        panel_w
        -
        14
    )


    overlay_box(
        frame,
        panel_x,
        14,
        w - 14,
        310,
        0.90
    )


    cv2.putText(
        frame,
        "GESTURE CONTROL",
        (panel_x + 18, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (225, 242, 250),
        1,
        cv2.LINE_AA
    )


    cv2.line(
        frame,
        (panel_x + 18, 53),
        (w - 30, 53),
        (55, 80, 92),
        1
    )


    controls = [

        ("INDEX", "POINTER"),

        ("PINCH", "LEFT CLICK"),

        ("PINCH HOLD", "DRAG"),

        ("THUMB + MIDDLE", "RIGHT CLICK"),

        ("2 FINGERS", "SCROLL"),

        ("CLOSED FIST", "MINIMIZE ALL"),

        ("OPEN PALM", "RESTORE ALL"),

        ("3 FINGERS SWIPE", "BACK / FORWARD"),
    ]


    y = 76


    for gesture, operation in controls:

        cv2.putText(
            frame,
            gesture,
            (panel_x + 18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.285,
            (90, 180, 210),
            1,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            operation,
            (panel_x + 18, y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (232, 240, 245),
            1,
            cv2.LINE_AA
        )


        y += 29


    # ========================================================
    # ACTION PANEL
    # ========================================================

    overlay_box(
        frame,
        14,
        h - 70,
        305,
        h - 14,
        0.88
    )


    cv2.putText(
        frame,
        "ACTION",
        (28, h - 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.27,
        (90, 180, 210),
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        controller.action,
        (28, h - 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (235, 245, 250),
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # FPS
    # ========================================================

    cv2.putText(
        frame,
        f"{fps:.0f} FPS",
        (w - 88, h - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (90, 180, 210),
        1,
        cv2.LINE_AA
    )


# ============================================================
# HAND VISUALS
# ============================================================

def draw_hand(
    frame,
    landmarks,
    controller
):

    h, w, _ = frame.shape


    connections = [

        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        (0, 17),
    ]


    for a, b in connections:

        x1 = int(
            landmarks[a].x * w
        )

        y1 = int(
            landmarks[a].y * h
        )

        x2 = int(
            landmarks[b].x * w
        )

        y2 = int(
            landmarks[b].y * h
        )


        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (55, 125, 155),
            1,
            cv2.LINE_AA
        )


    # ========================================================
    # INDEX TRAIL
    # ========================================================

    for i in range(
        1,
        len(controller.trail)
    ):

        x1 = int(
            controller.trail[i - 1][0]
            *
            w
        )

        y1 = int(
            controller.trail[i - 1][1]
            *
            h
        )

        x2 = int(
            controller.trail[i][0]
            *
            w
        )

        y2 = int(
            controller.trail[i][1]
            *
            h
        )


        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (60, 145, 180),
            1,
            cv2.LINE_AA
        )


    # ========================================================
    # FINGERTIPS
    # ========================================================

    for tip in [
        4,
        8,
        12,
        16,
        20
    ]:

        x = int(
            landmarks[tip].x * w
        )

        y = int(
            landmarks[tip].y * h
        )


        cv2.circle(
            frame,
            (x, y),
            4,
            (105, 225, 255),
            -1,
            cv2.LINE_AA
        )


    # ========================================================
    # INDEX RETICLE
    # ========================================================

    ix = int(
        landmarks[8].x * w
    )

    iy = int(
        landmarks[8].y * h
    )


    cv2.circle(
        frame,
        (ix, iy),
        16,
        (90, 220, 255),
        1,
        cv2.LINE_AA
    )


    cv2.line(
        frame,
        (ix - 24, iy),
        (ix - 7, iy),
        (90, 220, 255),
        1
    )

    cv2.line(
        frame,
        (ix + 7, iy),
        (ix + 24, iy),
        (90, 220, 255),
        1
    )

    cv2.line(
        frame,
        (ix, iy - 24),
        (ix, iy - 7),
        (90, 220, 255),
        1
    )

    cv2.line(
        frame,
        (ix, iy + 7),
        (ix, iy + 24),
        (90, 220, 255),
        1
    )


# ============================================================
# DISABLE WINDOW CLOSE BUTTON
# Windows only.
# ============================================================

def disable_close_button(
    window_title
):

    if os.name != "nt":
        return

    try:

        hwnd = ctypes.windll.user32.FindWindowW(
            None,
            window_title
        )

        if not hwnd:
            return

        SC_CLOSE = 0xF060
        MF_BYCOMMAND = 0x00000000

        ctypes.windll.user32.DeleteMenu(
            hwnd,
            SC_CLOSE,
            MF_BYCOMMAND
        )

    except Exception:

        pass


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("                    JARVIS AIR CONTROL")
    print("=" * 72)
    print()
    print("INDEX                 -> Pointer")
    print("THUMB + INDEX         -> Left click")
    print("HOLD THUMB + INDEX    -> Drag")
    print("THUMB + MIDDLE        -> Right click")
    print("INDEX + MIDDLE        -> Scroll")
    print("CLOSED FIST           -> Minimize all windows")
    print("OPEN PALM             -> Restore all windows")
    print("3 FINGERS SWIPE       -> Back / Forward")
    print()
    print("The camera window cannot be closed normally.")
    print("Move the PHYSICAL mouse to the TOP-LEFT for emergency stop.")
    print()
    print("=" * 72)
    print()


    ensure_model()


    detector = None
    camera = None
    controller = None


    try:

        detector = create_detector()


        screen_width, screen_height = (
            pyautogui.size()
        )


        controller = JarvisController(
            screen_width,
            screen_height
        )


        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        camera = cv2.VideoCapture(
            CAMERA_INDEX
        )


        if not camera.isOpened():

            raise RuntimeError(
                "Could not open webcam."
            )


        camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH
        )

        camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT
        )

        # Small camera buffer.
        # Prevents processing old frames.
        camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )


        WINDOW_NAME = "JARVIS AIR CONTROL"


        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL
        )


        cv2.resizeWindow(
            WINDOW_NAME,
            CAMERA_WIDTH,
            CAMERA_HEIGHT
        )


        # Try to disable the Windows close button.
        disable_close_button(
            WINDOW_NAME
        )


        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        previous_time = (
            time.perf_counter()
        )

        fps = 0.0


        # ====================================================
        # LOOP
        # ====================================================

        while True:

            success, frame = (
                camera.read()
            )


            if not success:

                continue


            # Mirror.
            frame = cv2.flip(
                frame,
                1
            )


            # RGB.
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            mp_image = mp.Image(
                image_format=(
                    mp.ImageFormat.SRGB
                ),
                data=rgb
            )


            result = detector.detect(
                mp_image
            )


            now = time.perf_counter()


            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            dt = max(
                now
                -
                previous_time,
                0.0001
            )

            previous_time = now


            instant_fps = (
                1.0 / dt
            )


            fps = (
                fps * 0.90
                +
                instant_fps * 0.10
            )


            # ------------------------------------------------
            # Hand
            # ------------------------------------------------

            if result.hand_landmarks:

                landmarks = (
                    result.hand_landmarks[0]
                )


                fingers, scroll = (
                    controller.process(
                        landmarks,
                        now
                    )
                )


                draw_hand(
                    frame,
                    landmarks,
                    controller
                )


            else:

                # Hand disappeared.
                #
                # Stop cursor movement immediately and reset
                # filters so the hand cannot jump when it
                # comes back.

                controller.filter_x.reset()
                controller.filter_y.reset()

                controller.scroll_y = None
                controller.trail.clear()

                controller.status = (
                    "SEARCHING"
                )


                fingers = {}


                cv2.putText(
                    frame,
                    "HAND SEARCHING...",
                    (18, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (90, 180, 215),
                    1,
                    cv2.LINE_AA
                )


            # ------------------------------------------------
            # HUD
            # ------------------------------------------------

            draw_hud(
                frame,
                controller,
                fingers,
                False,
                fps
            )


            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            cv2.imshow(
                WINDOW_NAME,
                frame
            )


            # ------------------------------------------------
            # No Q exit.
            #
            # waitKey is still needed to keep the OpenCV
            # window responsive.
            #
            # Any key is ignored.
            # ------------------------------------------------

            cv2.waitKey(1)


    except KeyboardInterrupt:

        print()
        print(
            "Stopped from terminal."
        )


    except Exception as error:

        print()
        print(
            "ERROR:",
            error
        )


        input(
            "\nPress Enter to return to terminal..."
        )


    finally:

        # Release mouse if program is interrupted during drag.
        if (
            controller is not None
            and
            controller.dragging
        ):

            try:

                pyautogui.mouseUp()

            except Exception:

                pass


        if camera is not None:

            camera.release()


        cv2.destroyAllWindows()


        if detector is not None:

            detector.close()


        print()
        print(
            "JARVIS stopped."
        )


if __name__ == "__main__":

    main()
