import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main_spotify():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == "Not applicable":  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )

                # Performs operations on spotify ###########################################################
                def perform_action_for_hand_sign(hand_sign_id):
                    if hand_sign_id == 0:  # TO OPEN THE SPOTIFY APP
                        exec(open(r"C:\hand-gesture-recognition-mediapipe-main\open_n_play.py").read())
                    elif hand_sign_id == 1:  # PLAY PAUSE
                        import pyautogui
                        import time
                        import psutil
                        import win32gui
                        import win32process
                        from pywinauto import Application

                        print("Hand is closed")

                        time.sleep(5)

                        def get_spotify_window_title(pids):
                            titles = []
                            returnpid = 0

                            def _enum_cb(hwnd, results):
                                if win32gui.IsWindowVisible(hwnd):
                                    pid = win32process.GetWindowThreadProcessId(hwnd)[1]
                                    if pids is None or pid in pids:
                                        nonlocal returnpid
                                        returnpid = pid

                            win32gui.EnumWindows(_enum_cb, titles)
                            return returnpid

                        def press_key(spotify_pids):
                            app = Application().connect(process=get_spotify_window_title(spotify_pids))
                            app.top_window().set_focus()
                            time.sleep(1)
                            pyautogui.press('space')
                            # window = app.top_window()
                            # window.minimize()

                        program_name = 'Spotify.exe'
                        process_ids = []
                        timeout = time.time() + 120
                        isOpen = False
                        while True and time.time() < timeout:
                            for process in psutil.process_iter():
                                try:
                                    if program_name in process.name():
                                        isOpen = True
                                        process_ids.append(process.pid)
                                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                                    pass
                            else:
                                if isOpen:
                                    print("Spotify is OPEN")
                                    print("Play/Pause")
                                    time.sleep(1)
                                    press_key(process_ids)
                                else:
                                    print("Spotify is CLOSED")
                                break

                    elif hand_sign_id == 2:  # SKIP SONGS
                        import pyautogui
                        import time
                        import psutil
                        import win32gui
                        import win32process
                        from pywinauto import Application

                        def get_spotify_window_title(pids):
                            titles = []
                            returnpid = 0

                            def _enum_cb(hwnd, results):
                                if win32gui.IsWindowVisible(hwnd):
                                    pid = win32process.GetWindowThreadProcessId(hwnd)[1]
                                    if pids is None or pid in pids:
                                        nonlocal returnpid
                                        returnpid = pid

                            win32gui.EnumWindows(_enum_cb, titles)
                            return returnpid

                        program_name = 'Spotify.exe'

                        timeout = time.time() + 120
                        isOpen = False
                        for process in psutil.process_iter():
                            try:
                                if process.name() == program_name:
                                    print("Spotify is open!")
                                    print("Skipped to next song")
                                    isOpen = True
                                    break
                            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                                pass
                        else:
                            print("Spotify is not Open!")
                            exit()

                        if isOpen:
                            spotify_pids = []
                            process_name = "Spotify.exe"
                            for proc in psutil.process_iter():
                                if process_name in proc.name():
                                    spotify_pids.append(proc.pid)

                            app = Application().connect(process=get_spotify_window_title(spotify_pids))
                            app.top_window().set_focus()
                            time.sleep(1)
                            pyautogui.hotkey('ctrl', 'right', interval=0.25)
                    elif hand_sign_id == 3:  # LIKE SONGS
                        import pyautogui
                        import time
                        import psutil
                        import win32gui
                        import win32process
                        from pywinauto import Application

                        def get_spotify_window_title(pids):
                            titles = []
                            returnpid = 0

                            def _enum_cb(hwnd, results):
                                if win32gui.IsWindowVisible(hwnd):
                                    pid = win32process.GetWindowThreadProcessId(hwnd)[1]
                                    if pids is None or pid in pids:
                                        nonlocal returnpid
                                        returnpid = pid

                            win32gui.EnumWindows(_enum_cb, titles)
                            return returnpid

                        program_name = 'Spotify.exe'

                        timeout = time.time() + 120
                        isOpen = False
                        for process in psutil.process_iter():
                            try:
                                if process.name() == program_name:
                                    print("Spotify is open!")
                                    print("Added to Liked Songs")
                                    isOpen = True
                                    break
                            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                                pass
                        else:
                            print("Spotify is not Open!")
                            exit()

                        if isOpen:
                            spotify_pids = []
                            process_name = "Spotify.exe"
                            for proc in psutil.process_iter():
                                if process_name in proc.name():
                                    spotify_pids.append(proc.pid)

                            app = Application().connect(process=get_spotify_window_title(spotify_pids))
                            app.top_window().set_focus()
                            time.sleep(1)
                            pyautogui.hotkey('alt', 'shift', 'b', interval=0.25)
                    elif hand_sign_id == 4:  # PREV SONG
                        import pyautogui
                        import time
                        import psutil
                        import win32gui
                        import win32process
                        from pywinauto import Application

                        def get_spotify_window_title(pids):
                            titles = []
                            returnpid = 0

                            def _enum_cb(hwnd, results):
                                if win32gui.IsWindowVisible(hwnd):
                                    pid = win32process.GetWindowThreadProcessId(hwnd)[1]
                                    if pids is None or pid in pids:
                                        nonlocal returnpid
                                        returnpid = pid

                            win32gui.EnumWindows(_enum_cb, titles)
                            return returnpid

                        program_name = 'Spotify.exe'

                        timeout = time.time() + 120
                        isOpen = False
                        for process in psutil.process_iter():
                            try:
                                if process.name() == program_name:
                                    print("Spotify is open!")
                                    print("Play Previous Track")
                                    isOpen = True
                                    break
                            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                                pass
                        else:
                            print("Spotify is not Open!")
                            exit()

                        if isOpen:
                            spotify_pids = []
                            process_name = "Spotify.exe"
                            for proc in psutil.process_iter():
                                if process_name in proc.name():
                                    spotify_pids.append(proc.pid)

                            app = Application().connect(process=get_spotify_window_title(spotify_pids))
                            app.top_window().set_focus()
                            time.sleep(1)
                            pyautogui.hotkey('ctrl', 'left', interval=0.25)
                            pyautogui.hotkey('ctrl', 'left', interval=0.25)

                    elif hand_sign_id == 5:  # GENRE - POP
                        import keyboard
                        import pyautogui
                        import time
                        import psutil
                        import win32gui
                        import win32process
                        from pywinauto import Application

                        time.sleep(2)

                        def get_spotify_window_title(pids):
                            titles = []
                            returnpid = 0

                            def _enum_cb(hwnd, results):
                                if win32gui.IsWindowVisible(hwnd):
                                    pid = win32process.GetWindowThreadProcessId(hwnd)[1]
                                    if pids is None or pid in pids:
                                        nonlocal returnpid
                                        returnpid = pid

                            win32gui.EnumWindows(_enum_cb, titles)
                            return returnpid

                        def press_key(spotify_pids):
                            app = Application().connect(process=get_spotify_window_title(spotify_pids))
                            app.top_window().set_focus()
                            time.sleep(1)
                            keyboard.press_and_release('ctrl+alt+f')

                            keyboard.write('My Pop Playlist')
                            pyautogui.press('enter')

                            pyautogui.moveTo(932, 497, duration=2)
                            pyautogui.click()

                            pyautogui.moveTo(1220, 485, duration=2)
                            pyautogui.click()

                            time.sleep(3)

                            pyautogui.moveTo(1857, 93, duration=2)
                            pyautogui.click()

                            pyautogui.moveTo(874, 423, duration=2)
                            pyautogui.click()
                            pyautogui.hotkey('ctrl', 'a')
                            pyautogui.press('delete')

                            pyautogui.moveTo(x=1229, y=182, duration=2)
                            pyautogui.click()

                        program_name = 'Spotify.exe'
                        process_ids = []
                        timeout = time.time() + 120
                        isOpen = False
                        while True and time.time() < timeout:
                            for process in psutil.process_iter():
                                try:
                                    if program_name in process.name():
                                        isOpen = True
                                        process_ids.append(process.pid)
                                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                                    pass
                            else:
                                if isOpen:
                                    print("Spotify is OPEN")
                                    print("Playing My Pop Playlist")
                                    time.sleep(1)
                                    press_key(process_ids)
                                else:
                                    print("Spotify is CLOSED")
                                break

                    elif hand_sign_id == 6:  # MOOD - SAD
                        import keyboard
                        import pyautogui
                        import time
                        import psutil
                        import win32gui
                        import win32process
                        from pywinauto import Application

                        time.sleep(2)

                        def get_spotify_window_title(pids):
                            titles = []
                            returnpid = 0

                            def _enum_cb(hwnd, results):
                                if win32gui.IsWindowVisible(hwnd):
                                    pid = win32process.GetWindowThreadProcessId(hwnd)[1]
                                    if pids is None or pid in pids:
                                        nonlocal returnpid
                                        returnpid = pid

                            win32gui.EnumWindows(_enum_cb, titles)
                            return returnpid

                        def press_key(spotify_pids):
                            app = Application().connect(process=get_spotify_window_title(spotify_pids))
                            app.top_window().set_focus()
                            time.sleep(1)
                            keyboard.press_and_release('ctrl+alt+f')

                            keyboard.write('My Sad Playlist')
                            pyautogui.press('enter')

                            pyautogui.moveTo(932, 497, duration=2)
                            pyautogui.click()

                            pyautogui.moveTo(1220, 485, duration=2)
                            pyautogui.click()

                            time.sleep(3)

                            pyautogui.moveTo(1857, 93, duration=2)
                            pyautogui.click()

                            pyautogui.moveTo(874, 423, duration=2)
                            pyautogui.click()
                            pyautogui.hotkey('ctrl', 'a')
                            pyautogui.press('delete')

                            pyautogui.moveTo(x=1229, y=182, duration=2)
                            pyautogui.click()

                        program_name = 'Spotify.exe'
                        process_ids = []
                        timeout = time.time() + 120
                        isOpen = False
                        while True and time.time() < timeout:
                            for process in psutil.process_iter():
                                try:
                                    if program_name in process.name():
                                        isOpen = True
                                        process_ids.append(process.pid)
                                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                                    pass
                            else:
                                if isOpen:
                                    print("Spotify is OPEN")
                                    print("Playing My Sad Playlist")
                                    time.sleep(1)
                                    press_key(process_ids)
                                else:
                                    print("Spotify is CLOSED")
                                break

                    elif hand_sign_id == 7: # MOOD - HAPPY
                        import keyboard
                        import pyautogui
                        import time
                        import psutil
                        import win32gui
                        import win32process
                        from pywinauto import Application

                        time.sleep(2)

                        def get_spotify_window_title(pids):
                            titles = []
                            returnpid = 0

                            def _enum_cb(hwnd, results):
                                if win32gui.IsWindowVisible(hwnd):
                                    pid = win32process.GetWindowThreadProcessId(hwnd)[1]
                                    if pids is None or pid in pids:
                                        nonlocal returnpid
                                        returnpid = pid

                            win32gui.EnumWindows(_enum_cb, titles)
                            return returnpid

                        def press_key(spotify_pids):
                            app = Application().connect(process=get_spotify_window_title(spotify_pids))
                            app.top_window().set_focus()
                            time.sleep(1)
                            keyboard.press_and_release('ctrl+alt+f')

                            keyboard.write('My Happy Playlist')
                            pyautogui.press('enter')

                            pyautogui.moveTo(932, 497, duration=2)
                            pyautogui.click()

                            pyautogui.moveTo(1220, 485, duration=2)
                            pyautogui.click()

                            time.sleep(3)

                            pyautogui.moveTo(1857, 93, duration=2)
                            pyautogui.click()

                            pyautogui.moveTo(874, 423, duration=2)
                            pyautogui.click()
                            pyautogui.hotkey('ctrl', 'a')
                            pyautogui.press('delete')

                            pyautogui.moveTo(x=1229, y=182, duration=2)
                            pyautogui.click()

                        program_name = 'Spotify.exe'
                        process_ids = []
                        timeout = time.time() + 120
                        isOpen = False
                        while True and time.time() < timeout:
                            for process in psutil.process_iter():
                                try:
                                    if program_name in process.name():
                                        isOpen = True
                                        process_ids.append(process.pid)
                                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                                    pass
                            else:
                                if isOpen:
                                    print("Spotify is OPEN")
                                    print("Playing My Happy Playlist")
                                    time.sleep(1)
                                    press_key(process_ids)
                                else:
                                    print("Spotify is CLOSED")
                                break
                    elif hand_sign_id == 8: # GENRE - ROCK
                        import keyboard
                        import pyautogui
                        import time
                        import psutil
                        import win32gui
                        import win32process
                        from pywinauto import Application

                        time.sleep(2)

                        def get_spotify_window_title(pids):
                            titles = []
                            returnpid = 0

                            def _enum_cb(hwnd, results):
                                if win32gui.IsWindowVisible(hwnd):
                                    pid = win32process.GetWindowThreadProcessId(hwnd)[1]
                                    if pids is None or pid in pids:
                                        nonlocal returnpid
                                        returnpid = pid

                            win32gui.EnumWindows(_enum_cb, titles)
                            return returnpid

                        def press_key(spotify_pids):
                            app = Application().connect(process=get_spotify_window_title(spotify_pids))
                            app.top_window().set_focus()
                            time.sleep(1)
                            keyboard.press_and_release('ctrl+alt+f')

                            keyboard.write('My Rock Playlist')
                            pyautogui.press('enter')

                            pyautogui.moveTo(932, 497, duration=2)
                            pyautogui.click()

                            pyautogui.moveTo(1220, 485, duration=2)
                            pyautogui.click()

                            time.sleep(3)

                            pyautogui.moveTo(1857, 93, duration=2)
                            pyautogui.click()

                            pyautogui.moveTo(874, 423, duration=2)
                            pyautogui.click()
                            pyautogui.hotkey('ctrl', 'a')
                            pyautogui.press('delete')

                            pyautogui.moveTo(x=1229, y=182, duration=2)
                            pyautogui.click()

                        program_name = 'Spotify.exe'
                        process_ids = []
                        timeout = time.time() + 120
                        isOpen = False
                        while True and time.time() < timeout:
                            for process in psutil.process_iter():
                                try:
                                    if program_name in process.name():
                                        isOpen = True
                                        process_ids.append(process.pid)
                                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                                    pass
                            else:
                                if isOpen:
                                    print("Spotify is OPEN")
                                    print("Playing My Rock Playlist")
                                    time.sleep(1)
                                    press_key(process_ids)
                                else:
                                    print("Spotify is CLOSED")
                                break
                perform_action_for_hand_sign(hand_sign_id)
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    # if finger_gesture_text != "":
    #     cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
    #                cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    #     cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
    #                cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
    #                cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

def youtube_main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == "Not applicable":  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )

                # Performs operations on youtube ###########################################################
                def perform_action_for_hand_sign_youtube(hand_sign_id):
                    if hand_sign_id == 0:  # OPEN THE YOUTUBE APP
                        exec(open(r"C:\hand-gesture-recognition-mediapipe-main\youtube_open.py").read())
                    elif hand_sign_id ==1: #PLAY PAUSE
                        import pyautogui
                        import time
                        import psutil
                        import win32gui
                        import win32process
                        from pywinauto import Application

                        print("Hand is closed")

                        time.sleep(5)

                        def get_youtube_window_title(pids):
                            titles = []
                            returnpid = 0

                            def _enum_cb(hwnd, results):
                                if win32gui.IsWindowVisible(hwnd):
                                    pid = win32process.GetWindowThreadProcessId(hwnd)[1]
                                    if pids is None or pid in pids:
                                        nonlocal returnpid
                                        returnpid = pid

                            win32gui.EnumWindows(_enum_cb, titles)
                            return returnpid

                        def press_key(youtube_pids):
                            app = Application().connect(process=get_youtube_window_title(youtube_pids))
                            app.top_window().set_focus()
                            time.sleep(1)
                            pyautogui.press('space')
                            # window = app.top_window()
                            # window.minimize()

                        program_name = 'msedge.exe'
                        process_ids = []
                        timeout = time.time() + 120
                        isOpen = False
                        while True and time.time() < timeout:
                            for process in psutil.process_iter():
                                try:
                                    if program_name in process.name():
                                        isOpen = True
                                        process_ids.append(process.pid)
                                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                                    pass
                            else:
                                if isOpen:
                                    print("Youtube is OPEN")
                                    print("Play/Pause")
                                    time.sleep(1)
                                    press_key(process_ids)
                                else:
                                    print("Youtube is CLOSED")
                                break

                    elif hand_sign_id == 2:  # SKIP SONGS
                        import pyautogui
                        import time
                        import psutil
                        import win32gui
                        import win32process
                        from pywinauto import Application

                        def get_youtube_window_title(pids):
                            titles = []
                            returnpid = 0

                            def _enum_cb(hwnd, results):
                                if win32gui.IsWindowVisible(hwnd):
                                    pid = win32process.GetWindowThreadProcessId(hwnd)[1]
                                    if pids is None or pid in pids:
                                        nonlocal returnpid
                                        returnpid = pid

                            win32gui.EnumWindows(_enum_cb, titles)
                            return returnpid

                        program_name = 'msedge.exe'

                        timeout = time.time() + 120
                        isOpen = False
                        for process in psutil.process_iter():
                            try:
                                if process.name() == program_name:
                                    print("Youtube is open!")
                                    print("Skipped Video")
                                    isOpen = True
                                    break
                            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                                pass
                        else:
                            print("Youtube is not Open!")
                            exit()

                        if isOpen:
                            youtube_pids = []
                            process_name = "msedge.exe"
                            for proc in psutil.process_iter():
                                if process_name in proc.name():
                                    youtube_pids.append(proc.pid)

                            app = Application().connect(process=get_youtube_window_title(youtube_pids))
                            app.top_window().set_focus()
                            time.sleep(1)
                            pyautogui.hotkey('shift', 'n', interval=0.25)


                    elif hand_sign_id == 4:  # PREV SONG
                        import pyautogui
                        import time
                        import psutil
                        import win32gui
                        import win32process
                        from pywinauto import Application

                        def get_youtube_window_title(pids):
                            titles = []
                            returnpid = 0

                            def _enum_cb(hwnd, results):
                                if win32gui.IsWindowVisible(hwnd):
                                    pid = win32process.GetWindowThreadProcessId(hwnd)[1]
                                    if pids is None or pid in pids:
                                        nonlocal returnpid
                                        returnpid = pid

                            win32gui.EnumWindows(_enum_cb, titles)
                            return returnpid

                        program_name = 'msedge.exe'

                        timeout = time.time() + 120
                        isOpen = False
                        for process in psutil.process_iter():
                            try:
                                if process.name() == program_name:
                                    print("Youtube is open!")
                                    print("Playing Previous Track")
                                    isOpen = True
                                    break
                            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                                pass
                        else:
                            print("Youtube is not Open!")
                            exit()

                        if isOpen:
                            youtube_pids = []
                            process_name = "msedge.exe"
                            for proc in psutil.process_iter():
                                if process_name in proc.name():
                                    youtube_pids.append(proc.pid)

                            app = Application().connect(process=get_youtube_window_title(youtube_pids))
                            app.top_window().set_focus()
                            time.sleep(1)
                            pyautogui.hotkey('shift', 'p', interval=0.25)

                perform_action_for_hand_sign_youtube(hand_sign_id)
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()



if __name__ == '__main__':
    user_in = int(input("Enter 1 for Spotify or 2 for Youtube: "))
    if user_in == 1:
        main_spotify()
    elif user_in == 2:
        youtube_main()
    else:
        print("Enter valid inputs")