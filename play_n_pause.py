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
