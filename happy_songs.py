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

    pyautogui.moveTo(932,497,duration=2)
    pyautogui.click()

    pyautogui.moveTo(1220,485,duration=2)
    pyautogui.click()

    time.sleep(5)

    pyautogui.moveTo(1857, 93, duration=2)
    pyautogui.click()

    pyautogui.moveTo(874, 423, duration=2)
    pyautogui.click()
    pyautogui.hotkey('ctrl', 'a')
    pyautogui.press('delete')

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
