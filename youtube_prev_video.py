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
            pid=win32process.GetWindowThreadProcessId(hwnd)[1]
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
            print("Playing Previous Video")
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

    app = Application(). connect(process=get_youtube_window_title(youtube_pids))
    app.top_window().set_focus()
    time.sleep(1)
    pyautogui.hotkey('shift','p', interval=0.25)

