import pyautogui
import time
import psutil

print("Hand is open")
pyautogui.press('win')
time.sleep(1)
pyautogui.write('Spotify')
time.sleep(2)
pyautogui.press('enter')

program_name = 'Spotify.exe'
timeout = time.time() + 120
while True:
    for process in psutil.process_iter():
        try:
            if process.name() == program_name:
                print('Spotify is open!')
                break
        except(psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    else:
        if time.time() > timeout:
            print("Timed out!")
            break
        else:
            time.sleep(1)
            continue
    break
time.sleep(10)
pyautogui.press('space')


