import pyautogui
import time
import psutil

print("Hand is open")
pyautogui.press('win')
time.sleep(1)
pyautogui.write('https://www.youtube.com/watch?v=lm4OJxGQm_E&list=PLxZrqnUosEl3U7SyF7v3zcxRiW5Tu1dYZ')
time.sleep(2)
pyautogui.press('enter')
time.sleep(9)
pyautogui.press('enter')

program_name = 'msedge.exe'
timeout = time.time() + 120
while True:
    for process in psutil.process_iter():
        try:
            if process.name() == program_name:
                print('Youtube is open!')
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



