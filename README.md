# STEM Smart Mirror
This is the software for the smart mirror
## What This Software Can Do
- Overlay clothes on a user in frame to try out the fit
- Allows user to use their hand to control the mouse
## Instructions
- Setup: 
    1. Open a terminal in the code folder of this program
    2. Refer to the requirements for what this program needs
    3. Run the following command depending on OS and configuration: 
        - Windows: `python 08_skeleton_3D.py -m body -c 1`
        - macOS/Linux: `python3 08_skeleton_3D.py -m body`
    4. To make it ask you which camera to use (to check which camera is the correct one), remove the `-c 1` argument, once you found the correct camera index, take note of it and specify the `-c #` argument replacing the # with the index
- Actual Use:
    - IMPORTANT: To close the software, focus on the `img 2D` window and hold `Esc` and the program will shutdown safely, while possible it is advised against using ctrl+c cuz the program has to cleanup
    - To try on clothes ~~select from the options at the bottom~~
    - To interact with the system, just move your hand to wherever you want to click and the mouse will follow. To click, just pinch your fingers together and open after a second. To click and drag, just keep pinching until you've dragged your target to wherever and release
## Requirements
- Python 3
### Python Requirements
- Open3d
- opencv
- Mediapipe
- pyautogui
- argparse
### System Requirements
- x86_64 systems only, no Apple M series chips or any other ARM chips
- A good CPU
- NVIDIA GPU (optional)
## Common Errors
- opencv not found, pip error
    - If you are on a Linux or macOS system, opencv is called `opencv-python` not `opencv`
- pip not found
    - If you are on Linux/macOS then the command is probably `python3 -m pip `
    - If windows, make sure python is on PATH
- pip module not found
    - You are probably using the wrong python install, check your installs using `where python3` for Linux/macOS or `where python` for Windows
    - If you are on Linux, it probably has to be installed using `sudo apt-get install pip` or `sudo apt-get install python3-pip`
    - If all else fails, reinstall python: 
        - Windows: Use installer, make sure it is using the windows cmd and it is added to PATH
        - Linux: `sudo apt-get install python3`
        - macOS: Same thing as Windows
- found arm64 but x86_64 installed (Common issue on macOS)
    - ! Note: This program was not built or intended to run on ARM CPUs, if you have a macbook, go to the apple logo -> About this Mac -> First line: if it says something like Intel Core ..., then it will work fine, if its something like Apple M1 or smth, then it will not work with this software
    - Make sure your terminal isn't spoofing your system as an architecture its not
    - use `arch -x86_x64 ...` to make sure the command is run as x86_64, replace ... with your command

lmk for any other issues


## Debug
This project has a debug config called `Python Debugger: AI program with camera index 1`. 

Make sure you change -c argument to the correct camera index or remove it to always ask which camera to use