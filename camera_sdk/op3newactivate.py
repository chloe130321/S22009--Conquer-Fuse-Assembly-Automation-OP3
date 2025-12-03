import os
import subprocess
import sys

# Path to your working Python interpreter (the one that works in VSCode)
PYTHON_EXE = r"C:\Users\功得\AppData\Local\Programs\Python\Python311\python.exe"

# Your camera script
APP_PATH = r"C:\3-1_3-3\camera_sdk\op3-1_3-3.py"
WORK_DIR = r"C:\3-1_3-3\camera_sdk"

# Add Huaray SDK runtime dirs so DLLs load
SDK_DIRS = [
    r"C:\Program Files\HuarayTech\MV Viewer\Runtime\x64",
    r"C:\Program Files\HuarayTech\MV Viewer\Runtime\3rdParty\x64",
]

def main():
    if not os.path.isfile(PYTHON_EXE):
        print(f"❌ Python not found at {PYTHON_EXE}")
        input("Press Enter to exit...")
        return

    if not os.path.isfile(APP_PATH):
        print(f"❌ Script not found: {APP_PATH}")
        input("Press Enter to exit...")
        return

    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([*(d for d in SDK_DIRS if os.path.isdir(d)), env.get("PATH", "")])

    cmd = [PYTHON_EXE, APP_PATH]
    print(f"Launching: {cmd}")
    try:
        subprocess.run(cmd, cwd=WORK_DIR, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"App crashed: {e}")
    finally:
        input("\nDone. Press Enter to close...")

if __name__ == "__main__":
    main()
