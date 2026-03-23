import os

# Lokasi file ini (maincode.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Naik satu level ke mainfolder
TARGET_DIR = os.path.dirname(CURRENT_DIR)

# Masuk ke folder_1/folder_1.1/target
BASE_DIR = os.path.join(TARGET_DIR, "folder_1.1", "target")

print("BASE_DIR:", BASE_DIR)
print("TARGET_DIR:", TARGET_DIR)
