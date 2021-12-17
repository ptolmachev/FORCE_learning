import os
import re
from pathlib import Path

def get_project_root():
    return Path(__file__).parent.parent.parent

def create_dir_if_not_exist(path):
    try:
        os.makedirs(path, exist_ok=False)
    except:
        pass
    return None

def get_folders(root_folder, pattern):
    folders_all = os.listdir(root_folder + '/')
    folders = []
    for i, folder in enumerate(folders_all):
        m = re.search(pattern, str(folder))
        if m is not None:
            folders.append(folder)
    return folders

def get_files(root_folder, pattern):
    folders_and_files = os.listdir(root_folder + '/')
    files = []
    for i, el in enumerate(folders_and_files):
        if os.path.isfile(root_folder  + '/' + el):
            m = re.search(pattern, str(el))
            if m is not None:
                files.append(el)
    return files