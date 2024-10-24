import os 

root = "dataset/papers/"

def count_files_in_folder(root):
    count = 0
    folders = os.listdir(root)
    print(folders)
    for folder in folders:
        items = os.listdir(root+folder)
        count += len(items)
    return print(count)    


count_files_in_folder(root)