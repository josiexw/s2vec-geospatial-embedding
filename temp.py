import os

input_dir = './data/patches_info'

for file_name in os.listdir(input_dir):
    old_path = os.path.join(input_dir, file_name)

    if os.path.isfile(old_path):
        name, ext = os.path.splitext(file_name)
        new_name = f"{name.replace("__", "_")}{ext}"
        new_path = os.path.join(input_dir, new_name)

        os.rename(old_path, new_path)
        print(f"Renamed: {file_name} -> {new_name}")
