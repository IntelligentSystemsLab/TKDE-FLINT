import pandas as pd
import os
import shutil

original_data_dir = './state-farm-distracted-driver-detection'
new_data_dir = 'reorganized_data'

csv_path = os.path.join(original_data_dir, 'driver_imgs_list.csv')
df = pd.read_csv(csv_path)

os.makedirs(new_data_dir, exist_ok=True)

for _, row in df.iterrows():
    driver_id = row['subject']
    class_name = row['classname']
    img_name = row['img']

    src_path = os.path.join(original_data_dir, 'imgs/train', class_name, img_name)
    dst_driver_dir = os.path.join(new_data_dir, driver_id)
    dst_class_dir = os.path.join(dst_driver_dir, class_name)

    os.makedirs(dst_class_dir, exist_ok=True)

    shutil.copy(src_path, os.path.join(dst_class_dir, img_name))

print(f"{os.path.abspath(new_data_dir)}")