import os
import shutil
main_folder_path=''#source folder path
li=os.listdir(main_folder_path)
for i in li:
    if (len(os.listdir(os.path.join(main_folder_path,i)))<2) or (os.path.isfile(os.path.join(main_folder_path,i))):
        print(i)
        shutil.rmtree(os.path.join(main_folder_path,i))
        
        
