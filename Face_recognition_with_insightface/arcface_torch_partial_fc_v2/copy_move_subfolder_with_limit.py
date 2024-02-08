import shutil
import os
main_folder_path = "" #src folder path
des_path = "" #destination folder 
li1=os.listdir()
k=1
j=1
for i in li1:
    src_path = os.path.join(main_folder_path,i)
    try:
        if j%100==0:
            print(j)
        if j==429564: 
            break
        shutil.copytree(src_path,os.path.join(des_path,i))
        # shutil.move(src_path,des_path)
        j+=1
    except Exception as E:
        if k%100==0:
            print(j)
            print("folder is there",i,k,j)
        k+=1