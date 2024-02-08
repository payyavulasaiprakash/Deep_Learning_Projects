import glob
main_folder = '' #source folder path
images = glob.glob(f'{main_folder}/*/*')
print(len(images))
# for image in images:
#     print(image)