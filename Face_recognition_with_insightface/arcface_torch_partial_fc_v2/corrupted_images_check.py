import glob,shutil
import cv2,os,tqdm
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

folder_path = ''
images = glob.glob(folder_path+'/*/*')
corrupted_number = 0

for image_path in tqdm.tqdm(images,desc='progress'):
    # print(image_path)
    try:
        img = tf.keras.preprocessing.image.load_img(image_path)
        img = tf.keras.preprocessing.image.img_to_array(img)
        shape = img.shape
    except Exception as E:
        corrupted_number+=1
        os.remove(image_path)
        print(E,image_path)

print(corrupted_number)
print('done')