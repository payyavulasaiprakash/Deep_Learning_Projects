import easyocr 
import cv2, glob, os

#sample images are taken from google
images = glob.glob(os.path.join('sample_images','*'))

#output folder for saving the text detection of the input images
des_folder_path = "td_of_input_images"  

os.makedirs(des_folder_path,exist_ok=True)

languages = ['en']  #can include any language that is supported by easy ocr

def cv2_rect_text(image,top_left,bottom_right):
    image = cv2.rectangle(image, top_left,bottom_right, (0, 255, 0), 2)
    # image = cv2.putText(image, text, (top_left[0], top_left[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
    return image

for image_path in images:
    image_name = image_path.split(os.path.sep)[-1]
    img = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    reader = easyocr.Reader(languages, gpu=False)
    results = reader.readtext(img)
    for (bounding_box, text, confidence) in results:
        top_left, top_right, bottom_right, bottom_left = bounding_box
        image = cv2_rect_text(img,top_left,bottom_right)
        cv2.imwrite(os.path.join(des_folder_path, image_name),image)
        print("results ",results,"of image", image_name)


