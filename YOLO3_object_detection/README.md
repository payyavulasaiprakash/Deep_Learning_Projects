Step1:

Create a python environment with 3.6.3 python version

Step2:

activate the environment

Step3:

install the requirements in that environment using - pip install -r requirements.txt
pip freeze > requirements.txt
Step4:

input_images_path - folder path consists of images for which detection to be done

output_images_path - folder path in which we save the bounding box drawn images

yolo3_weights_config - consists of 

1. coco.names - coco dataset labels

2. yolov3.cfg - architecture details

3. yolov3.weights - weights of the model, can download from https://pjreddie.com/media/files/yolov3.weights

Step6:

run python yolo3_open_source_cv2.py



