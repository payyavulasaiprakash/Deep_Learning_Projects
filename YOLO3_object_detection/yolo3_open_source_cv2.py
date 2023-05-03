import numpy as np
import time
import cv2
import os

input_images_path = "input_images"

output_images_path = 'output_images'

images_list = os.listdir(input_images_path)

# image_path = 'input_images/coco-examples.jpg'

input_confidence = 0.5  # if confidence of the model about the class is more than this value then we take the bounding box, by this we will filter weak detections
#input_confidence value is more then we are making more strict

nms_threshold = 0.3  # Non Max Supression Threshold

coco_labels_path = os.path.join("yolo3_weights_config",'coco.names')

coco_labels = open(coco_labels_path).read().strip().split("\n")

np.random.seed(0)

colors_for_label = np.random.randint(0, 255, size=(len(coco_labels), 3),dtype="uint8")

#paths to weights and config file

yolo3_weights_path = os.path.join("yolo3_weights_config",'yolov3.weights')

yolo3_config_path = os.path.join("yolo3_weights_config",'yolov3.cfg')

#loading the YOLO object detector trained on 80 classes of COCO dataset

network = cv2.dnn.readNetFromDarknet(yolo3_config_path, yolo3_weights_path)


#to get the output layers of the YOLO object detection model

layers = network.getLayerNames()

print(layers)

layers = [layers[i[0] - 1] for i in network.getUnconnectedOutLayers()]

print(layers)

def single_image_detection(image_path):

    image = cv2.imread(image_path)

    img_height, img_width = image.shape[0:2]

    image_blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    network.setInput(image_blob)
    start = time.time()
    layers_Output = network.forward(layers)  
    print("Time taken for bounding boc detection and their respective probabilities: ",time.time()-start)

    bounding_boxes, confidences, class_ids = [],[],[]

    for output in layers_Output:
        
        # loop over each of the detections of the yolo3
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > input_confidence:
                print(confidence,'confidence')
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                bounding_boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(bounding_boxes, confidences, input_confidence,nms_threshold)

    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (bounding_boxes[i][0], bounding_boxes[i][1])
            (w, h) = (bounding_boxes[i][2], bounding_boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in colors_for_label[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(coco_labels[class_ids[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


for image_name in images_list:
    image_path = os.path.join(input_images_path,image_name)
    image = single_image_detection(image_path)    
    des_image_path = os.path.join(output_images_path,os.path.basename(image_path))
    print(des_image_path)
    cv2.imwrite(des_image_path, image)







