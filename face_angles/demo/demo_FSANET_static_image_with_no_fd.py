import os, glob
import cv2
import sys
sys.path.append('..')
import numpy as np
from math import cos, sin
# from moviepy.editor import *
from lib.FSANET_model import *
import numpy as np
from keras.layers import Average
# from moviepy.editor import *
# from mtcnn.mtcnn import MTCNN

#The blue line indicates the direction the subject is facing; the green line for
# the downward direction while the red one for the side

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 50):
    print(yaw,roll,pitch)

    cv2.putText(img, "pitch: " + str(np.round(pitch,2)), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(img, "yaw: " + str(np.round(yaw,2)), (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(img, "roll: " + str(np.round(roll,2)), (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img
    
def draw_results_ssd(detected,input_img,faces,ad,img_size,img_w,img_h,model,time_detection,time_network,time_plot):
 
    # cv2.imshow('check_image',cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size)))
    faces[0,:,:,:] = cv2.resize(input_img[:,:], (img_size, img_size))
    faces[0,:,:,:] = cv2.normalize(faces[0,:,:,:], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)        
    # cv2.imshow("face", faces[i,:,:,:])
    face = np.expand_dims(faces[0,:,:,:], axis=0)
    
    p_result = model.predict(face)
    
    face = face.squeeze()
    img = draw_axis(input_img[:, :, :], p_result[0][0], p_result[0][1], p_result[0][2])
    
    input_img[:,:, :] = img
                
    cv2.imshow("result", input_img)
    
    return input_img #,time_network,time_plot

def main():
    
    main_folder_path = '/home/user/Pictures/FSA-Net/jiffy_all_relax/low_jiffy_occlusion_Sheet1_image2'
    des_folder_path = main_folder_path + '_no_fd'
    main_folder_images_path = glob.glob(main_folder_path+'/*')
    try:
        os.makedirs(des_folder_path)
    except OSError:
        pass
    # face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
    # detector = MTCNN()

    # load model and weights
    img_size = 64
    stage_num = [3,3,3]
    lambda_local = 1
    lambda_d = 1
    img_idx = 0
    detected = '' #make this not local variable
    time_detection = 0
    time_network = 0
    time_plot = 0
    skip_frame = 1 # every 5 frame do 1 detection and network forward propagation
    ad = 0.6

    #Parameters
    num_capsule = 3
    dim_capsule = 16
    routings = 2
    stage_num = [3,3,3]
    lambda_d = 1
    num_classes = 3
    image_size = 64
    num_primcaps = 7*3
    m_dim = 5
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model1 = FSA_net_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
    model2 = FSA_net_Var_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
    
    num_primcaps = 8*8*3
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model3 = FSA_net_noS_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
    
    print('Loading models ...')

    weight_file1 = '../pre-trained/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
    model1.load_weights(weight_file1)
    print('Finished loading model 1.')
    
    weight_file2 = '../pre-trained/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
    model2.load_weights(weight_file2)
    print('Finished loading model 2.')

    weight_file3 = '../pre-trained/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
    model3.load_weights(weight_file3)
    print('Finished loading model 3.')

    inputs = Input(shape=(64,64,3))
    x1 = model1(inputs) #1x1
    x2 = model2(inputs) #var
    x3 = model3(inputs) #w/o
    avg_model = Average()([x1,x2,x3])
    model = Model(inputs=inputs, outputs=avg_model)
    
    print('Start detecting pose ...')
    detected_pre = np.empty((1,1,1))
    
    for img_path in main_folder_images_path:
        try:
            
            input_img = cv2.imread(img_path)
            image_name = img_path.split('/')[-1]
            img_idx = img_idx + 1
            img_h, img_w, _ = np.shape(input_img)
            
            if img_idx==1 or img_idx%skip_frame == 0:
                time_detection = 0
                time_network = 0
                time_plot = 0

                detected = None
 
                faces = np.empty((1, img_size, img_size, 3))
                input_img = draw_results_ssd(detected,input_img,faces,ad,img_size,img_w,img_h,model,time_detection,time_network,time_plot)
                cv2.imwrite(des_folder_path+'/'+str(image_name)+'_ssd.png',input_img)
            else:
                input_img = draw_results_ssd(detected,input_img,faces,ad,img_size,img_w,img_h,model,time_detection,time_network,time_plot)


            if detected.shape[2] > detected_pre.shape[2] or img_idx%(skip_frame*3) == 0:
                detected_pre = detected

            key = cv2.waitKey(1)
        except Exception as E:
            print(E)

        
if __name__ == '__main__':
    main()
