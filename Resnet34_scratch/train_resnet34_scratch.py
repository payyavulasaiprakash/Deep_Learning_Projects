import os, cv2, numpy as np
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from resnet34 import Build_Fit_Resnet34
import matplotlib.pyplot as plt

dataset_path = 'cats_dogs_data_set_sample'
learning_rate = 0.001
batch_size = 32
epochs = 10
image_paths = list(paths.list_images(dataset_path))
shape = (100,100,3)  #can change as per requirement. I kept this to decrease the computation, as i am having less resources
classes = 2
save_model_to = 'output_models_scratch_224'
plot_image_name = "scratch_resnet34.jpg"
os.makedirs(save_model_to,exist_ok=True)

data = []
labels = []

for image_path in image_paths:
    label = image_path.split(os.path.sep)[-2]
    img = cv2.imread(image_path)
    img = cv2.resize(img,(100,100)) #can change as per requirement
    data.append(img)
    labels.append(label)

std_data = np.array(data, dtype='float') / 255.0
le = LabelEncoder()

X_train, X_test, y_train, y_test = train_test_split(std_data, labels, random_state=42, test_size=0.30,shuffle=True)

y_train_labels = le.fit_transform(y_train)
y_train_labels = to_categorical(y_train_labels, 2)

print(std_data.shape,y_train_labels.shape)

y_test_labels = le.transform(y_test)
y_test_labels = to_categorical(y_test_labels, 2)

resnet_network_class = Build_Fit_Resnet34(shape=shape,classes=classes,x_training_data=X_train,y_training_data=y_train_labels,x_validation_data=X_test,y_validation_data=y_test_labels,epochs=epochs,batch=batch_size,learning_rate=learning_rate,save_model_to=save_model_to)
history = resnet_network_class.fit_resnet_34_architechture()

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), history.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="val_accuracy")
plt.title("Loss and Accuracy on training and testing Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(plot_image_name)

print('Training_accuracy: ',history.history["accuracy"])
print('Testing_accuracy: ',history.history["val_accuracy"])