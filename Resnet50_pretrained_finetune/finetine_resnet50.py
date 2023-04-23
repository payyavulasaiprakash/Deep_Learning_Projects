from tensorflow.keras.applications import ResNet50
import os, cv2, numpy as np
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D, Dropout
from tensorflow.keras import optimizers
import tensorflow
import matplotlib.pyplot as plt

def resnet50_finetune(shape,classes):
    pretrained_resnet_architechture = ResNet50(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=shape)
    head_model = pretrained_resnet_architechture.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(256, activation="relu")(head_model)
    head_model = Dropout(0.6)(head_model)
    head_model = Dense(classes, activation="softmax")(head_model)
    final_model = Model(inputs = pretrained_resnet_architechture.input, outputs = head_model)
    #transfer learning
    for layer in pretrained_resnet_architechture.layers:
	    layer.trainable = False
    return final_model


train_dataset_path=os.path.join('cats_dogs_data_set_sample','train')
test_dataset_path=os.path.join('cats_dogs_data_set_sample','test')

learning_rate=0.001
batch_size=32
epochs=10

train_image_paths = list(paths.list_images(train_dataset_path))
test_image_paths = list(paths.list_images(test_dataset_path))

total_training_images=len(train_image_paths)
total_testing_images=len(test_image_paths)

train_Augmentor = ImageDataGenerator(
	rotation_range=25,
	zoom_range=0.1,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.2,
	horizontal_flip=True,
	fill_mode="nearest")

test_Augmentor = ImageDataGenerator()

mean = np.array([123.68, 116.779, 103.939], dtype="float32")

train_Augmentor.mean = mean
test_Augmentor.mean = mean

# initialize the validation generator
trainGen = train_Augmentor.flow_from_directory(
	train_dataset_path,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=True,
	batch_size=batch_size)

# initialize the test generator
testGen = test_Augmentor.flow_from_directory(
	test_dataset_path,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=batch_size)


shape = (224,224,3) 
classes = 2
save_model_to = 'output_models_finetune'
os.makedirs(save_model_to,exist_ok=True)

model = resnet50_finetune(shape,classes)

print(model.summary())

adam=optimizers.Adam(learning_rate=learning_rate)

if classes==2:
    model.compile(optimizer=adam,loss='binary_crossentropy',metrics='accuracy')
elif classes>=3:
    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics='accuracy')
else:
    print("Please give correct classes as input")

model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
filepath = save_model_to + '/' + 'full_model_epoch_{epoch:02d}-val_acc_{val_accuracy:.4f}.h5',
monitor = "val_accuracy",
verbose = 0,
save_best_only = False,
save_weights_only = False)
callbacks=[model_checkpoint_callback]
history = model.fit(trainGen,
	steps_per_epoch=total_training_images//batch_size,
	validation_data=testGen,
	validation_steps=total_testing_images//batch_size ,epochs=epochs,callbacks=callbacks)


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
plt.savefig("finetune_resnet50.jpg")

print('Training_accuracy: ',history.history["accuracy"])
print('Testing_accuracy: ',history.history["val_accuracy"])
