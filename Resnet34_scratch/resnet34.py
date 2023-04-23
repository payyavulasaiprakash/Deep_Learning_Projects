from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Activation, Input, AveragePooling2D, Flatten, ZeroPadding2D
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Model,load_model
import tensorflow

class Build_Fit_Resnet34():

    def __init__(self,shape,classes,x_training_data,y_training_data,x_validation_data,y_validation_data,epochs,batch,learning_rate,save_model_to):
        self.shape = shape
        self.classes = classes
        self.x_training_data = x_training_data
        self.x_validation_data = x_validation_data
        self.y_training_data = y_training_data
        self.y_validation_data = y_validation_data
        self.epochs = epochs
        self.batch = batch
        self.learning_rate = learning_rate
        self.save_model_to = save_model_to

    def conv_block(self,x,filter):
        x_skip = x

        #Layer 1
        x = Conv2D(filter, (3,3), padding='same', strides=(2,2))(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        #Layer 2
        x = Conv2D(filter, (3,3), padding='same')(x)
        x = BatchNormalization(axis=3)(x)

        x_skip = Conv2D(filter,(1,1),strides=(2,2))(x_skip)

        #Adding residue
        x = layers.Add()([x,x_skip])
        x = Activation('relu')(x)
        return x
    
    def identity_block(self,x,filter):
        x_skip = x

        #layer1
        x = Conv2D(filter,(3,3),padding='same')(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        #layer2
        x = Conv2D(filter,(3,3),padding='same')(x)
        x = BatchNormalization(axis=3)(x)

        #Adding residue
        x = layers.Add()([x,x_skip])
        x = Activation('relu')(x)
        return x

    def build_resnet_34_architechture(self):

        input = Input(self.shape)
        x = ZeroPadding2D((3, 3))(input)

        x = Conv2D(64,(3,3),strides=(2, 2),padding='same')(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        x = MaxPooling2D(pool_size=3,strides=(2, 2),padding='same')(x)
        layers_in_block = [3,4,6,3]
        filter_size = 64

        for index, layers in enumerate(layers_in_block):
            if index == 0:
                for layer_number in range(layers):
                    x = self.identity_block(x,filter_size)
            else:
                filter_size *= 2
                for layer_number in range(layers):
                    # print(layer_number,filter_size)
                    if layer_number == 0:
                        x = self.conv_block(x,filter_size)
                    else:
                        x = self.identity_block(x,filter_size)

        x = AveragePooling2D((2,2),padding = 'same')(x)
        x = Flatten()(x)
        x = Dense(512)(x)
        x = Activation('relu')(x)
        x = Dense(self.classes,activation='softmax')(x)
        model=Model(inputs = input, outputs = x , name = 'resnet34architechture')
        return model
        

    def fit_resnet_34_architechture(self):
        model=self.build_resnet_34_architechture()
        print(model.summary())
        adam=optimizers.Adam(learning_rate=self.learning_rate)
        if self.classes==2:
            model.compile(optimizer=adam,loss='binary_crossentropy',metrics='accuracy')
        elif self.classes>=3:
            model.compile(optimizer=adam,loss='categorical_crossentropy',metrics='accuracy')
        else:
            print("Please give correct classes as input")
        model_checkpoint_callback=tensorflow.keras.callbacks.ModelCheckpoint(
        filepath =self.save_model_to + '/' + 'full_model_epoch_{epoch:02d}-val_acc_{val_accuracy:.4f}.h5',
        monitor = "val_loss",
        verbose = 0,
        save_best_only = False,
        save_weights_only = False)
        callbacks=[model_checkpoint_callback]
        history = model.fit(self.x_training_data,self.y_training_data,epochs=self.epochs,batch_size=self.batch,validation_data=(self.x_validation_data,self.y_validation_data), verbose = 1,callbacks=callbacks)
        return history
            
        







