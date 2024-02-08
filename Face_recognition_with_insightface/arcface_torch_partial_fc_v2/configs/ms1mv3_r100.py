from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()

#config.margin_list = [0.9, 0.4, 0.15] #for combined loss
config.margin_list = [64,0.7] #for arc face loss 64 and 0.5 default values
config.network = "r100"
config.resume = False
config.output = '' #to save models, specify folder path 
config.base_model_path = '' #to train with base model, specify file path
config.check_point_path = config.base_model_path
config.embedding_size = 512 #embedding size, number
config.sample_rate = 0.3 #negative sample rate to consider
config.fp16 = True
config.momentum = 0.9 #optimizer momentum
config.weight_decay = 5e-3 #optimizer weight_decay
config.batch_size = 64
config.lr = 0.001 #optimizer learning rate
config.verbose = 2000
config.dali = False
config.fine_tune = False #if fine tune with base model then True else False
config.finetune_with_checkpoint = True #if fine tune with base model check point then True else False

config.rec = "" #dataset path - subfolders defines each unique user, each sub folder consists of unique user images 
config.num_classes = 125553 #number of unique users or number of subfolders 
config.num_image = 255527 #number of images used for training
config.optimizer = 'adamw'
config.num_epoch = 100
config.warmup_epoch = 0
# config.val_rec = '/home/vishwam/mountpoint/saiprakash/insight_face/insightface/recognition/arcface_torch/our_datasets/unifeid_v2/test_cropped'
# config.val_freq = 1
config.val_targets = ['test_cropped'] #specify folder name, not required to have any data in that folder as it does take validation data for testing
config.save_all_states = True #to save all models
config.description = 'trained with base model --> models_unified_v1_v2_nnadhar_test_cropped_copied_0.001_0.3_64_jan26' #to write model training details etc..
