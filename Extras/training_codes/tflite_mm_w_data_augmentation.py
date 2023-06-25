#Useful imports
import tensorflow as tf
from tflite_model_maker.config import QuantizationConfig
#from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_model_maker.object_detector import DataLoader

#Import the same libs that TFLiteModelMaker interally uses
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import train
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import train_lib



#Setup variables
batch_size = 32 #or whatever batch size you want
epochs = 20
checkpoint_dir = '/content/drive/MyDrive/UM/TIF/TIF_III/workspace/checkpoints/model.ckpt' #whatever your checkpoint directory is


#Create whichever object detector's spec you want
spec = model_spec.get('efficientdet_lite1')



#Load you datasets

train_data = object_detector.DataLoader.from_pascal_voc(
    '/content/drive/MyDrive/UM/TIF/TIF_III/workspace/data_set/train(completo)',
    '/content/drive/MyDrive/UM/TIF/TIF_III/workspace/data_set/train(completo)',
    ['A', 'B', 'C', 'D', 'E', 'ESPACIO', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Ñ', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    # ['A', 'B']
)

val_data = object_detector.DataLoader.from_pascal_voc(
    '/content/drive/MyDrive/UM/TIF/TIF_III/workspace/data_set/test(completo)',
    '/content/drive/MyDrive/UM/TIF/TIF_III/workspace/data_set/test(completo)',
    ['A', 'B', 'C', 'D', 'E', 'ESPACIO', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Ñ', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    # ['A', 'B']
)

#Create the object detector 
detector = object_detector.create(
    train_data, 
    model_spec=spec, 
    batch_size=batch_size, 
    train_whole_model=True, 
    validation_data=val_data,
    epochs = epochs,
    do_train = False
)


# """
# From here on we use internal/"private" functions of the API,
# you can tell because the methods' names begin with an underscore
# """

#Convert the datasets for training
train_data, steps_per_epoch, _ = detector._get_dataset_and_steps(train_data, batch_size, is_training = True)
validation_ds, validation_steps, val_json_file = detector._get_dataset_and_steps(val_data, batch_size, is_training = False)

# DATA AUGMENTATION
from keras import layers

IMG_SIZE = 384

data_augmentation = tf.keras.Sequential([
    layers.Resizing(IMG_SIZE, IMG_SIZE),
    layers.Rescaling(1./255),
    # layers.RandomFlip("horizontal"),
    # layers.RandomRotation(0.2),
])

train_data = train_data.map(lambda x, y: (data_augmentation(x, training=True), y))
val_data = validation_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

#Get the internal keras model    
model = detector.create_model()

#Copy what the API internally does as setup
config = spec.config
config.update(
    dict(
        steps_per_epoch=steps_per_epoch,
        eval_samples=batch_size * validation_steps,
        val_json_file=val_json_file,
        batch_size=batch_size
    )
)
train.setup_model(model, config) #This is the model.compile call basically
model.summary()



"""
Here we restore the weights
"""
try:
    model.load_weights(checkpoint_dir)
    print("FOUND EXISTING CHECKPOINT!")

except Exception as e:
    print("Checkpoint not found: ", e)


## ACA DEBERIA AGREGAR WANDB
import wandb
from wandb.keras import WandbCallback

wandb.login(key='51b44a141a1503d6973f7aaae08cf99e69966cf7')

# #Monitoreo con WandB
n_prueba = 3
wandb.init(project='TLSA_Project', entity='juanjricci', name=str(f'Run_{n_prueba}'))
wandb.config.architecture = 'EfficientNet'
wandb.config.backbone = 'efficientdet_lite1'


#Retrieve the needed default callbacks
# all_callbacks = train_lib.get_callbacks(config.as_dict(), validation_ds)

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
import math
import keras
import keras.backend as K
from keras.callbacks import Callback
import numpy as np
import tensorflow as tf

#https://machinelearningmastery.com/check-point-deep-learning-models-keras/
#https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
#--------------- CHECKPOINT VALIDATION LOSS ---------------
def set_val_loss_checkpoint(main_dir):
    #filepath = (main_dir + "/my_model")#.replace('#', n_split)
    checkpoint = ModelCheckpoint(main_dir, save_format='tf', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    
    # filepath = (main_dir + "/min_val_loss.hdf5")#.replace('#', n_split)
    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
    
    
    return checkpoint


#--------------- CALLBACK EARLY STOPPING ---------------
def set_early_stopping (patience = 20, monitor='val_loss'):
    early_stopping = EarlyStopping(monitor=monitor,
                                   min_delta=0,
                                   patience=patience,
                                   verbose=0, mode='auto')
    return early_stopping


check_val_loss = set_val_loss_checkpoint(checkpoint_dir)
# early_stopping = set_early_stopping(config['ckp_patience'], config['early_ckp_monitor'])
callbacks_list = [check_val_loss, WandbCallback(save_model=False, save_graph=False)]

"""
Optional step.
Add callbacks that get executed at the end of every N 
epochs: in this case I want to log the training results to tensorboard.
"""
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)
#all_callbacks.append(tensorboard_callback)


"""
Train the model 
"""
model.fit(
    train_data,
    epochs=epochs,
    # initial_epoch=completed_epochs, 
    steps_per_epoch=steps_per_epoch,
    validation_data=val_data,
    validation_steps=validation_steps,
    # validation_data=val_data,
    callbacks=callbacks_list #all_callbacks #This is for saving checkpoints at the end of every epoch + running the above added callbacks
)

print(model.summary())


"""
Save/export the trained model
Tip: for integer quantization you simply have to NOT SPECIFY 
the quantization_config parameter of the detector.export method.
In this case it would be: 
detector.export(export_dir = export_dir, tflite_filename='model.tflite')
"""
exported_filename = f'lsa_v{n_prueba}.tflite'
export_dir = "/content/drive/MyDrive/UM/TIF/TIF_III/workspace/models" #save the tflite wherever you want
quant_config = QuantizationConfig.for_float16() #or whatever quantization you want
# detector.model = model #inject our trained model into the object detector
model.export(export_dir = export_dir, tflite_filename=exported_filename, quantization_config = quant_config)
print(f"--------------------------------------")
print(f"EXPORTED MODEL AS: {exported_filename}")
print(f"--------------------------------------")
print(f"Evaluating...")
print(detector.evaluate_tflite(f'/content/drive/MyDrive/UM/TIF/TIF_III/workspace/models/{exported_filename}', val_data))