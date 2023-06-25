import numpy as np
import os

from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

from tflite_support import metadata

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

train_data = object_detector.DataLoader.from_pascal_voc(
    '/content/drive/MyDrive/UM/TIF/TIF_III/workspace/data_set/trian',
    '/content/drive/MyDrive/UM/TIF/TIF_III/workspace/data_set/trian',
    # ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Ñ', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    ['A', 'B']
)

val_data = object_detector.DataLoader.from_pascal_voc(
    '/content/drive/MyDrive/UM/TIF/TIF_III/workspace/data_set/test',
    '/content/drive/MyDrive/UM/TIF/TIF_III/workspace/data_set/test',
    # ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Ñ', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    ['A', 'B']
)

print(train_data)

spec = model_spec.get('efficientdet_lite1')

model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=30, validation_data=val_data)

model.evaluate(val_data)

model.export(export_dir='.', tflite_filename='lsa_model.tflite')