import tensorflow as tf
from stable_diffusion_tf.clip_encoder import CLIPTextTransformer
from stable_diffusion_tf.diffusion_model import UNetModel
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

for name in logging.Logger.manager.loggerDict.keys():
    logging.getLogger(name).setLevel(logging.CRITICAL)



directory = "Tflite models"
parent_dir=os.getcwd()
print(f"parent_directory {parent_dir}")
path=os.path.join(parent_dir, directory)
os.mkdir(path)
path_to_model=os.path.join(parent_dir, "H5 Models")
print(f"Tensorflow lite Models will be saved at {path}")

class EncoderTflite:
    def __init__(path_to_model,folder):
        self.path_to_model=path_to_model,
        self.folder=folder

    def totflite(path_to_model,folder):
        # custom_objects={'CLIPTextTransformer':CLIPTextTransformer}
        encoder = tf.keras.models.load_model(path_to_model)
        converter = tf.lite.TFLiteConverter.from_keras_model(encoder)
        converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        open(folder+"/converted_encoder.tflite", "wb").write(tflite_model)

class TextEncoderTflite:
    def __init__(path_to_model,folder):
        self.path_to_model=path_to_model,
        print(self.path_to_model)
        self.folder=folder

    def totflite(path_to_model,folder):
        custom_objects={'CLIPTextTransformer':CLIPTextTransformer}
        text_encoder = tf.keras.models.load_model(path_to_model,custom_objects = custom_objects)
        converter = tf.lite.TFLiteConverter.from_keras_model(text_encoder)
        converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        open(folder+"/converted_text_encoder.tflite", "wb").write(tflite_model)

class DecoderTflite:
    def __init__(path_to_model,folder):
        self.path_to_model=path_to_model,
        self.folder=folder

    def totflite(path_to_model,folder):
        decoder = tf.keras.models.load_model(path_to_model)
        converter = tf.lite.TFLiteConverter.from_keras_model(decoder)
        converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        open(folder+"/converted_decoder.tflite", "wb").write(tflite_model)
class DiffusionTflite:
    def __init__(path_to_model,folder):
        self.path_to_model=path_to_model,
        self.folder=folder

    def totflite(path_to_model,folder):
        custom_objects={'UNetModel':UNetModel}
        diffusion = tf.keras.models.load_model(path_to_model,custom_objects = custom_objects)
        converter = tf.lite.TFLiteConverter.from_keras_model(diffusion)
        converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        open(folder+"/converted_diffusion_model.tflite", "wb").write(tflite_model)

EncoderTflite.totflite(path_to_model+"/encoder",path)
DecoderTflite.totflite(path_to_model+"/decoder",path)
TextEncoderTflite.totflite(path_to_model+"/text_encoder.hdf5",path)
DiffusionTflite.totflite(path_to_model+"/diffusion_model.hdf5",path)