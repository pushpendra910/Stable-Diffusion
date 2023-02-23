import tensorflow as tf
from stable_diffusion_tf.clip_encoder import CLIPTextTransformer
from stable_diffusion_tf.diffusion_model import UNetModel
import os
import tf2onnx
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

for name in logging.Logger.manager.loggerDict.keys():
    logging.getLogger(name).setLevel(logging.CRITICAL)





class EncoderTflite:
    def __init__(path_to_model,folder):
        self.path_to_model=path_to_model,
        self.folder=folder

    def toonnx(path_to_model,folder):
        # custom_objects={'CLIPTextTransformer':CLIPTextTransformer}
        encoder = tf.keras.models.load_model(path_to_model)
        # input_names = ['input']
        # output_names = ['output']
        onnx_model = tf2onnx.convert.from_keras(encoder)
        onnx_string = onnx_model.SerializeToString()
        with open('encoder.onnx', 'wb') as f:
            f.write(onnx_string)

class TextEncoderTflite:
    def __init__(path_to_model,folder):
        self.path_to_model=path_to_model,
        print(self.path_to_model)
        self.folder=folder

    def toonnx(path_to_model,folder):
        custom_objects={'CLIPTextTransformer':CLIPTextTransformer}
        text_encoder = tf.keras.models.load_model(path_to_model,custom_objects = custom_objects)
        onnx_model = tf2onnx.convert.from_keras(text_encoder)
        onnx_string = onnx_model.SerializeToString()
        with open('text_encoder.onnx', 'wb') as f:
            f.write(onnx_string)

class DecoderTflite:
    def __init__(path_to_model,folder):
        self.path_to_model=path_to_model,
        self.folder=folder

    def toonnx(path_to_model,folder):
        decoder = tf.keras.models.load_model(path_to_model)
        onnx_model = tf2onnx.convert.from_keras(decoder)
        onnx_string = onnx_model.SerializeToString()
        with open('decoder.onnx', 'wb') as f:
            f.write(onnx_string)
class DiffusionTflite:
    def __init__(path_to_model,folder):
        self.path_to_model=path_to_model,
        self.folder=folder

    def toonnx(path_to_model,folder):
        custom_objects={'UNetModel':UNetModel}
        diffusion = tf.keras.models.load_model(path_to_model,custom_objects = custom_objects)
        onnx_model = tf2onnx.convert.from_keras(diffusion)
        onnx_string = onnx_model.SerializeToString()
        with open('diffusion_model.onnx', 'wb') as f:
            f.write(onnx_string)

            
directory = "Onnx models"
parent_dir=os.getcwd()
print(f"parent_directory {parent_dir}")
path=os.path.join(parent_dir, directory)
os.mkdir(path)
path_to_model=os.path.join(parent_dir, "H5 Models")
print(f"Tensorflow lite Models will be saved at {path}")
EncoderTflite.toonnx(path_to_model+"/encoder",path)
DecoderTflite.toonnx(path_to_model+"/decoder",path)
TextEncoderTflite.toonnx(path_to_model+"/text_encoder.hdf5",path)
DiffusionTflite.toonnx(path_to_model+"/diffusion_model.hdf5",path)