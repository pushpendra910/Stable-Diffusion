from stable_diffusion_tf.stable_diffusion import StableDiffusion
from IPython.display import display
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

for name in logging.Logger.manager.loggerDict.keys():
    logging.getLogger(name).setLevel(logging.CRITICAL)



directory = "H5 Models"
parent_dir=os.getcwd()
print(f"parent_directory {parent_dir}")
save_folder_path=os.path.join(parent_dir, directory)
print(f"Models will be saved at {os.path.join(parent_dir, directory)}")

keras.mixed_precision.set_global_policy("mixed_float16")
class Instanciate_Model:
    def __init__(self,img_height=512,img_weight=512,jit_compile=False,download_weights=True):
        self.img_height=img_height,
        self.img_weight=weight,
        self.jit_compile=jit_compile,
        self.download_weights=download_weights

    def LoadModel(img_height=512,img_weight=512,jit_compile=False,download_weights=True):
        return StableDiffusion(img_height,img_weight,jit_compile,download_weights)
    
model=Instanciate_Model.LoadModel(512,512,False,True)
img=model.generate("DSLR photograph of an astronaut riding a horse",
    num_steps=50,
    unconditional_guidance_scale=7.5,
    temperature=1,
    batch_size=1)

print(".....................Saving text_encoder")
model.text_encoder.save(save_folder_path+"/text_encoder.hdf5")
print(f".....................Text Encoder Saved at {save_folder_path}")
print(".....................Saving Encoder")
model.encoder.save(save_folder_path+"/encoder")
print(f".....................Encoder Saved at {save_folder_path}")

print(".....................Saving Decoder")
model.decoder.save(save_folder_path+"/decoder")
print(f".....................Decoder Saved at {save_folder_path}")

print(".....................Saving Diffusion Model")
model.diffusion_model.save(save_folder_path+"/diffusion_model.hdf5") # SAved Successfully
print(f".....................Diffusion model Saved at {save_folder_path}")
