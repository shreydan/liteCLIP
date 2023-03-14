from dataclasses import dataclass

@dataclass
class Config:
    text_encoder = 'model'
    image_encoder = 'resnet50d'
    
    # CLIP CONFIG
    proj_dim = 256
    dropout = 0.1
    max_length = 128

    state_dict_path = 'model/clip_model.pt'
