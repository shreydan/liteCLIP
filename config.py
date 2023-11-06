from dataclasses import dataclass

@dataclass
class Config:
    text_encoder = './model'
    image_encoder = 'convnext_tiny'
    
    # CLIP CONFIG
    proj_dim = 256
    dropout = 0.1
    max_length = 128

    state_dict_path = './model/liteclip2.pt'
