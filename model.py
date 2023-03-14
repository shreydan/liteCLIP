import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from timm import create_model
from config import Config    


class ImageEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.backbone = create_model(self.model_name, 
                                     pretrained=False, 
                                     num_classes=1,
                                    )
        self.embed_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
    def forward(self,x):
        return self.backbone(x)
    


class TextEncoder(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        model_config = AutoConfig.from_pretrained(model_path)
        self.backbone = AutoModel.from_config(model_config)
        self.embed_dim = self.backbone.config.hidden_size
        
    def mean_pooler(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    
    def forward(self, inputs):
        outputs = self.backbone(**inputs)
        pooled_output = self.mean_pooler(outputs['last_hidden_state'],inputs['attention_mask'])
        return pooled_output
    


class ProjectionHead(nn.Module):
    def __init__(self, embed_dim, Config):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim = Config.proj_dim
        self.dropout = Config.dropout
        
        self.proj = nn.Linear(self.embed_dim, self.proj_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(self.dropout)
        self.ln = nn.LayerNorm(self.proj_dim)
        
    def forward(self, x):
        x = self.proj(x)
        out = self.act(x)
        out = self.drop(out)
        x = x + out
        x = self.ln(x)
        return x
    

class CLIP(nn.Module):
    def __init__(self, Config):
        super().__init__()
        
        self.image_encoder = ImageEncoder(Config.image_encoder)
        self.text_encoder = TextEncoder(Config.text_encoder)
        
        self.im_embed_dim = self.image_encoder.embed_dim
        self.txt_embed_dim = self.text_encoder.embed_dim 
        
        self.img_projection = ProjectionHead(self.im_embed_dim,Config)
        self.txt_projection = ProjectionHead(self.txt_embed_dim,Config)
        
        
    def forward(self,inputs):
        image, text = inputs
        
        image_embeddings = self.image_encoder(image)
        image_embeddings = self.img_projection(image_embeddings)
        
        text_embeddings = self.text_encoder(text)
        text_embeddings = self.txt_projection(text_embeddings)
        
        # logits will be in the shape batch_size X batch_size
        logits = (text_embeddings @ image_embeddings.T)
        
        return logits
    


if __name__ == '__main__':

    model = CLIP(Config)
    model.load_state_dict(torch.load(Config.state_dict_path))
    model.eval()
    print(model)
