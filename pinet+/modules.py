from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import os
import math



class MHA_self(nn.Module):
    def __init__(self,emb,head,ffn_emb,ln=1,drop=0.1):
        super(MHA_self, self).__init__()
        self.ln=ln
        self.multihead_attn = torch.nn.MultiheadAttention(emb, head)
        self.dropout = nn.Dropout(p=drop)
        if ln:
            self.attn_layer_norm=torch.nn.LayerNorm(emb)
            self.final_layer_norm=torch.nn.LayerNorm(emb)
        self.activation_fn = nn.GELU()
        self.fc1=nn.Linear(emb,ffn_emb)
        self.fc2=nn.Linear(ffn_emb,emb)



    def forward(self, x,y,my=None,mxy=None):
        residual = x
        x, attn = self.multihead_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=my,
            attn_mask=mxy,
        )
        x = self.dropout(x)
        x = residual + x
        if self.ln:
            x = self.attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x
        if self.ln:
            x = self.final_layer_norm(x)
        return x, attn
    
    
class MHA_cross(nn.Module):
    def __init__(self,emb,head,ffn_emb,ln=1,drop=0.1):
        super(MHA_cross, self).__init__()
        self.ln=ln
        self.multihead_attn0 = torch.nn.MultiheadAttention(emb, head)
        self.multihead_attn1 = torch.nn.MultiheadAttention(emb, head)
        self.dropout = nn.Dropout(p=drop)
        if ln:
            self.attn_layer_norm0=torch.nn.LayerNorm(emb)
            self.attn_layer_norm1=torch.nn.LayerNorm(emb)
            self.final_layer_norm=torch.nn.LayerNorm(emb)
        self.activation_fn = nn.GELU()
        self.fc1=nn.Linear(emb,ffn_emb)
        self.fc2=nn.Linear(ffn_emb,emb)



    def forward(self, x,y,mx,my,mxx=None,mxy=None):
        residual = x
        x, attn = self.multihead_attn0(
            query=x,
            key=x,
            value=x,
            key_padding_mask=mx,
            attn_mask=mxx,
        )
        x = self.dropout(x)
        x = residual + x
        if self.ln:
            x = self.attn_layer_norm0(x)
        
        residual = x
        x, attn = self.multihead_attn1(
            query=x,
            key=y,
            value=y,
            key_padding_mask=my,
            attn_mask=mxy,
        )
        x = self.dropout(x)
        x = residual + x
        if self.ln:
            x = self.attn_layer_norm1(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x
        if self.ln:
            x = self.final_layer_norm(x)
        return x, attn
    
    
class MHA_cross2(nn.Module):
    def __init__(self,emb,head,ffn_emb,ln=1,drop=0.1):
        super(MHA_cross2, self).__init__()
        self.ln=ln
        self.multihead_attn0 = torch.nn.MultiheadAttention(emb, head)
        self.multihead_attn1 = torch.nn.MultiheadAttention(emb, head)
        self.dropout = nn.Dropout(p=drop)
        if ln:
            self.attn_layer_norm0=torch.nn.LayerNorm(emb)
            self.attn_layer_norm1=torch.nn.LayerNorm(emb)
            self.final_layer_norm=torch.nn.LayerNorm(emb)
        self.activation_fn = nn.GELU()
        self.fc1=nn.Linear(emb,ffn_emb)
        self.fc2=nn.Linear(ffn_emb,emb)



    def forward(self, x,y,mx=None,my=None,mxx=None,mxy=None):
        residual = x
        x, attn = self.multihead_attn0(
            query=x,
            key=x,
            value=x,
            key_padding_mask=mx,
            attn_mask=mxx,
        )
        x = self.dropout(x)
        x = residual + x
        if self.ln:
            x = self.attn_layer_norm0(x)
        
        residual = x
        x, attn = self.multihead_attn1(
            query=x,
            key=y,
            value=y,
            key_padding_mask=my,
            attn_mask=mxy,
        )
        x = self.dropout(x)
        x = residual + x
        if self.ln:
            x = self.attn_layer_norm1(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x
        if self.ln:
            x = self.final_layer_norm(x)
        return x, attn

class MHA_self_pos(nn.Module):
    def __init__(self,emb,head,ffn_emb,ln=1,drop=0.1):
        super(MHA_self_pos, self).__init__()
        self.ln=ln
        self.pos=nn.Linear(3,emb)
        self.multihead_attn = torch.nn.MultiheadAttention(emb, head)
        self.dropout = nn.Dropout(p=drop)
        if ln:
            self.attn_layer_norm=torch.nn.LayerNorm(emb)
            self.final_layer_norm=torch.nn.LayerNorm(emb)
        self.activation_fn = nn.GELU()
        self.fc1=nn.Linear(emb,ffn_emb)
        self.fc2=nn.Linear(ffn_emb,emb)



    def forward(self, px,x,my=None,mxy=None):
        x=(self.pos(px).transpose(1,2)+x).permute(2,0,1).contiguous()
        residual = x
        x, attn = self.multihead_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=my,
            attn_mask=mxy,
        )
        x = self.dropout(x)
        x = residual + x
        if self.ln:
            x = self.attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x
        if self.ln:
            x = self.final_layer_norm(x)
        return x, attn
    
class MHA_cross_pos(nn.Module):
    def __init__(self,emb,head,ffn_emb,ln=1,drop=0.1):
        super(MHA_cross_pos, self).__init__()
        self.ln=ln
#         self.pos=nn.Linear(3,emb)
        self.multihead_attn0 = torch.nn.MultiheadAttention(emb, head)
        self.multihead_attn1 = torch.nn.MultiheadAttention(emb, head)
        self.dropout = nn.Dropout(p=drop)
        if ln:
            self.attn_layer_norm0=torch.nn.LayerNorm(emb)
            self.attn_layer_norm1=torch.nn.LayerNorm(emb)
            self.final_layer_norm=torch.nn.LayerNorm(emb)
        self.activation_fn = nn.GELU()
        self.fc1=nn.Linear(emb*2,ffn_emb)
        self.fc2=nn.Linear(ffn_emb,emb)



    def forward(self, px,x,py,y,mx=None,my=None,mxx=None,mxy=None):
#         x=(self.pos(px).transpose(1,2)+x).permute(2,0,1).contiguous()
#         y=(self.pos(py).transpose(1,2)+y).permute(2,0,1).contiguous()

        residual = x
        x, attn = self.multihead_attn0(
            query=x,
            key=x,
            value=x,
            key_padding_mask=mx,
            attn_mask=mxx,
        )
        x = self.dropout(x)
        x = residual + x
        if self.ln:
            x = self.attn_layer_norm0(x)
        
        residual = x
        x, attn = self.multihead_attn1(
            query=x,
            key=y,
            value=y,
            key_padding_mask=my,
            attn_mask=mxy,
        )
#         x = self.dropout(x)
        if self.ln:
            x = self.attn_layer_norm1(x)
        x = torch.cat((residual , x),2)
        x = self.dropout(x)

#         residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
#         x = self.dropout(x)
#         x = residual + x
        if self.ln:
            x = self.final_layer_norm(x)
        return x, attn
    
    
class MHA_cross_pos2(nn.Module):
    def __init__(self,emb,head,ffn_emb,ln=1,drop=0.1,nopos=0):
        super(MHA_cross_pos2, self).__init__()
        self.ln=ln
        self.nopos=nopos
        self.pos=nn.Linear(3,emb)
        self.multihead_attn0 = torch.nn.MultiheadAttention(emb, head)
        self.multihead_attn1 = torch.nn.MultiheadAttention(emb, head)
        self.dropout = nn.Dropout(p=drop)
        if ln:
            self.attn_layer_norm0=torch.nn.LayerNorm(emb)
            self.attn_layer_norm1=torch.nn.LayerNorm(emb)
            self.final_layer_norm=torch.nn.LayerNorm(emb)
        self.activation_fn = nn.GELU()
        self.fc1=nn.Linear(emb*2,ffn_emb)
        self.fc2=nn.Linear(ffn_emb,emb)



    def forward(self, px,x,py,y,mx=None,my=None,mxx=None,mxy=None):
        if self.nopos:
            x=x.permute(2,0,1).contiguous()
            y=y.permute(2,0,1).contiguous()
            
        else:
            x=(self.pos(px).transpose(1,2)+x).permute(2,0,1).contiguous()
            y=(self.pos(py).transpose(1,2)+y).permute(2,0,1).contiguous()

        residual = x
        x, attn = self.multihead_attn0(
            query=x,
            key=x,
            value=x,
            key_padding_mask=mx,
            attn_mask=mxx,
        )
        x = self.dropout(x)
        x = residual + x
        if self.ln:
            x = self.attn_layer_norm0(x)
        
        residual = x
        x, attn = self.multihead_attn1(
            query=x,
            key=y,
            value=y,
            key_padding_mask=my,
            attn_mask=mxy,
        )
#         x = self.dropout(x)
        if self.ln:
            x = self.attn_layer_norm1(x)
        x = torch.cat((residual , x),2)
        x = self.dropout(x)

#         residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
#         x = self.dropout(x)
#         x = residual + x
        if self.ln:
            x = self.final_layer_norm(x)
        return x, attn
    
class MHA_cross_pos3(nn.Module):
    def __init__(self,emb,head,ffn_emb,ln=1,drop=0.1):
        super(MHA_cross_pos3, self).__init__()
        self.ln=ln
        self.pos=nn.Linear(3,emb)
#         self.multihead_attn0 = torch.nn.MultiheadAttention(emb, head)
        self.multihead_attn1 = torch.nn.MultiheadAttention(emb, head)
        self.dropout = nn.Dropout(p=drop)
        if ln:
#             self.attn_layer_norm0=torch.nn.LayerNorm(emb)
            self.attn_layer_norm1=torch.nn.LayerNorm(emb)
            self.final_layer_norm=torch.nn.LayerNorm(32)
        
        self.activation_fn = nn.GELU()
        self.fc1=nn.Linear(emb*2,ffn_emb)
#         self.fc2=nn.Linear(ffn_emb,emb)
        self.fc2=nn.Linear(ffn_emb,32)



    def forward(self, px,x,py,y,mx=None,my=None,mxx=None,mxy=None):
        x=x.permute(2,0,1).contiguous()
        y=y.permute(2,0,1).contiguous()

#         residual = x
#         x, attn = self.multihead_attn0(
#             query=x,
#             key=x,
#             value=x,
#             key_padding_mask=mx,
#             attn_mask=mxx,
#         )
#         x = self.dropout(x)
#         x = residual + x
#         if self.ln:
#             x = self.attn_layer_norm0(x)
        
        residual = x
        x, attn = self.multihead_attn1(
            query=x,
            key=y,
            value=y,
            key_padding_mask=my,
            attn_mask=mxy,
        )
#         x = self.dropout(x)
        if self.ln:
            x = self.attn_layer_norm1(x)
        x = torch.cat((residual , x),2)
        x = self.dropout(x)

#         residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
#         x = self.dropout(x)
#         x = residual + x
        if self.ln:
            x = self.final_layer_norm(x)
        return x, attn