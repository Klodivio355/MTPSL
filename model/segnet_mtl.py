import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
import pdb
from transformers import AutoImageProcessor, Swinv2Model

# Define SegNet
# The implementation of SegNet is from https://github.com/lorenmt/mtan

class SegNet(nn.Module):
    def __init__(self, type_='standard', class_nb=13):
        super(SegNet, self).__init__()
        # initialise network parameters
        self.type = type_
        if self.type == 'wide':
            filter = [512, 128, 256, 512, 1024]
        else:
            filter = [512, 128, 256, 512, 512]

        self.class_nb = class_nb

        self.backbone = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256", cache_dir='hf_cache')

        # define task specific layers
        self.teacher_task1 = DecoderWithCrossAttention(512, 5, self.class_nb)
        self.student_task1 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
                                        nn.Conv2d(in_channels=filter[0], out_channels=self.class_nb, kernel_size=1, padding=0))

        self.teacher_task2 = DecoderWithCrossAttention(512, 5, 1)
        self.student_task2 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
                                        nn.Conv2d(in_channels=filter[0], out_channels=1, kernel_size=1, padding=0))

        self.teacher_task3 = DecoderWithCrossAttention(512, 5, 3)
        self.student_task3 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
                                        nn.Conv2d(in_channels=filter[0], out_channels=3, kernel_size=1, padding=0))

        self.teachers = nn.ModuleList([self.teacher_task1, self.teacher_task2, self.teacher_task3])
        self.channel_reduction = nn.Conv2d(2208, 512, kernel_size=1, stride=1, bias=False)

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # This is the parameters for uncertainty loss weighting strategies (https://arxiv.org/abs/1705.07115)
        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        feature_maps = self.backbone(x, output_hidden_states=True).reshaped_hidden_states
        interpolated_features = [
            F.interpolate(f, size=(288, 384), mode='bilinear', align_corners=False)
            for f in feature_maps
        ]

        feat = feature_maps[-1]
        latent_representation = self.channel_reduction(torch.cat(interpolated_features, dim=1))  

        # define task prediction layers
        t1_pred = F.log_softmax(self.student_task1(latent_representation), dim=1)
        t2_pred = self.student_task2(latent_representation)
        t3_pred = self.student_task3(latent_representation)
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return [t1_pred, t2_pred, t3_pred], self.logsigma, feat, latent_representation

    def teacher_forward(self, x, index, gts):
        prediction = self.teachers[index](x, gts)
        return prediction

    def model_fit(self, x_pred1, x_output1, x_pred2, x_output2, x_pred3, x_output3):
        # Compute supervised task-specific loss for all tasks when all task labels are available

        # binary mark to mask out undefined pixel space
        binary_mask = (torch.sum(x_output2, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).cuda()
        binary_mask_3 = (torch.sum(x_output3, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).cuda()

        # semantic loss: depth-wise cross entropy
        loss1 = F.nll_loss(x_pred1, x_output1, ignore_index=-1)

        # depth loss: l1 norm
        loss2 = torch.sum(torch.abs(x_pred2 - x_output2) * binary_mask) / torch.nonzero(binary_mask).size(0)

        # normal loss: dot product
        loss3 = 1 - torch.sum((x_pred3 * x_output3) * binary_mask_3) / torch.nonzero(binary_mask_3).size(0)

        return [loss1, loss2, loss3]


    def model_fit_task(self, x_pred, x_output, task='semantic'):
        # Compute supervised task-specific loss for a specific task, [semantic, depth, normal]

        # binary mark to mask out undefined pixel space
        if task == 'semantic':
            # semantic loss: depth-wise cross entropy
            loss = F.nll_loss(x_pred, x_output, ignore_index=-1)
        elif task == 'depth':
            # binary mark to mask out undefined pixel space
            binary_mask = (torch.sum(x_output, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).cuda()
            # depth loss: l1 norm
            loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask).size(0)
        elif task == 'normal':
            # binary mark to mask out undefined pixel space
            binary_mask = (torch.sum(x_output, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).cuda()
            # normal loss: dot product
            loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(binary_mask).size(0)
        return loss

    def model_unsup(self, x_pred_s, x_pred_dt, x_pred_ds, x_pred_nt, x_pred_ns, threshold1=0.95, threshold2=0.1, threshold3=0.1):
        # compute unsupervised loss for all tasks

        loss1 = self.seg_con(x_pred_s, threshold1)
        loss2 = self.depth_con(x_pred_dt, x_pred_ds, threshold2)
        loss3 = self.normal_con(x_pred_nt, x_pred_ns, threshold3)

        return [loss1, loss2, loss3]

    def seg_con(self, x_pred, x_pred_t=None, threshold=0.95):
        # unsupervised loss for segmentation.
        if x_pred_t is None:
            prob, pseudo_labels = F.softmax(x_pred, dim=1).max(1)
            binary_mask = (prob > threshold).type(torch.FloatTensor).cuda()
            loss = F.nll_loss(x_pred, pseudo_labels, reduction='none') * binary_mask
        else:
            prob, pseudo_labels = F.softmax(x_pred_t, dim=1).max(1)
            # x_pred = F.log_softmax(x_pred, dim=1)
            binary_mask = (prob > threshold).type(torch.FloatTensor).cuda()
            loss = F.nll_loss(x_pred, pseudo_labels, reduction='none') * binary_mask
        return loss.mean()

    def depth_con(self, x_pred, x_pred_s, threshold=0.1):
        # unsupervised loss for depth.

        binary_mask = ((x_pred.data - x_pred_s.data).abs() < threshold).type(torch.FloatTensor).cuda()
        loss = ((x_pred.data - x_pred_s).abs() * binary_mask).mean()

        return loss

    def normal_con(self, x_pred, x_pred_s, threshold=0.1):
        # unsupervised loss for surface normal

        loss = 1 - (x_pred.data * x_pred_s)
        binary_mask = (loss.data < threshold).type(torch.FloatTensor).cuda()
        loss = (loss * binary_mask).mean()
        return loss
    
    # evaluation metircs from https://github.com/lorenmt/mtan
    def compute_miou(self, x_pred, x_output):
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        for i in range(batch_size):
            true_class = 0
            first_switch = True
            for j in range(self.class_nb):
                pred_mask = torch.eq(x_pred_label[i], j * torch.ones(x_pred_label[i].shape).type(torch.LongTensor).cuda())
                true_mask = torch.eq(x_output_label[i], j * torch.ones(x_output_label[i].shape).type(torch.LongTensor).cuda())
                mask_comb = pred_mask.type(torch.FloatTensor) + true_mask.type(torch.FloatTensor)
                union     = torch.sum((mask_comb > 0).type(torch.FloatTensor))
                intsec    = torch.sum((mask_comb > 1).type(torch.FloatTensor))
                if union == 0:
                    continue
                if first_switch:
                    class_prob = intsec / union
                    first_switch = False
                else:
                    class_prob = intsec / union + class_prob
                true_class += 1
            if i == 0:
                batch_avg = class_prob / true_class
            else:
                batch_avg = class_prob / true_class + batch_avg
        return batch_avg / batch_size

    def compute_iou(self, x_pred, x_output):
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        for i in range(batch_size):
            if i == 0:
                pixel_acc = torch.div(torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).type(torch.FloatTensor)),
                            torch.sum((x_output_label[i] >= 0).type(torch.FloatTensor)))
            else:
                pixel_acc = pixel_acc + torch.div(torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).type(torch.FloatTensor)),
                            torch.sum((x_output_label[i] >= 0).type(torch.FloatTensor)))
        return pixel_acc / batch_size

    def depth_error(self, x_pred, x_output):
        binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).cuda()
        x_pred_true = x_pred.masked_select(binary_mask)
        x_output_true = x_output.masked_select(binary_mask)
        abs_err = torch.abs(x_pred_true - x_output_true)
        rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
        return torch.sum(abs_err) / torch.nonzero(binary_mask).size(0), torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)

    def normal_error(self, x_pred, x_output):
        binary_mask = (torch.sum(x_output, dim=1) != 0)
        error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
        error = np.degrees(error)
        return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)

class DecoderWithCrossAttention(nn.Module):
    def __init__(self, in_channels, aux_dim, output_channels, num_heads=4):
        super().__init__()
        # Transposed Convolution for Upsampling
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.upsample = nn.ConvTranspose2d(in_channels, output_channels, kernel_size=2, stride=2)
        
        # Cross Attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        #self.cross_attention = CrossAttentionLayer
        # Optional LayerNorm for stability
        self.linear = nn.Linear(5, 512)
        self.norm = nn.LayerNorm(in_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        #self.patch_size = 16  # Size of each patch


    def forward(self, x, aux_input):
        # Initial Conv Layer
        x = self.conv1(x)
        # Cross-Attention: Skip if no aux_input
        if aux_input is not None:
            x = self.pool1(x)
            aux_input = self.pool2(aux_input)
            #breakpoint()
            B, C, H, W = x.shape
            B_, C_, H_, W_ = aux_input.shape
            # Flatten spatial dimensions for MultiheadAttention
            x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (batch_size, seq_len, in_channels)
            aux_flat = aux_input.view(B_, C_, -1).permute(0, 2, 1)  # (batch_size, seq_len, aux_dim)
            aux_flat = self.linear(aux_flat)
            #breakpoint()
            # Cross-Attention
            attended_x, _ = self.cross_attention(query=x_flat, key=aux_flat, value=aux_flat)
            x = attended_x.permute(0, 2, 1).view(B, C, H, W)  # Reshape back to spatial dims
            
            # Optional normalization
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Transposed Convolution for Upsampling
        x = self.upsample(x)
        return x

class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value):
        # Multi-head cross-attention
        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = self.norm1(attn_output + query)

        # Feed-forward network
        ffn_output = self.ffn(attn_output)
        output = self.norm2(ffn_output + attn_output)

        return output

