import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
import pdb
from transformers import AutoImageProcessor, Swinv2Model, AutoModelForImageClassification, AutoImageProcessor, AutoModel, MobileNetV2Model
from torch.optim.swa_utils import AveragedModel
from model.vit import ViT
# Define SegNet
# The implementation of SegNet is from https://github.com/lorenmt/mtan

class SegNet(nn.Module):
    def __init__(self, type_='standard', class_nb=13):
        super(SegNet, self).__init__()
        # initialise network parameters
        self.type = type_
        if self.type == 'wide':
            #filter = [512, 128, 256, 512, 1024] # for swin
            filter = [64, 128, 256, 512, 1024] # for cnn
        else:
            #filter = [512, 128, 256, 512, 512] # for swin
            filter = [64, 128, 256, 512, 1024] # for cnn

        self.class_nb = class_nb

        #self.backbone = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256", cache_dir='hf_cache')

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([filter[i + 1], filter[i + 1]]),
                                                         self.conv_layer([filter[i + 1], filter[i + 1]])))
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([filter[i], filter[i]]),
                                                         self.conv_layer([filter[i], filter[i]])))


        # define task specific layers
        #self.teacher_task1 = DecoderWithCrossAttention2(512, 5, self.class_nb)
        """         self.teacher_task1 = ViT(
                                    image_size = (288, 384),
                                    patch_size = 96,
                                    num_classes = 1000,
                                    dim = 128,
                                    depth = 1,
                                    heads = 4,
                                    mlp_dim = 2048,
                                    dropout = 0.1,
                                    emb_dropout = 0.1,
                                    output_dim=self.class_nb
                                )
        self.student_task1 = ViT(
                                    image_size = (288, 384),
                                    patch_size = 96,
                                    num_classes = 1000,
                                    dim = 128,
                                    depth = 1,
                                    heads = 4,
                                    mlp_dim = 2048,
                                    dropout = 0.1,
                                    emb_dropout = 0.1,
                                    output_dim=self.class_nb
                                )

        self.teacher_task2 = ViT(
                                    image_size = (288, 384),
                                    patch_size = 96,
                                    num_classes = 1000,
                                    dim = 128,
                                    depth = 1,
                                    heads = 4,
                                    mlp_dim = 2048,
                                    dropout = 0.1,
                                    emb_dropout = 0.1,
                                    output_dim=1
                                )

        self.student_task2 = ViT(
                                    image_size = (288, 384),
                                    patch_size = 96,
                                    num_classes = 1000,
                                    dim = 128,
                                    depth = 1,
                                    heads = 4,
                                    mlp_dim = 2048,
                                    dropout = 0.1,
                                    emb_dropout = 0.1,
                                    output_dim=1
                                )

        self.teacher_task3 = ViT(
                                    image_size = (288, 384),
                                    patch_size = 96,
                                    num_classes = 1000,
                                    dim = 128,
                                    depth = 1,
                                    heads = 4,
                                    mlp_dim = 2048,
                                    dropout = 0.1,
                                    emb_dropout = 0.1,
                                    output_dim=3
                                )
        self.student_task3 = ViT(
                                    image_size = (288, 384),
                                    patch_size = 96,
                                    num_classes = 1000,
                                    dim = 128,
                                    depth = 1,
                                    heads = 4,
                                    mlp_dim = 2048,
                                    dropout = 0.1,
                                    emb_dropout = 0.1,
                                    output_dim=3
                                )      """                    
        
        self.availability_embedding = nn.Linear(3, 64)

        self.student_task1 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
                                        nn.Conv2d(in_channels=filter[0], out_channels=self.class_nb, kernel_size=1, padding=0))
        self.teacher_task1 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
                                        nn.Conv2d(in_channels=filter[0], out_channels=self.class_nb, kernel_size=1, padding=0))
        self.student_task2 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
                                        nn.Conv2d(in_channels=filter[0], out_channels=1, kernel_size=1, padding=0))
        self.teacher_task2 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
                                        nn.Conv2d(in_channels=filter[0], out_channels=1, kernel_size=1, padding=0))
        self.student_task3 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
                                        nn.Conv2d(in_channels=filter[0], out_channels=3, kernel_size=1, padding=0))
        self.teacher_task3 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
                                        nn.Conv2d(in_channels=filter[0], out_channels=3, kernel_size=1, padding=0))

        """ self.teacher_task1 = DecoderWithCrossAttention2(512, 5, self.class_nb)
        self.student_task1 = DecoderWithCrossAttention2(512, 5, self.class_nb)

        self.teacher_task2 = DecoderWithCrossAttention2(512, 5, 1)
        self.student_task2 = DecoderWithCrossAttention2(512, 5, 1)

        self.teacher_task3 = DecoderWithCrossAttention2(512, 5, 3)
        self.student_task3 = DecoderWithCrossAttention2(512, 5, 3) """
    

        self.linear_layer_sem = nn.Sequential(
                                            nn.Linear(in_features=1, out_features=64),  # Linear layer
                                            nn.ReLU()                                    # ReLU activation
                                        )
        self.linear_layer_depth = nn.Sequential(
                                            nn.Linear(in_features=1, out_features=64),  # Linear layer
                                            nn.ReLU()                                    # ReLU activation
                                        )
        self.linear_layer_norm = nn.Sequential(
                                            nn.Linear(in_features=3, out_features=64),  # Linear layer
                                            nn.ReLU()                                    # ReLU activation
                                        )
        self.linear_layers = nn.ModuleList([self.linear_layer_sem, self.linear_layer_depth, self.linear_layer_norm])

    
        #self.teachers = nn.ModuleList([self.teacher_task1, self.teacher_task2, self.teacher_task3])
        self.students = nn.ModuleList([self.student_task1, self.student_task2, self.student_task3])
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

    def forward(self, x=None, aux_input=None, latent=None, avail_embed=None, pred=None):

        #feature_maps = self.backbone(x, output_hidden_states=True).reshaped_hidden_states
        """ feature_maps = self.backbone(x, output_hidden_states=True).hidden_states
        #breakpoint()
        interpolated_features = [
            F.interpolate(f, size=(288, 384), mode='bilinear', align_corners=False)
            for f in feature_maps
        ]
        #breakpoint()
        feat = feature_maps[-1]
        latent_representation = self.channel_reduction(torch.cat(interpolated_features, dim=1))   """


        if x is not None:
            g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for _ in range(5))
            for i in range(5):
                g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

            # global shared encoder-decoder network
            for i in range(5):
                if i == 0:
                    g_encoder[i][0] = self.encoder_block[i](x)
                    g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                    g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
                else:
                    g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                    g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                    g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            feat = [g_maxpool[i]]
            # feat = [g_maxpool]
            for i in range(5):
                if i == 0:
                    g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                    g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                    g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
                else:
                    g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                    g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                    g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

            feat.append(g_decoder[i][1])
            latent_representation = g_decoder[i][1]
            """ binary_vectors = [torch.tensor([1, 1, 1])]

             # Get indices where Z == 1
            indices_list = [torch.nonzero(vec, as_tuple=True)[0] for vec in binary_vectors]
            max_length = 3
            padded_indices = torch.stack([
                torch.cat([idx, -1 * torch.ones(max_length - len(idx), dtype=torch.long)]) if len(idx) < max_length else idx[:max_length]
                for idx in indices_list
            ])
            Z_embed = self.availability_embedding(padded_indices.cuda()).sum(dim=1)   # Sum embeddings of available tasks
            breakpoint()
            Z_embed = Z_embed.view(1, 64, 1, 1)  
            Z_embed = Z_embed.expand_as(latent_representation)
            latent_representation = latent_representation + Z_embed """
            t1_pred = F.log_softmax(self.student_task1(latent_representation), dim=1)
            t2_pred = self.student_task2(latent_representation)
            t3_pred = self.student_task3(latent_representation)
            t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)
            return [t1_pred, t2_pred, t3_pred], self.logsigma, feat, latent_representation
        else:
            Z_embed = self.availability_embedding(avail_embed)
            Z_embed = Z_embed.view(1, 64, 1, 1)
            Z_embed = Z_embed.expand_as(latent)
            latent_representation = latent + Z_embed
            new_pred = pred 

            for i, tag in enumerate(avail_embed):
                if tag == 0 and i == 0:
                    task_information = aux_input[0][0].unsqueeze(0)
                    reshaped_aux = task_information.permute(1, 2, 0)
                    auxiliary_task_inf = self.linear_layers[i](reshaped_aux)
                    final_aux = auxiliary_task_inf.permute(2, 0, 1)
                    final_aux = final_aux.expand_as(latent)
                    latent_representation = latent_representation + final_aux
                    new_pred[0] = F.log_softmax(self.student_task1(latent_representation), dim=1)
                elif tag == 0 and i == 1:
                    task_information = aux_input[0][1].unsqueeze(0)
                    reshaped_aux = task_information.permute(1, 2, 0)
                    auxiliary_task_inf = self.linear_layers[i](reshaped_aux)
                    final_aux = auxiliary_task_inf.permute(2, 0, 1)
                    final_aux = final_aux.expand_as(latent)
                    latent_representation = latent_representation + final_aux
                    new_pred[1] = self.student_task2(latent_representation)
                elif tag == 0 and i == 2:
                    task_information = aux_input[0][2:5]
                    reshaped_aux = task_information.permute(1, 2, 0)
                    auxiliary_task_inf = self.linear_layers[i](reshaped_aux)
                    final_aux = auxiliary_task_inf.permute(2, 0, 1)
                    final_aux = final_aux.expand_as(latent)
                    latent_representation = latent_representation + final_aux
                    pred_surface = self.student_task3(latent_representation)
                    new_pred[2] = pred_surface / torch.norm(pred_surface, p=2, dim=1, keepdim=True)
            return new_pred

    def conv_layer(self, channel):
        if self.type == 'deep':
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channel[1], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True)
            )
        return conv_block

    def ema_update(self, alpha=0.99):
        for i in range(len(self.teachers)):
            for teacher_param, student_param in zip(self.teachers[i].parameters(), self.students[i].parameters()):
                teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data

    def update_ema(self, global_step, alpha=0.99):
        """
        https://github.com/colinlaganier/MeanTeacherSegmentation/blob/main/main.py
        Update the ema model weights with the model weights
        Args:
            model (torch.nn.Module): model
            ema_model (torch.nn.Module): ema model
            alpha (float): alpha
            global_step (int): global step
        """
        
        # Set alpha to 0.999 at the beginning and then linearly decay
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for i in range(len(self.teachers)):
            for teacher_param, student_param in zip(self.teachers[i].parameters(), self.students[i].parameters()):
                teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)

    def teacher_forward(self, x, index, gts=None):
        if index == 0:
            #prediction = torch.argmax(F.log_softmax(self.teachers[index](x), dim=1), dim=1)
             #prediction = F.softmax(self.teachers[index](x), dim=1).max(1)
             prediction = self.teachers[index](x)
        elif index == 1:
            prediction = self.teachers[index](x)
        elif index == 2:
            prediction = self.teachers[index](x)
            prediction = prediction / torch.norm(prediction, p=2, dim=1, keepdim=True)
        return prediction

    def model_fit(self, x_pred1, x_output1, x_pred2, x_output2, x_pred3, x_output3, seg_binary_mask=None):
        # Compute supervised task-specific loss for all tasks when all task labels are available

        # binary mark to mask out undefined pixel space
        binary_mask = (torch.sum(x_output2, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).cuda()
        binary_mask_3 = (torch.sum(x_output3, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).cuda()

        # semantic loss: depth-wise cross entropy
        #breakpoint()
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


class DecoderWithCrossAttention2(nn.Module):
    def __init__(self, in_channels, aux_dim, output_channels, num_heads=4):
        super(DecoderWithCrossAttention2, self).__init__()
        # Transposed Convolution for Upsampling
        self.gate = nn.Conv2d(in_channels=in_channels*2, out_channels=1, kernel_size=1)
        self.project_aux = nn.Conv2d(in_channels=aux_dim, out_channels=in_channels, kernel_size=1)
        self.conv_out = nn.Conv2d(in_channels=in_channels, out_channels=output_channels, kernel_size=3, padding=1)
        
    def forward(self, x, aux_input=None):
        # Initial Conv Layer
        if aux_input is not None:
            projected_aux = self.project_aux(aux_input)
            combined = torch.cat([x, projected_aux], dim=1) 
            gate = torch.sigmoid(self.gate(combined))

            x = gate * x + (1 - gate) * projected_aux

        out = self.conv_out(x)

        return out

class DecoderWithCrossAttention(nn.Module):
    def __init__(self, in_channels, aux_dim, output_channels, num_heads=4):
        super().__init__()
        # Transposed Convolution for Upsampling
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.upsample = nn.ConvTranspose2d(in_channels, output_channels, kernel_size=(4, 4), stride=(4, 4), padding=(1,1), output_padding=(0,0))
        
        # Cross Attention
        num_layers = 3
        self.layers = nn.ModuleList(
            [CrossAttentionLayer(embed_dim=512, num_heads=4, dropout=0.1) for _ in range(num_layers)]
        )
        #self.cross_attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
    
        # Linear Layer
        self.linear = nn.Linear(5, 512)

        # Normalisation Layer
        self.norm = nn.LayerNorm(in_channels)

        # Pooling Layers to reduce complexity
        # 2D
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2)

        # 1D
        self.pool1D = nn.AvgPool1d(kernel_size=4, stride=4)
        self.pool1D2 = nn.AvgPool1d(kernel_size=4, stride=4)
        
    def forward(self, x, aux_input=None):
        # Initial Conv Layer
        x = self.conv1(x)
        # Cross-Attention: Skip if no aux_input
        
        x = self.pool1(x)
        B, C, H, W = x.shape

        if aux_input is not None:
            aux_input = self.pool2(aux_input)
            B_, C_, H_, W_ = aux_input.shape

        # Flatten spatial dimensions for MultiheadAttention
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (batch_size, seq_len, in_channels)

        if aux_input is not None:
            aux_flat = aux_input.view(B_, C_, -1).permute(0, 2, 1)  # (batch_size, seq_len, aux_dim)
            aux_flat = self.linear(aux_flat)

        A, D = 165, 165
        x_pool = x_flat.transpose(1, 2)  # [Batch, Embedding Size, Sequence Length]
        x_pooled = self.pool1D(x_pool).transpose(1, 2)  # [1, 6828, 512]

        if aux_input is not None:
            aux_pool = aux_flat.transpose(1, 2)  # [Batch, Embedding Size, Sequence Length]
            aux_pooled = self.pool1D2(aux_pool).transpose(1, 2)  # [1, 6828, 512]

        # Cross-Attention
        #attended_x, _ = self.cross_attention(query=x_pooled, key=aux_pooled, value=aux_pooled)
        if aux_input is not None:
            for layer in self.layers:
                x_pooled = layer(query=x_pooled, key=aux_pooled, value=aux_pooled)
        else:
            for layer in self.layers:
                x_pooled = layer(query=x_pooled, key=x_pooled, value=x_pooled)

        attended_x = x_pooled
        target_seq_len = 6889
        padding_len = target_seq_len - attended_x.size(1)  # 6889 - 6828 = 61
        attended_x = torch.nn.functional.pad(attended_x, (0, 0, 0, padding_len))
        x = attended_x.permute(0, 2, 1).view(B, C, 83, 83)  # Reshape back to spatial dims

        # Normalization
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            
        # Transposed Convolution for Upsampling
        x = self.upsample(x)
        x = F.interpolate(x, size=(288, 384), mode='bilinear', align_corners=False)
        return x

class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # Multi-head cross-attention
        attn_output, _ = self.multihead_attn(query, key, value)
        query = query + self.dropout(attn_output)
        query = self.norm1(query)

        # Feed-forward network
        ffn_output = self.ffn(attn_output)
        query = query + self.dropout(ffn_output)
        output = self.norm2(query)

        return output

