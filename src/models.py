import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import pytorch_lightning as pl
from utils import exists, default, SupConLoss, Siren, plot_islands_agreement
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from argparse import Namespace

TOKEN_ATTEND_SELF_VALUE = -5e-4

def ConvTokenizer(in_channels=3, embedding_dim=64, patch_size=None):
    return nn.Sequential(   nn.Conv2d(in_channels, embedding_dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                            nn.BatchNorm2d(embedding_dim // 2),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(embedding_dim // 2, embedding_dim // 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                            nn.BatchNorm2d(embedding_dim // 2),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(embedding_dim // 2, embedding_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                            nn.BatchNorm2d(embedding_dim),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1)),
                            Rearrange('b d (h p1) (w p2) -> b (h w) (d p1 p2)', p1 = patch_size, p2 = patch_size), 
                        )

def ColumnNet(FLAGS, dim, groups, mult = 4, activation = nn.GELU):
    total_dim = dim * groups
    num_patches = (FLAGS.conv_image_size // FLAGS.patch_size) ** 2
    return  nn.Sequential(  Rearrange('b n l d -> b (l d) n'),
                            nn.LayerNorm(num_patches),
                            nn.Conv1d(total_dim, total_dim * mult, 1, groups = groups),
                            activation(),
                            nn.LayerNorm(num_patches),
                            nn.Conv1d(total_dim * mult, total_dim, 1, groups = groups),
                            Rearrange('b (l d) n -> b n l d', l = groups)
                        )


class ConsensusAttention(nn.Module):
    def __init__(self, num_patches_side, attend_self = True, local_consensus_radius = 0):
        super().__init__()
        self.attend_self = attend_self
        self.local_consensus_radius = local_consensus_radius

        if self.local_consensus_radius > 0:
            coords = torch.stack(torch.meshgrid( torch.arange(num_patches_side), torch.arange(num_patches_side) )).float()
            coords = rearrange(coords, 'c h w -> (h w) c')
            dist = torch.cdist(coords, coords)
            mask_non_local = dist > self.local_consensus_radius
            mask_non_local = rearrange(mask_non_local, 'i j -> () i j')
            self.register_buffer('non_local_mask', mask_non_local)

    def forward(self, levels):
        _, n, _, d, device = *levels.shape, levels.device
        q, k, v = levels, F.normalize(levels, dim = -1), levels

        sim = einsum('b i l d, b j l d -> b l i j', q, k) * (d ** -0.5)

        if not self.attend_self:
            self_mask = torch.eye(n, device = device, dtype = torch.bool)
            self_mask = rearrange(self_mask, 'i j -> () () i j')
            sim.masked_fill_(self_mask, TOKEN_ATTEND_SELF_VALUE)

        if self.local_consensus_radius > 0:
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(self.non_local_mask, max_neg_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b l i j, b j l d -> b i l d', attn, levels)
        return out

##############################################

class Agglomerator(pl.LightningModule):
    def __init__(self, FLAGS, *, consensus_self = False, local_consensus_radius = 0):
        super().__init__()
        if not isinstance(FLAGS, Namespace):
            FLAGS = Namespace(**FLAGS.flag_values_dict())
        self.save_hyperparameters(FLAGS) # Errors out: "can't pickle FlagValues"

        self.num_gpus = len(FLAGS.gpus)
        self.FLAGS = FLAGS

        num_patches_side = (FLAGS.conv_image_size // FLAGS.patch_size)
        num_patches =  num_patches_side ** 2
        self.features = []
        self.labels = []
        self.iters = default(FLAGS.iters, FLAGS.levels * 2)
        self.batch_acc = 0

        self.wl = torch.nn.parameter.Parameter(torch.tensor(0.25, device=self.device), requires_grad=True)
        self.wBU = torch.nn.parameter.Parameter(torch.tensor(0.25, device=self.device), requires_grad=True)
        self.wTD = torch.nn.parameter.Parameter(torch.tensor(0.25, device=self.device), requires_grad=True)
        self.wA = torch.nn.parameter.Parameter(torch.tensor(0.25, device=self.device), requires_grad=True)

        self.image_to_tokens = ConvTokenizer(in_channels=FLAGS.n_channels, 
                                             embedding_dim=FLAGS.patch_dim // (FLAGS.patch_size ** 2), 
                                             patch_size = FLAGS.patch_size)

        self.contrastive_head = nn.Sequential(  nn.LayerNorm(FLAGS.patch_dim),
                                                nn.Dropout(p=FLAGS.dropout),
                                                Rearrange('b n d -> b (n d)'),
                                                nn.LayerNorm(num_patches * FLAGS.patch_dim),
                                                nn.Dropout(p=FLAGS.dropout),
                                                nn.Linear(num_patches * FLAGS.patch_dim, num_patches * FLAGS.patch_dim),
                                                nn.LayerNorm(num_patches * FLAGS.patch_dim),
                                                nn.GELU(),
                                                nn.LayerNorm(num_patches * FLAGS.patch_dim),
                                                nn.Dropout(p=FLAGS.dropout),
                                                nn.Linear(num_patches * FLAGS.patch_dim, FLAGS.contr_dim)   )

        self.classification_head_from_contr = nn.Sequential(nn.Linear(FLAGS.contr_dim, FLAGS.contr_dim),
                                                            nn.GELU(),
                                                            nn.Linear(FLAGS.contr_dim, FLAGS.n_classes)     )

        self.init_levels = nn.Parameter(torch.randn(FLAGS.levels, FLAGS.patch_dim))
        self.bottom_up = ColumnNet(FLAGS, dim = FLAGS.patch_dim, activation=nn.GELU, groups = FLAGS.levels)
        self.top_down = ColumnNet(FLAGS, dim = FLAGS.patch_dim, activation=Siren, groups = FLAGS.levels - 1)
        self.attention = ConsensusAttention(num_patches_side, attend_self = consensus_self, local_consensus_radius = local_consensus_radius)

        self.load_chkp(FLAGS)

    def load_chkp(self, FLAGS):
        if FLAGS.ckpt_dir: # is not None: # FLAGS.resume_training 
            import os
            dir = os.path.join(os.getcwd(), FLAGS.ckpt_dir)
            self.load_from_checkpoint(dir, FLAGS=FLAGS, strict=False)

    def forward(self, img, levels = None):
        b, device = img.shape[0], img.device

        tokens = self.image_to_tokens(img)
        n = tokens.shape[1]

        bottom_level = tokens
        bottom_level = rearrange(bottom_level, 'b n d -> b n () d')

        if not exists(levels):
            levels = repeat(self.init_levels, 'l d -> b n l d', b = b, n = n)

        hiddens = [levels]

        num_contributions = torch.empty(self.FLAGS.levels, device = device).fill_(4)
        num_contributions[-1] = 3 

        for _ in range(self.iters):
            levels_with_input = torch.cat((bottom_level, levels), dim = -2)

            bottom_up_out = self.bottom_up(levels_with_input[..., :-1, :])

            top_down_out = self.top_down(torch.flip(levels_with_input[..., 2:, :], [2]))
            top_down_out = F.pad(torch.flip(top_down_out, [2]), (0, 0, 0, 1), value = 0.)

            consensus = self.attention(levels)

            levels_sum = torch.stack((  levels * self.wl, \
                                        bottom_up_out * self.wBU, \
                                        top_down_out * self.wTD, \
                                        consensus * self.wA           )).sum(dim = 0)
            levels_mean = levels_sum / rearrange(num_contributions, 'l -> () () l ()')

            self.log('Weights/wl', self.wl)
            self.log('Weights/wBU', self.wBU)
            self.log('Weights/wTD', self.wTD)
            self.log('Weights/wA', self.wA)

            levels = levels_mean
            hiddens.append(levels)

        all_levels = torch.stack(hiddens)

        top_level = all_levels[self.FLAGS.denoise_iter, :, :, -1]

        top_level = self.contrastive_head(top_level)
        top_level = F.normalize(top_level, dim=1)

        return top_level, all_levels[-1,0,:,:,:]

    def training_step(self, train_batch, batch_idx):
        image, label = train_batch[0:2]
        # self.training_batch_idx = batch_idx

        if(not self.FLAGS.supervise):
            image = torch.cat([image[0], image[1]], dim=0)

        if(self.FLAGS.supervise):
            with torch.no_grad():
                top_level, _ = self.forward(image)
                self.features.append(list(top_level.data.cpu().numpy()))
                self.labels.append(label.cpu().numpy())
        else:
            top_level, toplot = self.forward(image)
            if(self.FLAGS.plot_islands):
                plot_islands_agreement(toplot, image[0,:,:,:])

        if(self.FLAGS.supervise):
            output = self.classification_head_from_contr(top_level)
            loss = F.cross_entropy(output, label)
            self.batch_acc = self.accuracy(output.data,label,topk=(1,))[0]
            self.log('Training/accuracy', self.batch_acc, prog_bar=True, sync_dist=True)
            self.batch_acc = 0

        else:
            f1, f2 = torch.split(top_level, [self.FLAGS.batch_size // self.num_gpus, self.FLAGS.batch_size // self.num_gpus], dim=0)
            output = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            criterion = SupConLoss(temperature=self.FLAGS.temperature)
            loss = criterion(output, label)

        self.log('Training/loss', loss, sync_dist=True)
        # self.log('Training/LR', self.optimizer.param_groups[0]['lr'], prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        image = val_batch[0]
        label = val_batch[1]
        self.val_batch_idx = batch_idx

        if (not self.FLAGS.supervise):
            image = torch.cat([image[0], image[1]], dim=0)

        if (self.FLAGS.supervise):
            with torch.no_grad():
                top_level, _ = self.forward(image)
                self.features.append(list(top_level.data.cpu().numpy()))
                self.labels.append(label.cpu().numpy())
        else:
            top_level, toplot = self.forward(image)
            if(self.FLAGS.plot_islands):
                plot_islands_agreement(toplot, image[0,:,:,:])

        if (self.FLAGS.supervise):
            output = self.classification_head_from_contr(top_level)
            loss = F.cross_entropy(output, label)
            self.batch_acc = self.accuracy(output.data,label,topk=(1,))[0]
            self.log('Validation/accuracy', self.batch_acc, prog_bar=True, sync_dist=True)
            self.batch_acc = 0

        else:
            f1, f2 = torch.split(top_level, [self.FLAGS.batch_size // self.num_gpus, self.FLAGS.batch_size // self.num_gpus], dim=0)
            output = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            criterion = SupConLoss(temperature=self.FLAGS.temperature)
            loss = criterion(output, label)

        self.log('Validation/loss', loss, sync_dist=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        image = test_batch[0]
        label = test_batch[1]
        self.test_batch_idx = batch_idx

        if(not self.FLAGS.supervise):
            image = torch.cat([image[0], image[1]], dim=0)

        if(self.FLAGS.supervise):
            with torch.no_grad():
                top_level, _ = self.forward(image)
                self.features.append(list(top_level.data.cpu().numpy()))
                self.labels.append(label.cpu().numpy())
        else:
            top_level, toplot = self.forward(image)
            if(self.FLAGS.plot_islands):
                plot_islands_agreement(toplot, image[0,:,:,:])

        if(self.FLAGS.supervise):
            output = self.classification_head_from_contr(top_level)
            loss = F.cross_entropy(output, label)
            self.batch_acc = self.accuracy(output.data,label,topk=(1,))[0]
            self.log('Test/accuracy', self.batch_acc, prog_bar=True, sync_dist=True)

        else:
            f1, f2 = torch.split(top_level, [self.FLAGS.batch_size // self.num_gpus, self.FLAGS.batch_size // self.num_gpus], dim=0)
            output = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            criterion = SupConLoss(temperature=self.FLAGS.temperature)
            loss = criterion(output, label)

        self.log('Test/loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.FLAGS.learning_rate,
            weight_decay=self.FLAGS.weight_decay,
        )
        steps_per_epoch = 45000 // self.FLAGS.batch_size
        scheduler_dict = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.FLAGS.learning_rate,
                epochs=2000,
                steps_per_epoch=steps_per_epoch,
            ),
            'interval': 'step',
        }
        return {'optimizer': self.optimizer, 'lr_scheduler': scheduler_dict}

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res