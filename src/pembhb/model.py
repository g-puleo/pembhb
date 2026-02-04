import torch
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from line_profiler import profile
import torch.nn as nn
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from typing import Iterable
from pembhb.data import MBHBDataset
from torch.utils.data import DataLoader

from pembhb.utils import _ORDERED_PRIOR_KEYS, mbhb_collate_fn
from pembhb import ROOT_DIR
import numpy as np
# class GWTransformer(LightningModule):
#     def __init__(self, n_chunks=100, embed_dim=512, num_heads=8, num_layers=4):
#         super().__init__()
#         self.n_chunks = n_chunks
#         self.chunk_size = 10000 // n_chunks
#         self.input_dim = 3 * self.chunk_size  # 3 channels × chunk length

#         # Input projection
#         self.embedding = nn.Linear(self.input_dim, embed_dim)

#         # Positional encoding
#         self.positional_encoding = nn.Parameter(torch.randn(n_chunks, embed_dim))

#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)



#     def forward(self, data, parameters):
#         """
#         x: Tensor of shape (batch_size, 3, 10000) containing the frequency domain signal in the 3 TDI channels
#         extra_features: Tensor of shape (batch_size, 11)
#         """
#         B, C, T = data.shape
#         assert C == 3 and T == self.n_chunks * self.chunk_size, "Invalid input shape"

#         # Split into chunks and reshape
#         data = data.view(B, C, self.n_chunks, self.chunk_size)
#         data = data.permute(0, 2, 1, 3).reshape(B, self.n_chunks, -1)  # (B, n_chunks, 3 * chunk_size)

#         # Embed and add positional encodings
#         x = self.embedding(data) + self.positional_encoding  # (B, n_chunks, embed_dim)

#         # Transformer encoding
#         x = self.transformer(x)  # (B, n_chunks, embed_dim)
#         x = x.mean(dim=1)  # global average pooling over sequence: (B, embed_dim)

#         # Process extra input features
#         extra = self.extra_fc(parameters)  # (B, embed_dim)

#         # Concatenate and classify
#         combined = torch.cat([x, extra], dim=-1)  # (B, embed_dim * 2)
#         out = self.classifier(combined)  # (B, 1)

#         return out
WEEK_SI = 7 * 24 * 3600  # seconds in a week


def build_data_summary(conf):
    ''' 
    conf must be from config.yaml "data_summary" section
    '''
    ds_type = conf["type"]
    init_kwargs = conf[ds_type]
    cls = DATA_SUMMARY_REGISTRY[conf["type"]]
    return cls(**init_kwargs)

class GradnormHandler : 

    def __init__(self, parent_module: 'InferenceNetwork', config: dict ):
        """Initialises gradnorm handler

        :param parent_module: the parent lightning module (in lightningmodule __init__ call, it is 'self')
        :type parent_module: LightningModule
        :param config: must contain the following keys: 'alpha' , 'common_params', 
        :type config: dict
        """
        self.model = parent_module
        self.alpha=config["alpha"]
        self.shared_layer = getattr(self.model.data_summary, config["common_params"])
        self.shared_params_gradnorm = [p for p in self.shared_layer.parameters() if p.requires_grad]
        self.was_called_once = False
        self.model.automatic_optimization=False 
    
    def training_step(self, batch, batch_idx):
        """Performs a training step.

        :param batch: The input batch of data.
        :type batch: dict
        :param batch_idx: The index of the batch.
        :type batch_idx: int
        :return: _tuple: all_logits, task_losses, total_loss
        :rtype: tuple
        """
        
        data_f = batch['data_fd']
        data_t = batch['data_td']
        parameters = batch['source_parameters']

        # same base call, but handler decides weighting logic
        all_logits, task_losses, total_loss, gradnorm_loss = self.calc_logits_losses_gradnorm(data_f, data_t, parameters)
        #update weights w_i (actually we store their logits to ensure positivity)
        optimizer_nre, optimizer_gradnorm = self.model.optimizers()
        optimizer_gradnorm.zero_grad()
        self.model.manual_backward(gradnorm_loss, retain_graph=True)
        optimizer_gradnorm.step()
        #update model params
        optimizer_nre.zero_grad()
        self.model.manual_backward(total_loss)
        optimizer_nre.step()

        self.model.log_dict(
            {
            "L_grad": gradnorm_loss,
            "weighted_loss": total_loss
            })
        weights_normalised = torch.nn.functional.softmax(self.model.weights_loss_logits, dim=0)
        for i, w_i in enumerate(weights_normalised):
            self.model.log(f"weight_{i}", w_i.detach(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return  all_logits, task_losses, total_loss

    def compute_gradnorm_loss(self, task_losses_all):
        ########## GRADNORM LOGIC ###############
        weights = torch.nn.functional.softmax(self.model.weights_loss_logits, dim=0)
        task_losses = torch.mean(task_losses_all, dim=0) 
        weighted_task_losses = weights * task_losses
        total_loss = torch.mean(weighted_task_losses)
        #compute gradient norms G_W^{(i)}
        G_W = []
        for i, wL_i in enumerate(weighted_task_losses):
            grads = torch.autograd.grad(
                outputs=wL_i,
                inputs=self.shared_params_gradnorm,
                retain_graph=True, 
                create_graph=True
            )
            grad_norm = torch.norm(torch.cat([g.flatten() for g in grads]), p=2)
            G_W.append(grad_norm)
            self.model.log(f"gradient_norm_{self.model.output_names[i]}", grad_norm, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        G_W = torch.stack(G_W)
        G_bar = G_W.mean()

        #compute r_i(t) and GradNorm loss
        loss_ratios = task_losses / self.initial_losses  # L_i(t) / L_i(0)
        inv_rate = loss_ratios / loss_ratios.mean()      # r_i(t)
        targets = G_bar * (inv_rate ** self.alpha)
        #breakpoint()
        gradnorm_loss = torch.abs(G_W - targets.detach()).sum()  # treat targets as constant
        self.model.log("gradient_norm", G_bar, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.model
        return total_loss, gradnorm_loss


    def calc_logits_losses_gradnorm(self, data_f, data_t, parameters):
        '''
        returns: 
            all_logits: tensor of shape (2*B, num_marginals)
            tails_losses: tensor of shape (num_marginals,), the non weighted losses
            weighted_tails_losses: tensor of shape (num_marginals,) the weighted losses, for sparing computation
            weights: tensor of shape (num_marginals,), the weights used for the weighted loss (after softmax)
            loss: scalar tensor, the full weighted loss'''
        all_logits, all_loss = self.model._calc_logits_base(data_f, data_t, parameters)
        task_losses = torch.mean(all_loss, dim=0)
        if not self.was_called_once:
            print("Setting initial losses for GradNorm")
            self.initial_losses = task_losses.detach()
        self.was_called_once = True
        total_loss, gradnorm_loss = self.compute_gradnorm_loss(task_losses)
        return all_logits, task_losses, total_loss, gradnorm_loss

class MarginalClassifierHead(nn.Module):

    def __init__(self, n_data_features: int , marginals: list[list], hlayersizes: Iterable[int]):
        """Classifier head for multiple marginals. 
        Performs a binary classification for each marginal in the list of marginals. Used for TMNRE. 

        :param n_data_features: the number of features in the data summary
        :type n_data_features: int
        :param marginals: list of marginals that you want 
        :type marginals: list[list]
        :param hidden_size: sizes of the hidden layers in the classifier
        :type hidden_size: iterable[int]
        """
        super().__init__()
        self.marginals_dict = marginals
        self.classifiers = nn.ModuleList()
        for marginal in marginals:
            classifier = nn.Sequential()
            for i, output_size in enumerate(hlayersizes):
                if i == 0:
                    input_size = n_data_features + len(marginal)
                    output_size = hlayersizes[i]
                else:
                    input_size = hlayersizes[i-1]
                    output_size = hlayersizes[i]
        
                classifier.add_module(f"fc_{i}", nn.Linear(input_size, output_size))
                classifier.add_module(f"relu_{i}", nn.ReLU())

            classifier.add_module("output", nn.Linear(output_size, 1)) 
            self.classifiers.append(classifier)
        


    def forward(self, features, parameters):
        outputs = []
        for i, marginal in enumerate(self.marginals_dict):
            indices = marginal
            input_data = torch.cat([features, parameters[:, indices]], dim=-1)
            outputs.append(self.classifiers[i](input_data))
            
        return torch.cat(outputs, dim=-1)

class InferenceNetwork(LightningModule):
    """ 
    Basic FC network for TMNRE of MBHB data. 
    """
    def __init__(self,  train_conf: dict,  dataset_info: dict, normalisation: dict,  data_summarizer: nn.Module=None):  
        super().__init__()
        if data_summarizer is not None:
            self.data_summary = data_summarizer
        else:
            self.data_summary = build_data_summary(train_conf["architecture"]["data_summary"])
        
        self.n_features_summary = self.data_summary.get_n_features() 
        self.marginals_dict = train_conf["marginals"]
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.lr = train_conf["learning_rate"]
        self.bounds_trained = dataset_info["conf"]["prior"]
        self.scheduler_patience = train_conf["scheduler_patience"]
        self.scheduler_factor = train_conf["scheduler_factor"]
        self.output_names = []
        self.marginals_list = []
        for d_idx, domain in enumerate(self.marginals_dict):
            for i, marginal in enumerate(self.marginals_dict[domain]):
                # create a nice string for the marginal 
                name_output = ""
                for idx in marginal: 
                    name_output += str(_ORDERED_PRIOR_KEYS[idx]) + "_"
                name_output = name_output[:-1]  # remove trailing underscore
                self.output_names.append(name_output)
                self.marginals_list.append(marginal)
        self.td_normalisation = normalisation["td_normalisation"].item()
        self.param_mean = torch.Tensor(normalisation["param_mean"]).to(train_conf["device"])
        self.param_std = torch.Tensor(normalisation["param_std"]).to(train_conf["device"])
        print("Parameter mean:", self.param_mean.shape)
        print("Parameter std:", self.param_std.shape)
        self.save_hyperparameters(logger=True)    
        
        self.use_gradnorm = train_conf["architecture"]["gradnorm"]["enabled"]

        if self.use_gradnorm:
            self.gradnorm = GradnormHandler(self, train_conf["architecture"]["gradnorm"])
            self.weights_loss_logits = torch.nn.Parameter(torch.zeros(len(self.marginals_list)))
            self.lr_gradnorm = train_conf["architecture"]["gradnorm"]["learning_rate"]
       
        self.logratios_model_dict = nn.ModuleDict()
        for key in self.marginals_dict.keys():
            self.logratios_model_dict[key] = MarginalClassifierHead(
                n_data_features=self.n_features_summary*len(key),
                marginals=self.marginals_dict[key], 
                hlayersizes=(64, 32, 16, 8)
            ).to(train_conf["device"])


    def transform_td(self, data_t):
        """Applies time-domain normalisation to the input data.

        :param data_t: Tensor of shape (batch_size, num_channels, num_timebins) containing the time domain signal in the TDI channels
        :type data_t: torch.Tensor
        :return: Normalised time-domain data.
        :rtype: torch.Tensor
        """
        return data_t / self.td_normalisation
    
    def transform_log(self, complex_data):
        """Applies log transformation to the complex input data, with separate channels for amplitude and phase.
        """
        amplitude = torch.abs(complex_data)
        phase = torch.angle(complex_data)
        log_amplitude = torch.log(amplitude+1e-33)
        # cat 
        transformed_data = torch.cat((log_amplitude, phase), dim=1)
        return transformed_data 
    
    def forward(self, d_f, d_t, parameters):
        """
        Forward pass of the network.
        
        Args:
            d_f: complex Tensor of shape (batch_size, num_channels, num_freqbins) containing the frequency domain signal in the TDI channels.
            d_t: real Tensor of shape (batch_size, num_channels, num_timebins) containing the time domain signal in the TDI channels
            parameters: Tensor of shape (batch_size, 11) containing the parameters to be used in the classifier
        """
        # preprocessing 
        # here, d_f is a complex array of raw data (wave + noise), and d_t is real array of raw data (wave + noise)
        d_f, d_t = self.data_summary(d_f, d_t)

        features_ft = None
        
        features_dict = {
            "ft": features_ft,
            "f": d_f,
            "t": d_t
        }
        # print(f"features_t device: {features_t.device}")
        # print(f"parameters device: {parameters.device}")
        # print(f"features_f device: {features_f.device}")
        # print(f"features_ft device: {features_ft.device}")
        # #print(f"features_f shape: {features_f.shape}")
        #breakpoint()
        normalised_parameters = (parameters - self.param_mean) / self.param_std
        logratios_1d_list = []
        for key in self.logratios_model_dict.keys():
            #breakpoint()
            logratios_1d_list.append(self.logratios_model_dict[key](features_dict[key], normalised_parameters))
        logratios_1d = torch.cat(logratios_1d_list, dim=-1)        
        return logratios_1d

    def _calc_logits_base(self, batch):
        data_f = batch['wave_fd']+batch["noise_fd"]
        data_t = batch['wave_td']+batch["noise_td"]
        parameters = batch['source_parameters']
        all_data_f = torch.cat((data_f, data_f), dim=0)
        all_data_t = torch.cat((data_t, data_t), dim=0)
        scrambled_params = torch.roll(parameters, shifts=1, dims=0)
        all_params = torch.cat((parameters, scrambled_params), dim=0)
        all_logits = self(all_data_f, all_data_t, all_params)
        shape_half = (all_logits.shape[0] // 2, all_logits.shape[1])
        labels = torch.cat(
            (torch.ones(shape_half, device=self.device), torch.zeros(shape_half, device=self.device)),
            dim=0,
        )
        all_loss = self.loss(all_logits, labels)
        return all_logits, all_loss
    

    def calc_logits_losses(self, batch):
        all_logits, all_loss = self._calc_logits_base(batch)
        task_losses = torch.mean(all_loss, dim=0)
        loss = torch.mean(task_losses)
        return all_logits, task_losses, loss

    def calc_accuracies(self, all_logits):
        # the first half of all_params contains parameters from the joint, associated with the first half of all_data
        # the second half of all_params contains parameters from the marginal, independent of the second half of all_data
        shape_logits_half = (all_logits.shape[0] // 2, all_logits.shape[1])
        joint_preds = (all_logits[:shape_logits_half[0]] > 0).float()
        scrambled_preds = (all_logits[shape_logits_half[0]:] > 0).float()
        joint_accurate = (joint_preds == 1).float()
        scrambled_accurate = (scrambled_preds == 0).float()
        joint_accuracy_params = torch.mean(joint_accurate, dim=0)
        scrambled_accuracy_params = torch.mean(scrambled_accurate, dim=0)
        accuracy_params = (joint_accuracy_params + scrambled_accuracy_params) / 2
        accuracy = torch.mean(accuracy_params)

        return accuracy_params, accuracy

        
    def training_step(self, batch, batch_idx):
        # these are complex representations
        if self.use_gradnorm:
            all_logits, task_losses, loss = self.gradnorm.training_step(batch, batch_idx)
        else:
            all_logits, task_losses, loss = self.calc_logits_losses(batch)

        accuracy_params, accuracy = self.calc_accuracies(all_logits)

        self.log('train_loss', loss , on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        overall_idx = 0
        for d_idx, domain in enumerate(self.marginals_dict.keys()):
            for i, marginal in enumerate(self.marginals_dict[domain]):
                name = self.output_names[overall_idx]
                self.log(f'train_accuracy_{name}', accuracy_params[i], on_step=True, on_epoch=True, prog_bar=True, logger=True)
                self.log(f'train_loss_{name}', task_losses[i], on_step=True, on_epoch=True, prog_bar=True, logger=True)
                overall_idx += 1

        if not self.use_gradnorm:
            return loss
    
    def validation_step(self, batch, batch_idx):
        # if self.use_gradnorm:
        #     all_logits, task_losses, weighted_losses_tails, weights,  loss= self.gradnorm.calc_logits_losses_gradnorm(batch['data_fd'], batch["data_td"], batch['source_parameters'])
        # else:
        all_logits, task_losses, loss= self.calc_logits_losses(batch)
        accuracy_params, accuracy = self.calc_accuracies(all_logits)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        overall_idx = 0
        for d_idx, domain in enumerate(self.marginals_dict):
            for i, marginal in enumerate(self.marginals_dict[domain]):
                name = self.output_names[overall_idx]
                self.log(f'val_accuracy_{name}', accuracy_params[overall_idx], on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log(f'val_loss_{name}', task_losses[overall_idx], on_step=False, on_epoch=True, prog_bar=True, logger=True)
                overall_idx += 1
        return loss
    
    def test_step(self, batch, batch_idx):
        all_logits, task_losses, loss = self.calc_logits_losses(batch['data_fd'], batch['source_parameters'])
        accuracy_params, accuracy = self.calc_accuracies(all_logits)
        # print accuracies on test set
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        for d_idx, domain in enumerate(self.marginals_dict):
            for i, marginal in enumerate(self.marginals_dict[domain]):
                self.log(f'test_accuracy_{domain}_{i}', accuracy_params[i], on_step=False, on_epoch=True)

    def configure_optimizers(self):

        if self.use_gradnorm:
            trainable_params = [p for n, p in self.named_parameters() if n != "weights_loss_logits"]
            optimizer_nre = torch.optim.AdamW(trainable_params, lr=self.lr)
            optimizer_gradnorm = torch.optim.Adam([self.weights_loss_logits], lr=self.lr_gradnorm)
            return optimizer_nre, optimizer_gradnorm

        else: 
            optimizer_nre = torch.optim.AdamW(self.parameters(), lr=self.lr)
            return optimizer_nre
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.scheduler_factor, patience=self.scheduler_patience, min_lr=1e-6)
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 10**(epoch//15) if epoch < 30 else 100)
        #scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     optimizer ,
        #     base_lr=0.0001,
        #     max_lr=0.001,
        #     step_size_up=15,
        # )
        

class SimpleModel(torch.nn.Module):
    def __init__(self, num_features: int, num_channels: int,  hlayersizes: tuple,  marginals: dict[list[list]], marginal_hidden_size: int, lr: float):  
        super().__init__()

        # self.normalise = nn.BatchNorm1d(num_features=num_channels, eps=1e-22)
        # self.conv1d = nn.Conv1d(num_channels, 1, kernel_size=3, padding=1) 
        self.fc_blocks = nn.Sequential()
        input_size = num_features
        for i, output_size in enumerate(hlayersizes):
            self.fc_blocks.add_module(f"fc_{i}", nn.Linear(input_size, output_size))
            self.fc_blocks.add_module(f"relu_{i}", nn.ReLU())
            # self.fc_blocks.add_module(f"dropout_{i}", nn.Dropout(p=0.2))
            input_size = output_size
        
        self.logratios = MarginalClassifierHead(input_size, marginals=marginals, hidden_size=marginal_hidden_size)
        self.marginals_dict = marginals
        

    def forward(self, x, parameters):
        """
        Forward pass of the network.
        
        Args:
            data: Tensor of shape (batch_size, num_channels, num_features) containing the frequency domain signal in the TDI channels
            parameters: Tensor of shape (batch_size, 11) containing the parameters to be used in the classifier
        """
        # data = self.normalise(x)
        # #print("data min max std mean", torch.min(data), torch.max(data), torch.std(data), torch.mean(data))
        # data = self.conv1d(data)  
        # #print("data after conv1d min max std mean", torch.min(data), torch.max(data), torch.std(data), torch.mean(data))
        # data = data.squeeze(1)  # (batch_size, num_features - 2)
        features = self.fc_blocks(x)  # (batch_size, hidden_size)
        #print("features min max std mean", torch.min(features), torch.max(features), torch.std(features), torch.mean(features))
        output = self.logratios(features, parameters)  # (batch_size, num_marginals)
        
        #print("output min max std mean", torch.min(output), torch.max(output), torch.std(output), torch.mean(output))
        return output


class PeregrineModel(torch.nn.Module):
    def __init__(self, n_channels: int , n_timesteps: int, n_freqs: int):
        super().__init__()

        
        
        #self.normalisation_f = nn.BatchNorm1d(num_features=n_channels*2)
        #self.normalisation_t = nn.BatchNorm1d(num_features=n_channels)
        self.unet_f = Unet(
            n_in_channels=n_channels*2,
            n_out_channels=1,
            sizes=(16, 32, 64, 128, 256),
            down_sampling=(2, 2, 2, 2),
        )
        self.unet_t = Unet(
            n_in_channels=n_channels,
            n_out_channels=1,
            sizes=(16, 32, 64, 128, 256),
            down_sampling=(8,8,8,8),
        )

        self.flatten = nn.Flatten(1)
        self.linear_t = LinearCompression(n_timesteps)
        self.linear_f = LinearCompression(n_freqs)






    def forward(self, d_f, d_t):
        #print(f"d_f shape: {d_f.shape}")
        #d_f_norm = self.normalisation_f(d_f)
        #d_t_norm = self.normalisation_t(d_t)
        #print(f"d_f_norm shape: {d_f_norm.shape}")
        d_f_processed = self.unet_f(d_f)
        d_t_processed = self.unet_t(d_t)
        #print(f"d_f_processed shape: {d_f_processed.shape}")
        flattened_f = self.flatten(d_f_processed)
        flattened_t = self.flatten(d_t_processed)

        #print(f"flattened shape: {flattened.shape}")
        datasummary_fd = self.linear_f(flattened_f)
        datasummary_td = self.linear_t(flattened_t)

        return datasummary_fd, datasummary_td

class ChannelizedLinearCompression(nn.Module): 
    """a linear compression that operates along dimension 1 (channels) independently, and produces, one number for each channel, then concatenates them and passes through a final MLP to get lowdim features.
    """
    def __init__(self, lowdim: int, Nfreqs: int,  in_channels: int  ): 
        """Initializes the ChannelizedLinearCompression module.

        :param lowdim: the output features will be of dimension lowdim
        :type lowdim: int
        :param Nfreqs: the number of frequency bins
        :type Nfreqs: int
        :param in_channels: the number of input channels (note: each TDI channel corresponds to TWO channels in this module, because amplitude and phase are treated separately)
        :type in_channels: int
        """
        super().__init__()
        self.N = Nfreqs
        self.lowdim = lowdim
        self.in_channels = in_channels

        self.channel_blocks = nn.ModuleList()
        int_dimension = (self.N*10)**(0.5)
        for _ in range(self.in_channels):
            block = nn.Sequential(
            nn.Flatten(),  # (B, 1, N) -> (B, N)
            nn.Linear(self.N, int(int_dimension)),
            nn.ReLU(),
            nn.Linear(int(int_dimension), 10),
            nn.ReLU(),
            nn.Linear(10, 1)  # produce a single number per channel
            )
            self.channel_blocks.append(block)

        # final 2-layer MLP: takes concatenated per-channel scalars (B, in_channels) -> lowdim
        final_hidden = 30
        self.final_mlp = nn.Sequential(
            nn.Linear(self.in_channels, final_hidden),
            nn.ReLU(),
            nn.Linear(final_hidden, self.lowdim),
        )
    
    def forward(self, x):
        # x shape: (B, C, N)
        B, C, N = x.shape

        # process each channel with its channel_block -> yields (B,1) per channel
        channel_outputs = []
        for c, block in enumerate(self.channel_blocks):
            out = block(x[:, c, :])  # (B, 1)
            channel_outputs.append(out)

        # concatenate per-channel scalars into (B, C)
        per_channel_scalars = torch.cat(channel_outputs, dim=1)  # (B, C)

        # final MLP to produce lowdim features (B, lowdim)
        compressed = self.final_mlp(per_channel_scalars)  # (B, lowdim)

        return compressed


class BrutalCompression(nn.Module):
    """
    Take top-100 values per channel from d_f and corresponding indices normalized to [0,1].
    Return tensor of shape (B, Nchannels*2, 100) where first Nchannels are values and
    next Nchannels are normalized indices.

    This implementation requires Nfreqs >= 100 and will raise a ValueError otherwise.
    """
    def __init__(self, lowdim: int, Nfreqs: int,  in_channels: int  ): 
        super().__init__()
        self.N = 100  # number of top frequencies to select
        self.lowdim = lowdim
        self.in_channels = in_channels
        self.network = ChannelizedLinearCompression(lowdim, self.N, in_channels)


    def forward(self, d_f, d_t):
        """
        Args:
            d_f: Tensor shape (B, C, Ntot_freqs)
            d_t: passed through unchanged
        Returns:
            compressed: Tensor shape (B, 1, lowdim)
            d_t: unchanged
        """
        B, C, Nfreqs = d_f.shape

        if Nfreqs < self.N:
            raise ValueError(f"BrutalCompression requires at least {self.N} frequencies, got {Nfreqs}")

        k = self.N
        # compute top-k indices from channel 0 only
        _, topk_idx = torch.topk(d_f[:, 0, :].abs(), k=k, dim=1)  # (B, k)
        # expand to all channels for gathering
        indices = topk_idx.unsqueeze(1).expand(-1, C, -1)  # (B, C, k)

        # gather values for all channels at these indices
        selected = torch.gather(d_f, 2, indices)  # (B, C, k)

        compressed = self.network(selected)  # (B, lowdim)

        return compressed, d_t
   
### THIS CHUNK OF CODE IS DIRECTLY COPIED FROM PEREGRINE
# 1D Unet implementation below
class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        mid_channels=None,
        padding=1,
        bias=False,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, down_sampling=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(down_sampling), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_channels, in_channels // 2, kernel_size=kernel_size, stride=stride
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_signal_length = x2.size()[2] - x1.size()[2]

        x1 = F.pad(
            x1, [diff_signal_length // 2, diff_signal_length - diff_signal_length // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(
        self,
        n_in_channels,
        n_out_channels,
        sizes=(16, 32, 64, 128),
        down_sampling=(2, 2, 2, 2),
    ):
        super(Unet, self).__init__()
        self.inc = DoubleConv(n_in_channels, sizes[0])
        self.down1 = Down(sizes[0], sizes[1], down_sampling[0])
        self.down2 = Down(sizes[1], sizes[2], down_sampling[1])
        self.down3 = Down(sizes[2], sizes[3], down_sampling[2])
        self.down4 = Down(sizes[3], sizes[4], down_sampling[3])
        self.up1 = Up(sizes[4], sizes[3])
        self.up2 = Up(sizes[3], sizes[2])
        self.up3 = Up(sizes[2], sizes[1])
        self.up4 = Up(sizes[1], sizes[0])
        self.outc = OutConv(sizes[0], n_out_channels)

    def forward(self, x):
        raise NotImplementedError("Unet forward pass is not implemented. The data should be transformed before passing to the Unet.")
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        f = self.outc(x)
        return f


class LinearCompression(nn.Module):
    def __init__(self, N_input):
        super(LinearCompression, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(N_input, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )

    def forward(self, x):
        return self.sequential(x)
    


class ReducedOrderModel:
    def __init__(self, batch_size=1000, tolerance=1e-3, device="cpu", filename=None, debugging=False):
        if filename is not None: 
            print("[ROM] initializing from file {filename}.\n Other arguments will be ignored.")
            self.device = device
            self.from_file(filename)
            
        else: 
            self.debugging = debugging
            self.basis = None
            self.n_channels = None
            self.n_freq = None
            self.batch_size = batch_size
            self.device = device
            self.tolerance = tolerance
            if debugging:
                self.plot_dir_debug = os.path.join(ROOT_DIR, "plots", "debug_plots_{time}".format(time=time.strftime("%Y%m%d-%H%M%S"))) 
                os.makedirs(self.plot_dir_debug, exist_ok=True)
    
    def from_file(self, filename):
        s = torch.load(filename, map_location=self.device)
        self.n_channels = s["n_channels"]
        self.n_freq = s["n_freq"]
        self.basis = s["basis"].to(self.device)
        self.global_scale_factor = s["global_scale_factor"]
        self.mean_vec = s["mean_vec"].to(self.device)
        print("[ROM] basis loaded.\nn_freq =", self.n_freq, ", n_channels =", self.n_channels, ", n_basis =", len(self.basis))
        return 

    def load_diagnostics(self, filename):
        self.training_diagnostics = torch.load(filename, map_location=self.device)
        return

    def save_diagnostics(self, filename):
        torch.save(self.training_diagnostics, filename)
        print(f"[ROM] diagnostics saved to {filename}")
        return 

    def to_file(self, filename):
        torch.save(
            {
                "basis": self.basis.cpu(),
                "tolerance": self.tolerance,
                "n_channels": self.n_channels,
                "n_freq": self.n_freq,
                "device": self.device,
                "global_scale_factor": self.global_scale_factor,
                "mean_vec": self.mean_vec
            },
            filename,
        )
        filename_diagnostics = filename.replace(".pt", "_diagnostics.pt")
        self.save_diagnostics(filename_diagnostics)
        print(f"[ROM] basis saved to {filename}")
        return

    @profile
    def train(self, train_dataloader: torch.utils.data.DataLoader):
        """Train the reduced order model

        Args:
            train_dataloader (torch.utils.data.DataLoader): the training dataloader, that has the training Subset as attribute
        """
        print(f"[ROM] training with tolerance={self.tolerance:.1e} on device={self.device}")
        train_subset = train_dataloader.dataset
        if not isinstance(train_dataloader.dataset, torch.utils.data.Subset):
            raise ValueError("[ROM] train_dataloader.dataset must be a torch.utils.data.Subset")
        train_subset.dataset.to(self.device) # move the (full) dataset to the right device.
        self.n_freq = train_subset[0]['wave_fd'].shape[1]
        self.n_channels = train_subset[0]['wave_fd'].shape[0]
        self.global_scale_factor, self.mean_vec = self._compute_global_scale(train_dataloader)
        self.mean_vec = self.mean_vec.to(self.device)
        print(f"[ROM] global scale factor: {self.global_scale_factor:.3e}")
        
        # raw = dataset.wave_fd # (N_pts, C, F)
        # self.data_f = raw.reshape(raw.shape[0], -1)
        # self.n_channels = raw.shape[1]
        # self.n_freq = raw.shape[2]
        # n = raw.shape[0]

        n = len(train_subset)
        seed = torch.randint(0, n, (1,)).item()
                

        first = self._normalize(train_subset[seed]['wave_fd'].unsqueeze(0).to(self.device)).squeeze(0)
        first = first / first.norm()
        self.basis = first.unsqueeze(0)
        self.basis_indices = [seed]


        sigma = float("inf")
        
        self.epoch = 0
        t0 = time.time()
        pbar = tqdm(total=0, bar_format='{desc}{postfix}', position=0, leave=True)  # dynamic manual updates
        
        log10mc_values = []
        q_values = []
        sigmas = []
        sigmas_unnorm = []
        sigmas_data = []
        picked = [seed]
        picked_set = {seed}
        self.gs_diagnose =  {
            "norm_v_original": [],
            "norm_v_trunc": [],
            "norm_projection": [],
            "max_cosine_angle_unnormalised": [],
            "max_cosine_angle_normalised":[]
        }
        self.training_diagnostics = {
            "log10mc_values": log10mc_values,
            "q_values": q_values,
            "sigmas": sigmas,
            "sigmas_unnorm": sigmas_unnorm,
            "sigmas_data": sigmas_data,
            "max_pointwise_relerrors_real": [],
            "max_pointwise_relerrors_imag": [],
            "picked_indices": picked,
            "gs_diagnose": self.gs_diagnose
        }

        try:
            sigma_last = torch.tensor(float("inf"), device=self.device)
        
            while sigma > self.tolerance:
                self.epoch += 1 # epoch 1 --> N_basis = 1
                sigma, idx , sigma_unnorm, sigma_data, idx_data = self._max_residual_index(train_dataloader, picked_set)
                if sigma_data >= sigma_last_data: 
                    print(f"\ntraining stopped early at iteration {self.epoch} because sigma did not decrease (sigma={sigma_data:.3e}, last sigma={sigma_last_data:.3e})")
                    break

                sigma_last_data = sigma_data
                picked_set.add(idx)
                picked.append(idx)
                log10mc_values.append(train_subset[idx]["params"][0])
                q_values.append(train_subset[idx]["params"][1])
                picked.append(idx)
                sigmas.append(sigma)
                sigmas_unnorm.append(sigma_unnorm)
                sigmas_data.append(sigma_data)
                v = self._normalize(train_subset[idx]['wave_fd'].unsqueeze(0)).squeeze(0)
                v = self._gram_schmidt_reorthogonalize(v)
                self.basis = torch.cat([self.basis, v.unsqueeze(0)], dim=0)
                elapsed = time.time() - t0
                pbar.set_postfix({f"N_basis_elems":self.basis.shape[0], "iter":self.epoch, "sigma":f"{sigma:.3e}", "elapsed":f"{elapsed:.1f}s", "rate":f"{self.epoch/elapsed:.1f} it/s"})
                pbar.update(0)  # just refresh the display
        except KeyboardInterrupt:
            print("\n[ROM] training interrupted by user.")
        total = time.time() - t0
        print(f"[ROM] done. basis={len(self.basis)}, time={total:.1f}s")

        
    @profile
    def _project_batch(self, batch):
        """Project the input batch onto the ROM basis.

        :param batch: Input batch of normalized waveforms, with shape (B_size, N_dim). 
        :type batch: torch.Tensor
        :return: Projected batch in normalized space.
        :rtype: torch.Tensor
        """
        # batch is already centered and scaled, so we work in normalized space
        # This is equivalent to compress-then-reconstruct without denormalization
        coeff = self._compress_normalized(batch)
        proj = self._reconstruct_normalized(coeff)
        return proj

    @profile
    def _max_residual_index(self, train_dataloader, picked_set):
        max_seen = -1.0
        max_seen_unnorm = -1.0
        max_seen_data = -1.0
        max_index = -1
        max_index = None
        seen_points = 0
        max_relerr_re = 0
        max_relerr_im = 0
        for idx, batch in enumerate(train_dataloader):
            bsize = batch['wave_fd'].shape[0]
            
            original_wave_batch = batch['wave_fd'].reshape(bsize, -1).to(self.device)
            # centered and scaled waveforms
            wave_fd_batch = self._normalize(batch['wave_fd'].to(self.device))  # shape (B_size, N_dim)
            # project onto basis (aka : compute coefficients and then reconstruct)
            proj = self._project_batch(wave_fd_batch)
            # back to original space
            reconstruction_original_space = self._denormalize(proj)
            residual_original_space = original_wave_batch - reconstruction_original_space

            # compute residuals in normalized space
            r = (wave_fd_batch - proj) / wave_fd_batch.norm(dim=1, keepdim=True)
            # compute max pointwise relative error in this batch
            scale_re  = wave_fd_batch.real.abs().amax(dim=1, keepdim=True)
            scale_im = wave_fd_batch.imag.abs().amax(dim=1, keepdim=True)
            # breakpoint()
            max_rel_error_real_batch = torch.amax((wave_fd_batch.real - proj.real).abs()/scale_re ) 
            max_rel_error_imag_batch = torch.amax((wave_fd_batch.imag - proj.imag).abs()/scale_im )
            max_relerr_re = max(max_relerr_re, max_rel_error_real_batch.item())
            max_relerr_im = max(max_relerr_im, max_rel_error_imag_batch.item())
            r_unnorm = wave_fd_batch - proj
            norms = (r.abs() ** 2).sum(dim=1)
            norms_unnorm = (r_unnorm.abs() ** 2).sum(dim=1)


            original_data_batch = (batch['wave_fd'] + batch['noise_fd']).reshape(bsize, -1).to(self.device)
            normalised_data_batch = self._normalize(original_data_batch)
            data_proj = self._project_batch(normalised_data_batch)
            #data_reconstruction_original_space = self._denormalize(data_proj)
            norms_residual_data = ((wave_fd_batch - data_proj).abs() ** 2).sum(dim=1)
            if self.debugging and idx == 0 and self.epoch <= 10:
                # plot the residuals as function of frequency
                for jj in range(min(3, bsize)):
                    plt.figure(figsize=(12,6))
                    plt.subplot(1,2,1)
                    plt.plot(wave_fd_batch[jj][:self.n_freq].abs().cpu().numpy(), label="original")
                    plt.plot(proj[jj][:self.n_freq].abs().cpu().numpy(), label="projected", linestyle="--")
                    plt.plot(r[jj][:self.n_freq].abs().cpu().numpy(), label=f"residual")
                    params = batch['params'][jj][:2].cpu().numpy()
                    plt.suptitle(f"log10mc: {params[0]:.8e} q: {params[1]:.8e} sum of abs square residual : {norms[jj].item():.3e}")
                    plt.title("channel 1")
                    # plt.plot(r[0][self.n_freq:], label=f"channel 2")
                    plt.xlabel("Frequency index")
                    plt.xscale("log")
                    plt.yscale("log")
                    plt.title("channel 1")
                    plt.legend()

                    plt.subplot(1,2,2)
                    plt.plot(wave_fd_batch[jj][self.n_freq:].abs().cpu().numpy(), label="original")
                    plt.plot(proj[jj][self.n_freq:].abs().cpu().numpy(), label="projected", linestyle="--")
                    plt.plot(r[jj][self.n_freq:].abs().cpu().numpy(), label=f"residual")

                    plt.title("channel 2")
                    plt.xlabel("Frequency index")
                    plt.xscale("log")
                    plt.yscale("log")
                    plt.legend()
                    plt.savefig(os.path.join(self.plot_dir_debug, f"residuals_epoch{self.epoch}_batch{idx}_event{jj}.png"))

                    plt.close()

                for jj in range(min(3, bsize)):
                    plt.figure(figsize=(12,6))
                    
                    plt.subplot(1,2,1)
                    plt.plot(reconstruction_original_space[jj][:self.n_freq].abs().cpu().numpy(), label="reconstruction")
                    plt.plot(original_wave_batch[jj][:self.n_freq].reshape(-1).abs().cpu().numpy(), label="original", linestyle="--")
                    plt.plot(residual_original_space[jj][:self.n_freq].abs().cpu().numpy(), label="residuals")
                    plt.xlabel("Frequency index")
                    plt.xscale("log")
                    plt.yscale("log")
                    plt.suptitle("Original space reconstruction and residuals")
                    plt.legend()
                    plt.title("Channel 1")

                    plt.subplot(1,2,2)
                    plt.plot(reconstruction_original_space[jj][self.n_freq:].abs().cpu().numpy(), label="reconstruction")
                    plt.plot(original_wave_batch[jj][self.n_freq:].reshape(-1).abs().cpu().numpy(), label="original", linestyle="--")
                    plt.plot(residual_original_space[jj][self.n_freq:].abs().cpu().numpy(), label="residuals")
                    plt.xlabel("Frequency index")
                    plt.xscale("log")
                    plt.yscale("log")
                    plt.title("Channel 2")
                    plt.legend()
                    plt.savefig(os.path.join(self.plot_dir_debug, f"{self.epoch}_batch{idx}_event{jj}.png"))
                    plt.close()
            # mask picked dataset indices
            batch_ids = torch.arange(bsize) + seen_points
            mask = torch.tensor(
                [(idx.item() not in picked_set) for idx in batch_ids],
                device=norms.device,
                dtype=torch.bool,
            ) # indicates which points are NOT picked yet

            if not mask.any():
                seen_points += bsize
                continue 
            masked_norms = norms.clone()
            # set already picked indices to -1 so they are ignored in max search
            masked_norms[~mask] = -1.0  # safe since norms >= 0

            current_max, current_idx = masked_norms.max(0)
            current_max_data , current_idx_data = norms_residual_data.max(0)
            if current_max.item() > max_seen:
                max_seen = current_max.item()
                max_index = (current_idx + seen_points).item()
                max_seen_unnorm = norms_unnorm[current_idx].item()

            if current_max_data.item() > max_seen_data:
                max_seen_data = current_max_data.item()
                max_index_data = (current_idx_data + seen_points).item()
            seen_points += bsize

        self.training_diagnostics["max_pointwise_relerrors_real"].append(max_relerr_re)
        self.training_diagnostics["max_pointwise_relerrors_imag"].append(max_relerr_im)
        return max_seen, max_index, max_seen_unnorm, max_seen_data, max_index_data
    
    def _compute_global_scale(self, dataloader):
        max_val = 0.0
        mean_vec = torch.zeros(self.n_channels * self.n_freq, device=self.device)
        for batch in dataloader:
            v = batch['wave_fd'].to(self.device)
            max_val = max(max_val, v.abs().max().item())
            mean_vec = mean_vec + v.reshape(v.shape[0], -1).sum(dim=0)
        mean_vec = mean_vec / len(dataloader.dataset)
        return max_val, mean_vec
    

    def _gram_schmidt_reorthogonalize(self, v): 
        """the gram-schmidt algorithm with reorthogonalization step as described in https://drum.lib.umd.edu/items/fe4fed4a-c5a2-49fb-844b-05457b532a89 
        """
        B = self.basis
        coeff = B.conj() @ v # shape (N_basis, N_dim) @ (N_dim, )  #/ torch.einsum("ik,ik->i", B.conj(), B)
        projection = coeff @ B # shape (N_basis) @ (N_basis, N_dim) = > (N_dim, )
        v_trunc = v - projection# now v is orthogonal to the basis

        # reorthogonalisation step 
        coeff2 = B.conj() @ v_trunc
        projection2 = coeff2 @ B
        v_trunc = v_trunc - projection2
        normalised = v_trunc / v_trunc.norm()# normalise before adding to the reduced basis. 
        scalar_products = B.conj() @ v_trunc  # shape (N_basis, )
        cosines_vtrunc = scalar_products / (B.norm(dim=1) * v_trunc.norm())
        scalar_products_v_normalised = B.conj() @ normalised
        cosines_vnormalised = scalar_products_v_normalised / (B.norm(dim=1) * normalised.norm())
        self.gs_diagnose["max_cosine_angle_unnormalised"].append(cosines_vtrunc.abs().max().item())
        self.gs_diagnose["max_cosine_angle_normalised"].append(cosines_vnormalised.abs().max().item())
        max_abs_scalar_product = scalar_products.abs().max().item()
        # compute max abs of removed part 

        norm_projection = projection.norm()
        norm_v_trunc = v_trunc.norm()
        norm_v_original = v.norm()
        self.gs_diagnose["norm_v_original"].append(norm_v_original.item())
        self.gs_diagnose["norm_projection"].append(norm_projection.item())
        self.gs_diagnose["norm_v_trunc"].append(norm_v_trunc.item())

        # triangular inequality checks
        assert norm_v_original <= norm_v_trunc + norm_projection, "Triangular inequality violated"
        assert norm_v_trunc <= norm_v_original + norm_projection, "Triangular inequality violated"
        assert norm_projection <= norm_v_original + norm_v_trunc, "Triangular inequality violated"
        ### debug line:  check orthogonality
        # try:
        #     assert torch.allclose((B.conj() @ v), torch.zeros_like(coeff), atol=1e-6), "Gram-Schmidt failed to produce orthogonal vector."
        # except AssertionError as e:
        #     print(e)
        #     print("if divide coefficients by norm of basis elements:")
        #     coeff_normed = coeff / torch.einsum("ik,ik->i", B.conj(), B)
        #     new_v = v - coeff_normed @ B
        #     assert torch.allclose((B.conj() @ new_v), torch.zeros_like(coeff), atol=1e-6), "Gram-Schmidt failed to produce orthogonal vector even after normalizing coefficients by basis element norms."
        
        return normalised

    def _modified_gram_schmidt(self, v) : 
        B = self.basis
        v_trunc = v.clone()
        for i in range(B.shape[0]):
            bi = B[i]
            coeff = (bi.conj() @ v_trunc) / (bi.conj() @ bi)
            v_trunc = v_trunc - coeff * bi
        normalised = v_trunc / v_trunc.norm()

        scalar_products = B.conj() @ v_trunc  # shape (N_basis, )
        cosines_vtrunc = scalar_products / (B.norm(dim=1) * v_trunc.norm())
        scalar_products_v_normalised = B.conj() @ normalised
        cosines_vnormalised = scalar_products_v_normalised / (B.norm(dim=1) * normalised.norm())
        self.gs_diagnose["max_cosine_angle_unnormalised"].append(cosines_vtrunc.abs().max().item())
        self.gs_diagnose["max_cosine_angle_normalised"].append(cosines_vnormalised.abs().max().item())
        #max_abs_scalar_product = scalar_products.abs().max().item()

        self.gs_diagnose["norm_v_original"].append(v.norm().item())
        self.gs_diagnose["norm_v_trunc"].append(v_trunc.norm().item())
        self.gs_diagnose["norm_projection"].append( (v - v_trunc).norm().item())

        return normalised

    @profile
    def _gram_schmidt(self, v):
        B = self.basis
        coeff = B.conj() @ v # shape (N_basis, N_dim) @ (N_dim, )  #/ torch.einsum("ik,ik->i", B.conj(), B)
        projection = coeff @ B # shape (N_basis) @ (N_basis, N_dim) = > (N_dim, )
        v_trunc = v - projection# now v is orthogonal to the basis
        normalised = v_trunc / v_trunc.norm()# normalise before adding to the reduced basis. 
        scalar_products = B.conj() @ v_trunc  # shape (N_basis, )
        cosines_vtrunc = scalar_products / (B.norm(dim=1) * v_trunc.norm())
        scalar_products_v_normalised = B.conj() @ normalised
        cosines_vnormalised = scalar_products_v_normalised / (B.norm(dim=1) * normalised.norm())
        self.gs_diagnose["max_cosine_angle_unnormalised"].append(cosines_vtrunc.abs().max().item())
        self.gs_diagnose["max_cosine_angle_normalised"].append(cosines_vnormalised.abs().max().item())
        max_abs_scalar_product = scalar_products.abs().max().item()

        if self.debugging and self.epoch <= 10: 
            # check orthogonality between v_trunc and basis B

            print(f"[ROM][debug] max abs scalar product between v_trunc and basis elements: {max_abs_scalar_product:.3e}")
            # plot the difference between cosines and cosines after normalisation (should be zero)
            cosine_residual = (cosines_vtrunc - cosines_vnormalised).abs()
            max_cosine_residual = cosine_residual.max().item()
            print(f"[ROM][debug] max abs difference between cosines before and after normalisation: {max_cosine_residual:.3e}")
            plt.plot(cosine_residual.cpu().numpy())
            plt.yscale("log")
            plt.xlabel("Basis element index")
            plt.title("Difference between cosines before and after normalisation")
            plt.savefig(os.path.join(self.plot_dir_debug, f"cosine_residual_epoch{self.epoch}.png"))
            plt.close()

        # compute max abs of removed part 

        norm_projection = projection.norm()
        norm_v_trunc = v_trunc.norm()
        norm_v_original = v.norm()
        self.gs_diagnose["norm_v_original"].append(norm_v_original.item())
        self.gs_diagnose["norm_projection"].append(norm_projection.item())
        self.gs_diagnose["norm_v_trunc"].append(norm_v_trunc.item())

        # triangular inequality checks
        assert norm_v_original <= norm_v_trunc + norm_projection, "Triangular inequality violated"
        assert norm_v_trunc <= norm_v_original + norm_projection, "Triangular inequality violated"
        assert norm_projection <= norm_v_original + norm_v_trunc, "Triangular inequality violated"
        ### debug line:  check orthogonality
        # try:
        #     assert torch.allclose((B.conj() @ v), torch.zeros_like(coeff), atol=1e-6), "Gram-Schmidt failed to produce orthogonal vector."
        # except AssertionError as e:
        #     print(e)
        #     print("if divide coefficients by norm of basis elements:")
        #     coeff_normed = coeff / torch.einsum("ik,ik->i", B.conj(), B)
        #     new_v = v - coeff_normed @ B
        #     assert torch.allclose((B.conj() @ new_v), torch.zeros_like(coeff), atol=1e-6), "Gram-Schmidt failed to produce orthogonal vector even after normalizing coefficients by basis element norms."
        
        return normalised

    def _normalize(self, data):
        """Center and scale data to normalized space.
        
        :param data: Input data with shape (Batch_size, Channels, Physics_dimension) or (Batch_size, N_dim).
        :type data: torch.Tensor
        :return: Normalized data with shape (Batch_size, N_dim).
        :rtype: torch.Tensor
        """
        # Flatten if needed
        if data.ndim == 3:
            d = data.reshape(data.shape[0], -1)
        else:
            d = data
        # Center and scale
        return (d - self.mean_vec) / self.global_scale_factor
    
    def _denormalize(self, data):
        """Unscale and uncenter data from normalized space.
        
        :param data: Normalized data with shape (Batch_size, N_dim).
        :type data: torch.Tensor
        :return: Denormalized data with shape (Batch_size, N_dim).
        :rtype: torch.Tensor
        """
        return data * self.global_scale_factor + self.mean_vec
    
    def _compress_normalized(self, normalized_data):
        """Compress normalized data into reduced basis coefficients.
        
        :param normalized_data: Normalized input data with shape (Batch_size, N_dim).
        :type normalized_data: torch.Tensor
        :return: Compressed representation of shape (Batch_size, N_basis).
        :rtype: torch.Tensor
        """
        B = self.basis
        coeff = (B.conj() @ normalized_data.T).T  # shape (B_size, N_basis)
        return coeff
    
    def _reconstruct_normalized(self, coeff):
        """Reconstruct normalized data from reduced basis coefficients.
        
        :param coeff: Input coefficients with shape (Batch_size, N_basis).
        :type coeff: torch.Tensor
        :return: Reconstructed normalized data with shape (Batch_size, N_dim).
        :rtype: torch.Tensor
        """
        B = self.basis
        return coeff @ B  # shape (B_size, N_dim)

    # public API with flatten → project → unflatten
    def compress(self, data):
        """compress data into reduced basis coefficients.

        :param data: Input data to compress with shape (Batch_size, Channels, Physics_dimension).
        :type data: torch.Tensor
        :return: Compressed representation of shape (Batch_size, N_basis).
        :rtype: torch.Tensor
        """
        normalized_data = self._normalize(data)
        return self._compress_normalized(normalized_data)

    def reconstruct(self, coeff):
        """reconstruct data from reduced basis coefficients.
        
        :param coeff: Input coefficients with shape (Batch_size, N_basis).
        :type coeff: torch.Tensor
        :return: Reconstructed data with shape (Batch_size, Channels, Physics_dimension).
        :rtype: torch.Tensor
        
        """
        normalized_data = self._reconstruct_normalized(coeff)
        denormalized_data = self._denormalize(normalized_data)
        return denormalized_data.reshape(denormalized_data.shape[0], self.n_channels, self.n_freq)
    

class ROMWrapper(nn.Module): 
    """A wrapper nn.Module around the ReducedOrderModel to be used as a data summarizer in the InferenceNetwork. Couples the ROM with a ChannelizedLinearCompression to produce lowdim features.
    """
    def __init__(self, filename: str, device: str = "cuda"):

        super().__init__()
        self.rom = ReducedOrderModel(filename=filename, device=device)
        # self.channelised_net = ChannelizedLinearCompression(
        #     lowdim=16, 
        #     Nfreqs=len(self.rom.basis), 
        #     in_channels=2# amplitude and phase
        # )
    def get_n_features(self):
        return 2*self.rom.basis.shape[0]
    
    def forward(self, d_f, d_t): 
        """forward pass of the ROMWrapper.

        :param d_f: raw data in the frequency domain , complex tensor of shape (B, C, F)
        :type d_f: torch.Tensor
        :param d_t: raw data in the time domain , real tensor of shape (B, C, T)
        :type d_t: torch.Tensor
        :return: compressed representation and time domain data
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """

        compressed_df = self.rom.compress(d_f)
        # compressed_ampl = torch.abs(compressed_df)
        # compressed_phase = torch.angle(compressed_df)
        # compressed_amplphase = torch.cat([compressed_ampl, compressed_phase], dim=1)  # shape (B, 2, N_basis)
        compressed_real = compressed_df.real
        compressed_imag = compressed_df.imag
        compressed_reim = torch.cat([compressed_real, compressed_imag], dim=1)  # shape (B, 2, N_basis)
        #nn_compression = self.channelised_net(compressed_amplphase) 
        return compressed_reim, d_t

DATA_SUMMARY_REGISTRY = { "BrutalCompression": BrutalCompression, "PeregrineModel": PeregrineModel, "ROM": ROMWrapper}