from datetime import datetime 
import os
import torch.nn as nn
import torch
from classes.epoch_results import EpochResult
class ProjectModelResnetConfig():
    
    def __init__(self, model_depth:int =10, no_cuda: bool= True, n_seg_classes:int =4, batch_size: int = 2, num_workers: int = 1):
        self.is_debug = True
        self.model_name = "resnet"
        # Option for model depth [10, 18, 34, 50, 101, 152, 200]
        self.model_depth = model_depth
        self.input_W = 232
        self.input_H = 232
        self.input_D = 128
        self.no_cuda = no_cuda
        self.max_epoch = 50
        self.batch_size = 1
        self.resnet_shortcut = 'B'
        self.n_seg_classes = n_seg_classes
        self.num_workers = 1
        self.pretrain_path = None
        self.phase_is_train = True
            
        self.pin_memory = None
        if not self.no_cuda:
            self.pin_memory = True
        self.model_save_path = "./training_checkpoints/"
        self.model_save_filename = "Model_{}_{}_epoch{}{}.pth.tar"
        self.nn_model = None
    
    def save_checkpoint_pathname(self, epoch:int, with_Datetime: False) -> str:
        datetime_str = ''
        if with_Datetime:
            datetime_str = "_" + datetime.now().strftime("%Y%m%d%H%M%S")
        
        filename = self.model_save_filename.format(self.model_name, self.model_depth, epoch, datetime_str)
        return os.path.join(self.model_save_path, filename)
    
    def set_net_model(self, _nn_model: nn.Module):
        self.nn_model = _nn_model
        # proj_resnet_model, pro_resnet_params = resnet_model_generator.generate_model(self)
        
    def load_med3d_pretrain_weigth(self, med3d_resnet_pth_path: str):
        net_dict = None
        pretrain_dict = {}
        if self.nn_model:
            net_dict = self.nn_model.state_dict()
        else:
            raise Exception("neural network model is not set")
        
        checkpoint = None
        if torch.cuda.is_available():
            checkpoint = torch.load(med3d_resnet_pth_path)
        else:
            checkpoint = torch.load(med3d_resnet_pth_path, map_location=torch.device('cpu'))
        
        for k,v in checkpoint['state_dict'].items():
            has_match = False
            for model_key in net_dict.keys():
                # the key has characteres 'module.' that causes layers arent matching, so these characters have to be stripped
                c_key = k[7:]
                if c_key == model_key or k == model_key:
                    pretrain_dict[model_key] = v
                    # print("Checkpoint Key: {:60s} model key: {:60s}  {}  {}".format(k, model_key, v.size(), type(v)))
                    has_match = True
                    break
            if not has_match:
                raise Exception("Checkpoint Key: {:60s} has no match, check model depth and load path".format(k))
        net_dict.update(pretrain_dict)
        self.nn_model.load_state_dict(net_dict)
    
    def load_weight_from_epoch(self, model_epoch_pth: str, is_colab=False):
        if not self.nn_model:
            raise Exception("neural network model is not set")
        print(model_epoch_pth)
        
        checkpoint = None
        if torch.cuda.is_available():
            checkpoint = torch.load(model_epoch_pth)
        else:
            checkpoint = torch.load(model_epoch_pth, map_location=torch.device('cpu'))
        self.nn_model.load_state_dict(checkpoint['state_dict'])
        ep_list = checkpoint['epoch_list']
        loss_list = checkpoint['loss_list']
        lr_list = checkpoint['lr_list']
        epoch_res = EpochResult(_epoch_list =ep_list, _loss_list=loss_list, _lr_list=lr_list)
        return checkpoint, epoch_res