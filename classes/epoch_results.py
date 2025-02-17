class EpochResult():
    def __init__(self, _epoch_list: list = None, _loss_list: list = None, _lr_list: list =None):
        self.epoch_list = _epoch_list
        self.loss_list = _loss_list
        self.lr_list = _lr_list
        
    def append_result(self, epoch_num: int, loss: float, lr: float):
        if self.epoch_list == None:
            self.epoch_list = []
            self.loss_list = []
            self.lr_list = []
        self.epoch_list.append(epoch_num)
        self.loss_list.append(loss)
        self.lr_list.append(lr)