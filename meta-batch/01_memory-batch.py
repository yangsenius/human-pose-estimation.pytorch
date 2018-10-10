import torch
import logging

logging.basicConfig(filename='meta-batch/meta-log',
                    level=logging.DEBUG,
                    filemode='a',
                    format='\n%(asctime)s: \n -- %(pathname)s \n --[line:%(lineno)d]\n -- %(levelname)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
logging.getLogger('').addHandler(console)
#logger=logging.getLogger(__name__)

memory={
    'index': torch.tensor([ 13061,  95320, 109957,  10488,  90388,  21084,  52478,  52296]), 
    'memory_difficult': torch.tensor([0.0669, 0.2955, 0.0984, 0.1302, 0.1510, 0.0758, 0.7272, 0.1129]), 
    'forget_degree': torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])}
#print(memory)

loss=torch.tensor([2.10,0.215,2.3,1.11,4.2,0.22,0.36,0.86])
#print(loss)

class Memory_Batch(object):
    """ @yangsen\n
    @:周期循环
      #:每个周期内打乱数据集，循环加载一个Mini—batch,计算一个mini-batch下的输出loss
        1.根据模型输出的loss，由大到小排序一个Mini-Batch中的所有样本S_{all}
        2.按照CSR在一个Batch中选择一定比例的样本集S_{csr}，考察这些样本的难度系数。其余样本设为S_{easy}
        3.难：如果S_{csr}中样本难度系数较大(超过设定阈值)，抛弃该样本对梯度回传的贡献，同时增加其遗忘程度的数值，增量为delta
        4.中：如果S_{csr}中样本难度系数一般(小于设定阈值)，保留该样本对梯度回传的贡献，同时减少其遗忘程度的数值，减量为delta
        5.易：对于S_{easy}中的样本，保留该样本对梯度回传的贡献，同时减少其遗忘程度的数值，减量为2*delta
        6.对于S_{all}的样本，重新计算其难度系数，新难度系数=1/2*(旧难度系数+遗忘程度)

        CSR is critical Sample Ratio roughly denoted for the noise ratio here\n
        In this algorithm, CSR value changes with epoch increasing\n
        CSR see the paper : https://arxiv.org/pdf/1706.05394.pdf
        
        `memory_difficult`: affected by initial difficult and loss in each iteration \n
        `forget_degree`: affected by the times of being discarded \n
        In each batch:\n
            1. compute the each instance's loss in a mini-batch by loss function
            2. sorted: all instances by the loss value of each instance
            3. pick some instances with large loss from a certain ratio of all. Refernce the CSR ratio.
            4. acoording to `memory_difficult` compute the batch_backward_loss
            5. update the `forget_degree` and `memory_difficult`
            
        """
    def __init__(self,loss,memory,total_epoch):
        self.loss=loss
        self.batchsize = len(loss)
        self.memory=memory
        self.total_epoch=total_epoch
        self.CSR = 0.3
        self.difficult_threshold=0.5
        self.forget_degree_incrase=0.2
        
   
        self.batch_backward_loss=0 # final gradient backward loss
        self.batch_meta =self.Sort_Batch(memory,loss)

        
        

    def Sort_Batch(self,memory,loss):
        batch_meta_index = [] #一个batch样本实例的索引表
        batch_meta = []
        for id,loss in enumerate(loss):
            tmp={'index':memory['index'][id], # identiy index
                'memory_difficult':memory['memory_difficult'][id], # difficult value for each epoch
                'forget_degree':memory['forget_degree'][id],
                'loss':loss} # forget_degree
            if tmp['index'] not in batch_meta_index:
                batch_meta_index.append(tmp['index'])
                batch_meta.append(tmp)
        batch_meta=sorted(batch_meta,key=lambda tmp:tmp['loss'],reverse=True) #sort by loss ,larger first   
        logger.info("==> batch_meta contain {} instances".format(len(batch_meta)))
        return batch_meta

    def Update_Memory_Batch(self,epoch): 
        batch_meta=self.batch_meta
        CSR=self.CSR      
        difficult_threshold=self.difficult_threshold
        forget_degree_incrase=self.forget_degree_incrase
        new_batch_meta=[]
        batch_backward_loss=0
        for id, tmp in enumerate(batch_meta):
            
            # in each batch, pick these large loss examples in CSR ratio range
            # and examine each difficult value 
            if id + 1 <= self.batchsize*CSR: 
                if tmp['memory_difficult']>= difficult_threshold: # 难
                    tmp['level']=2
                    # discard hard instances, meaning that their losses are not contributed to gradient backward
                    batch_backward_loss  += 0
                    # then we should make its forget_degree larger because the model doesn't learn it
                    tmp['forget_degree'] += forget_degree_incrase               
                else: # 中
                    tmp['level']=1
                    # meaning that their losses are not contributed to gradient backward
                    batch_backward_loss += tmp['loss']
                    tmp['forget_degree'] -= forget_degree_incrase                    
            else: # 易
                tmp['level']=0
                batch_backward_loss += tmp['loss']
                tmp['forget_degree'] -= 2*forget_degree_incrase #规定容易的被忘记地更快
            # final operation: 
            # we should redefine the difficult for each batch example
            tmp['forget_degree'] = 0 if tmp['forget_degree'] < 0 else tmp['forget_degree']
            tmp['forget_degree'] = 1 if tmp['forget_degree'] > 1 else tmp['forget_degree']

            tmp['memory_difficult'] = 0.5*tmp['memory_difficult']+0.5*tmp['forget_degree']
            new_batch_meta.append(tmp)

        self.batch_meta = new_batch_meta
        return batch_backward_loss

    def Build_Table(self):
        batchsize,total_epoch=self.batchsize,self.total_epoch
        Table_Index_ID = torch.zeros(size=(batchsize,total_epoch)) 
        Table_Index_Difficult = torch.zeros(size=(batchsize,total_epoch))
        Table_Index_Forget = torch.zeros(size=(batchsize,total_epoch))
        Table_Index_Loss = torch.zeros(size=(batchsize,total_epoch))
        Table_Index_Level = torch.zeros(size=(batchsize,total_epoch))
        Table=(Table_Index_ID,Table_Index_Difficult,Table_Index_Forget,Table_Index_Loss,Table_Index_Level)
        return Table

    def Update_Table_Index(self,Table,epoch,meta_batch):
        
        for id in range(len(meta_batch)):
            Table[0][id][epoch] = meta_batch[id]['index']
            Table[1][id][epoch] = meta_batch[id]['memory_difficult']
            Table[2][id][epoch] = meta_batch[id]['forget_degree']
            Table[3][id][epoch] = meta_batch[id]['loss'] if 'loss' in meta_batch[id] else 0.
            Table[4][id][epoch] = meta_batch[id]['level'] if 'level' in meta_batch[id] else 5
        return Table

def main():
    logging.basicConfig(filename='meta-batch/meta-log',
                    level=logging.DEBUG,
                    filemode='a',
                    format='\n%(asctime)s: \n -- %(pathname)s \n --[line:%(lineno)d]\n -- %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    #logger=logging.getLogger(__name__)
    memory={
        'index': torch.tensor([ 13061,  95320, 109957,  10488,  90388,  21084,  52478,  52296]), 
        'memory_difficult': torch.tensor([0.0669, 0.2955, 0.0984, 0.1302, 0.1510, 0.0758, 0.7272, 0.1129]), 
        'forget_degree': torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])}
    #print(memory)

    loss=torch.tensor([2.10,0.215,2.3,1.11,4.2,0.22,0.36,0.86])
    #print(loss)






    total_epoch=4
    YS = Memory_Batch(loss,memory,total_epoch)
    Table = YS.Build_Table()

    for epoch in range(total_epoch):
        Table=YS.Update_Table_Index(Table,epoch,YS.batch_meta)

        
        batch_backward_loss = YS.Update_Memory_Batch(epoch)

        
        print("==> batch_backward_loss is {}".format(batch_backward_loss))

    print("ID=\n{}".format(Table[0]))
    print("Difficult=\n{}".format(Table[1]))
    print("Forget=\n{}".format(Table[2]))
    print("Loss=\n{}".format(Table[3]))
    print("Level=\n{}".format(Table[4]))



if __name__ == '__main__':
    main()
