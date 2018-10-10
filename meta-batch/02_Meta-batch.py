from __future__ import absolute_import

import logging
import torch
import numpy as np
from torch.utils.data import Dataset

logging.basicConfig(filename='meta-batch/meta-log',
                    level=logging.DEBUG,
                    filemode='a',
                    format='\n%(asctime)s: \n -- %(pathname)s \n --[line:%(lineno)d]\n -- %(levelname)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
logging.getLogger('').addHandler(console)

class MetaData_Container(object):
    """
    push each new example into the MetaData_Container 
    and associate its ID with its attribute such as memory_difficult,forget_degree,loss and level.  
    """
    
    def __init__(self,total_num,batchsize,total_epoch):
        self.total_num=total_num
        self.total_epoch=total_epoch
        self.batchsize=batchsize
        self.ID_list=[]  #面向整个数据集建立 全局动态唯一索引表
        self.MetaData_Dict=[] #面向整个数据集建立 全局动态唯一索引表

        self.CSR = 0.3
        self.difficult_threshold=50
        self.forget_degree_incrase=20

        self.Gradient_Backward_Loss=0
        self.table=self.Bulid_Table()

    def MiniBatch_MetaData_Loader(self,loss,Batchmeta):
        """betchmeta is `firsty` loaded from the return tuple of torch.utils.data.DataLoader 
        if each example in minibatch does not appear in self.ID_list, append it
        """
        Batchmeta_list=[]
        assert len(loss)==self.batchsize
        for id, loss in enumerate(loss):
            if Batchmeta['index'][id] not in self.ID_list: #第一次碰到这个样本 epoch=0
                #print("index{}".format(Batchmeta['index'][id]))
                tmp={'index':Batchmeta['index'][id], # identiy index
                    'memory_difficult':Batchmeta['memory_difficult'][id], # difficult value for each epoch
                    'forget_degree':Batchmeta['forget_degree'][id],
                    'loss':loss} 
                Batchmeta_list.append(tmp)
                self.ID_list.append(tmp['index']) #面向整个数据集建立 全局动态索引表
                self.MetaData_Dict.append({'index':tmp['index'],
                                      'memory_difficult':tmp['memory_difficult'],
                                      'forget_degree':tmp['forget_degree']}) # loss 不被加入到全局索引表中
            else: #epoch>=1 从索引表中加载数据
                for ID, index in enumerate(self.ID_list):
                    #print('index=={}'.format(index))
                    if index == Batchmeta['index'][id]:
                        tmp=self.MetaData_Dict[ID]  #找到该样本在索引表中储存的META数据
                        tmp['loss']=loss
                        Batchmeta_list.append(tmp)
        return Batchmeta_list

    def Update_MiniBatch_MetaData(self,old_minibatchmeta):
        """input: old minibatch metadata
           update minibatch metadata and update the self.MetaData
           output：new minibatch metadata and minibatch-backwardloss
        """
        minibatch_backward_loss=0
        new_minibatchmeta=[]
        assert len(old_minibatchmeta)==self.batchsize
        old_minibatchmeta=sorted(old_minibatchmeta,key=lambda tmp:tmp['loss'],reverse=True)
        #logger.info("==> OldBatchmeta has been sorted and Batchmeta contain {} instances".format(len(old_minibatchmeta)))
        
        for id, tmp in enumerate(old_minibatchmeta):            
            # in each batch, pick these large loss examples in CSR ratio range
            # and examine each difficult value 
            if id + 1 <= self.batchsize*self.CSR: 
                if tmp['memory_difficult']>= self.difficult_threshold: # 难
                    tmp['level']=2
                    # discard hard instances, meaning that their losses are not contributed to gradient backward
                    minibatch_backward_loss  += 0
                    # then we should make its forget_degree larger because the model doesn't learn it
                    tmp['forget_degree'] += 2*self.forget_degree_incrase               
                else: # 中
                    tmp['level']=1
                    # meaning that their losses are not contributed to gradient backward
                    minibatch_backward_loss += tmp['loss']
                    tmp['forget_degree'] -= self.forget_degree_incrase                    
            else: # 易
                tmp['level']=0
                minibatch_backward_loss += tmp['loss']
                tmp['forget_degree'] -= 2*self.forget_degree_incrase #规定容易的被忘记地更快
            # final operation: 
            # we should redefine the difficult for each batch example
            tmp['forget_degree'] = 0 if tmp['forget_degree'] < 0 else tmp['forget_degree']
            tmp['forget_degree'] = 100 if tmp['forget_degree'] > 100 else tmp['forget_degree']

            tmp['memory_difficult'] = int(0.5*tmp['memory_difficult']+0.5*tmp['forget_degree'])
            new_minibatchmeta.append(tmp)
        #logger.info("minibatch_meta and backward_loss have been updated ")
        self.Gradient_Backward_Loss=minibatch_backward_loss
        return new_minibatchmeta

    def Update_New_Minibatch_To_MetaData_Dict(self,new_minibatchmeta):
        """input: new_minibatchmeta
           update: minibatchmeta value in self.MetaData
        """
        assert len(new_minibatchmeta)==self.batchsize
        for tmp in new_minibatchmeta:
            for id, index in enumerate(self.ID_list):
                if tmp['index'] == index:
                    self.MetaData_Dict[id]['memory_difficult']=tmp['memory_difficult']
                    self.MetaData_Dict[id]['forget_degree']=tmp['forget_degree']

    def Bulid_Table(self):
        """inital Table
        preserve all values in all epoches"""
        total_num,total_epoch=self.total_num,self.total_epoch
        Table_Index_ID = torch.zeros(size=(total_num,total_epoch)) 
        Table_Index_Difficult = torch.zeros(size=(total_num,total_epoch))
        Table_Index_Forget = torch.zeros(size=(total_num,total_epoch))
        Table_Index_Loss = torch.zeros(size=(total_num,total_epoch))
        Table_Index_Level = torch.zeros(size=(total_num,total_epoch))
        Table={'index_table':Table_Index_ID,
               'difficult_table':Table_Index_Difficult,
               'forget_table':Table_Index_Forget,
               'loss_table':Table_Index_Loss,
               'level_table':Table_Index_Level}
        return Table
    
    def Update_Table_Index(self,epoch,meta_batch):
        """update the values of table with epoch increases"""
        Table = self.table 
        for id in range(len(meta_batch)):
            index=meta_batch[id]['index']
            Table['index_table'][index][epoch] = meta_batch[id]['index']
            Table['difficult_table'][index][epoch] = meta_batch[id]['memory_difficult']
            Table['forget_table'][index][epoch] = meta_batch[id]['forget_degree']
            Table['loss_table'][index][epoch] = meta_batch[id]['loss'] if 'loss' in meta_batch[id] else 0.
            Table['level_table'][index][epoch] = meta_batch[id]['level'] if 'level' in meta_batch[id] else 5
        self.table=Table
        return Table
        
class Meta_dataset(Dataset):
    def __init__(self,l):
       self.len=l
    def __len__(self):
        return self.len

    def __getitem__(self, id):
        tmp={'index':id,
            'memory_difficult':np.random.randint(100),
            'forget_degree':100,
            'loss':5*np.random.rand(1)}
            
        return tmp['loss'],tmp

        

def train(train_loader,MetaData_Container, epoch):
    
    M_C=MetaData_Container

    for i, (loss, meta) in enumerate(train_loader):
        
        Batchmeta = M_C.MiniBatch_MetaData_Loader(loss,meta) 
        table=M_C.Update_Table_Index(epoch,Batchmeta)

        new_batchmeta = M_C.Update_MiniBatch_MetaData(Batchmeta)

        M_C.Update_New_Minibatch_To_MetaData_Dict(new_batchmeta)

        loss=M_C.Gradient_Backward_Loss





def main():
    batchsize=4
    total_epoch=10
    meta_dataset=Meta_dataset(36)
    train_loader = torch.utils.data.DataLoader(
        meta_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    datasets_num=len(meta_dataset)
    print(datasets_num)
    SEU_YS=MetaData_Container(datasets_num,batchsize,total_epoch)
    

    for epoch in range(total_epoch):
        train(train_loader,SEU_YS,epoch)

    #print(SEU_YS.MetaData_Dict)
    print(SEU_YS.table)

if __name__ =='__main__':
    main()