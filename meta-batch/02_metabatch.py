
# Copyright (c) 2018, PatternRec, The Southeast University of China
# All rights reserved.
# Email: yangsenius@seu.edu.cn

from __future__ import absolute_import


import logging
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

logging.basicConfig(filename='meta-log',
                    level=logging.DEBUG,
                    filemode='a',
                    format='\n%(asctime)s: \n -- %(pathname)s \n --[line:%(lineno)d]\n -- %(levelname)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
logging.getLogger('').addHandler(console)

class MetaData_Container(object):
       
    def __init__(self,total_num,batchsize,total_epoch):
        self.total_num=total_num
        self.total_epoch=total_epoch
        self.batchsize=batchsize
        self.ID_list=[]  #面向整个数据集建立 全局动态唯一索引表
        self.MetaData_Dict=[] #面向整个数据集建立 全局动态唯一索引表
        
        self.CSR = 0.3
        self.difficult_threshold=50
        self.forget_degree_incrase=20
        self.easy_pattern_learning_epoch_ratio=0.667
        self.hard_learning_epoch_begin = self.easy_pattern_learning_epoch_ratio * total_epoch

        self.Gradient_Backward_num=0
        self.Gradient_Backward_Loss=0
        self.table=self.Bulid_Table()

    def MiniBatch_MetaData_Loader(self,loss,Batchmeta):
        Batchmeta_list=[]
        assert len(loss)==self.batchsize
        for id, loss in enumerate(loss):
            # 耗时
            if Batchmeta['index'][id] not in self.ID_list: #第一次碰到这个样本 epoch=0                
                
                tmp={'index':Batchmeta['index'][id], # identiy index
                    'memory_difficult':Batchmeta['memory_difficult'][id], # difficult value for each epoch
                    'forget_degree':Batchmeta['forget_degree'][id],
                    'loss':loss,
                    'level':2} 
                Batchmeta_list.append(tmp)
                self.ID_list.append(tmp['index']) #面向整个数据集建立 全局动态索引表
                self.MetaData_Dict.append({'index':tmp['index'],
                                      'memory_difficult':tmp['memory_difficult'],
                                      'forget_degree':tmp['forget_degree'],
                                      'level':tmp['level']}) # 初始难度均为2
            else: #epoch>=1 从索引表中加载数据
                for ID, index in enumerate(self.ID_list):
                    #print('index=={}'.format(index))
                    if index == Batchmeta['index'][id]:
                        tmp=self.MetaData_Dict[ID]  #找到该样本在索引表中储存的META数据
                        tmp['loss']=loss
                        #tmp['level']=1
                        Batchmeta_list.append(tmp)
        return Batchmeta_list

    def Update_MiniBatch_MetaData(self,old_minibatchmeta):
        self.Gradient_Backward_num = 0
        self.Gradient_Backward_Loss = 0
        minibatch_backward_loss=0
        new_minibatchmeta=[]
        assert len(old_minibatchmeta)==self.batchsize
        old_minibatchmeta=sorted(old_minibatchmeta,key=lambda tmp:tmp['loss'],reverse=True)
        #logger.info("==> OldBatchmeta has been sorted and Batchmeta contain {} instances".format(len(old_minibatchmeta)))
        
        for id, tmp in enumerate(old_minibatchmeta):            
             
            if id + 1 <= self.batchsize*self.CSR: 
                if tmp['memory_difficult']>= self.difficult_threshold: # 难
                    tmp['level']=2                    
                    minibatch_backward_loss  += 0    
                    tmp['forget_degree'] += 2*self.forget_degree_incrase               
                else: # 中
                    tmp['level']=1
                    minibatch_backward_loss += tmp['loss']
                    self.Gradient_Backward_num +=1
                    tmp['forget_degree'] -= self.forget_degree_incrase                    
            else: # 易
                tmp['level']=0
                minibatch_backward_loss += tmp['loss']
                self.Gradient_Backward_num +=1
                tmp['forget_degree'] -= self.forget_degree_incrase 
            
            tmp['forget_degree'] = 0 if tmp['forget_degree'] < 0 else tmp['forget_degree']
            tmp['forget_degree'] = 100 if tmp['forget_degree'] > 100 else tmp['forget_degree']

            tmp['memory_difficult'] = (tmp['memory_difficult']+tmp['forget_degree'])/2           
            new_minibatchmeta.append(tmp)
        
        self.Gradient_Backward_Loss=minibatch_backward_loss
        return new_minibatchmeta

    def Update_New_Minibatch_To_MetaData_Dict(self,new_minibatchmeta):
        
        assert len(new_minibatchmeta)==self.batchsize
        for tmp in new_minibatchmeta:
            for id, index in enumerate(self.ID_list):
                if tmp['index'] == index:
                    self.MetaData_Dict[id]['memory_difficult']=tmp['memory_difficult']
                    self.MetaData_Dict[id]['forget_degree']=tmp['forget_degree']
                    self.MetaData_Dict[id]['level']=tmp['level']

    def Bulid_Table(self):
        """inital Table
        preserve all values in all epoches"""
        total_num,total_epoch=self.total_num,self.total_epoch
        #Table_Index_ID = torch.zeros(size=(total_num,total_epoch)) 
        Table_Index_Difficult = torch.zeros(size=(total_num,total_epoch))
        Table_Index_Forget = torch.zeros(size=(total_num,total_epoch))
        Table_Index_Loss = torch.zeros(size=(total_num,total_epoch))
        Table_Index_Level = torch.zeros(size=(total_num,total_epoch))
        Tabel_Index_Level_accmulation = torch.zeros(size=(total_num,total_epoch))
        Table={
               'difficult_table':Table_Index_Difficult,
               'forget_table':Table_Index_Forget,
               'loss_table':Table_Index_Loss,
               'level_table':Table_Index_Level,
               'level_accmulation':Tabel_Index_Level_accmulation}
        return Table
    
    def Update_Table_Index(self,epoch,meta_batch):
        
        Table = self.table 
        for id in range(len(meta_batch)):
            index=meta_batch[id]['index']
            #Table['index_table'][index][epoch] = meta_batch[id]['index']
            Table['difficult_table'][index][epoch] = meta_batch[id]['memory_difficult']
            Table['forget_table'][index][epoch] = meta_batch[id]['forget_degree']
            Table['loss_table'][index][epoch] = meta_batch[id]['loss'] 
            Table['level_table'][index][epoch] = meta_batch[id]['level']
            Table['level_accmulation'][index][epoch] = Table['level_accmulation'][index][epoch-1] \
                                                        +meta_batch[id]['level'] if epoch !=0 else 2
        self.table=Table

    def hard_example_learning(self,meta,epoch):
        if epoch > self.hard_learning_epoch_begin:
            for tmp in meta:
                if tmp['level'] == 2: # means hard example
                    Level_Accmulation=self.table['level_accmulation'][tmp['index']][epoch]
                    if (Level_Accmulation/epoch) > 1:                         
                        self.Gradient_Backward_Loss += tmp['loss']
                        self.Gradient_Backward_num += 1 
                        

    def API_LOSS(self,loss,meta,epoch):
               
        Batchmeta = self.MiniBatch_MetaData_Loader(loss,meta) 
        self.Update_Table_Index(epoch,Batchmeta)
        new_Batchmeta = self.Update_MiniBatch_MetaData(Batchmeta)
        self.Update_New_Minibatch_To_MetaData_Dict(new_Batchmeta)       
        #if epoch>self.hard_learning_epoch_begin:
           # self.hard_example_learning(new_Batchmeta,epoch)

        best_backward_gradient=self.Gradient_Backward_Loss/self.Gradient_Backward_num
        return best_backward_gradient

    def Add_Meta_To_Trainset(self,trainset):
        trainset_meta=trainset_add_meta(trainset)
        return trainset_meta

class trainset_add_meta(torch.utils.data.Dataset):
    
    def __init__(self,trainset):
           self.trainset=trainset
    def __len__(self):
        return len(self.trainset)
    def __getitem__(self,id):
        item=self.trainset.__getitem__(id)
        meta={'index':id,
          'memory_difficult':np.random.randint(100),
          'forget_degree':100}
        item=list(item)
        item.append(meta)
        item=tuple(item)
        return item

       
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

        Best_Loss = M_C.API_LOSS(loss,meta,epoch) 
        # please make  best_loss backward!! 让best_loss回传即可

def main():
    batchsize=8
    total_epoch=20
    meta_dataset=Meta_dataset(80)
    train_loader = torch.utils.data.DataLoader(
        meta_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    datasets_num=len(meta_dataset)
    logger.info('==> total examples is {}'.format(datasets_num))
    logger.info('==> total training epoches is {}'.format(total_epoch))
    logger.info('==> batchsize is {}'.format(batchsize))
    #在训练周期开始前，初始定义meta数据容器，参数为：数据集总量，batchsize大小，训练总周期
    SEU_YS=MetaData_Container(datasets_num,batchsize,total_epoch)
    
    for epoch in range(total_epoch):
        train(train_loader,SEU_YS,epoch)

    #print(SEU_YS.MetaData_Dict)
    logger.info('==>> difficult table is \n{}\n0:easy\n1:medium\n2:hard'.format(SEU_YS.table['level_table']))
    Difficult_Table=SEU_YS.table['level_table']
    table=Difficult_Table.numpy().astype(int)
    data1 = pd.DataFrame(table)
    data1.to_csv('difficult_talbe.csv')
    
         
if __name__ =='__main__':
    main()