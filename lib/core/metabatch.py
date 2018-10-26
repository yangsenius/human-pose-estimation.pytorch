# ==================================================================
# Copyright (c) 2018, PatternRec, The Southeast University of China
# All rights reserved.
# Email: yangsenius@seu.edu.cn
# ==================================================================

# ========================
#  update 04-metabatch :
# 1. Hyper parameter CSR decreases with epoch
# 2. Cancel the hard learning stage
# 3. add the function of counting the number of hard,medium,easy
# =========================

from __future__ import absolute_import
from __future__ import division

import os
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import datetime
'''logging.basicConfig(filename='meta-log',
                    level=logging.DEBUG,
                   filemode='a',
                    format='%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
logging.getLogger('').addHandler(console)'''
logger=logging.getLogger(__name__)

class MetaData_Container(object):
    """
    push each new example into the MetaData_Container 
    and associate its ID with its attribute such as memory_difficult,forget_degree,loss and level.  
    """
    
    def __init__(self,total_num,batchsize,total_epoch):
        self.total_num=total_num
        self.total_epoch=total_epoch
        self.batchsize=batchsize
        #self.ID_list=[]  #面向整个数据集建立 全局动态唯一索引表
        #self.MetaData_Dict=[] #面向整个数据集建立 全局动态唯一索引表

        self.CSR = 1 # decay with epoched
        self.CSR_Decay_Table=[]
        self.difficult_threshold=60
        self.forget_degree_incrase=20
        self.easy_pattern_learning_epoch_ratio=0.5
        self.hard_learning_epoch_begin = self.easy_pattern_learning_epoch_ratio * total_epoch

        self.Gradient_Backward_num=0
        self.Gradient_Backward_Loss=0
        self.table=self.Bulid_Table()
        logger.info("\n##### MetaBatch Setting ######")
        logger.info("==> initial CSR value is {}".format(self.CSR))
        logger.info("==> difficult_threshold is {} range=[0,100]".format(self.difficult_threshold))
        logger.info("==> forget_degree_incrase is {} range=[0,100]".format(self.forget_degree_incrase))
        logger.info("==> easy_pattern_learning_epoch_ratio is not used")
        logger.info("###############################################\n")


    def CSR_Decay(self,epoch,final_epoch_csr=0.3):
        """
        Here `CSR` represents the critical sample ratio, which can be used to control the 
        the number of samples with small loss in a mini-batch learning. Range from [0,1]

        Set the CSR decays with epoch exponentially and Make sure the final epoch's csr equals to a small value
        """
        decay_ratio=(self.total_epoch-1)/(-np.log(final_epoch_csr))  
        csr=np.exp((-epoch-1)/decay_ratio)
        return csr


    def MiniBatch_MetaData_Loader(self,loss,Batchmeta,epoch):
        """betchmeta is `firsty` loaded from the return tuple of torch.utils.data.DataLoader 
        if each example in minibatch does not appear in self.ID_list, append it
        """
        Batchmeta_list=[]
        #assert len(loss)==self.batchsize
               
        #epoch>=1 从索引表中加载数据,epoch=0 不需要加载
        if epoch == 0:
            for id, loss in enumerate(loss):
                
                    #print(epoch)
                index=Batchmeta['index'][id]
                            #print('index=={}'.format(index))
                tmp={'index':index,
                    'memory_difficult':Batchmeta['memory_difficult'][id],
                    'forget_degree':Batchmeta['forget_degree'][id]}
                            #找到该样本在索引表中储存的META数据
                tmp['loss']=loss
                tmp['level']=2
                                #tmp['level']=1
                Batchmeta_list.append(tmp)
                    
                self.table['difficult_table'][index][epoch] = tmp['memory_difficult']
                self.table['forget_table'][index][epoch] = tmp['forget_degree']
                self.table['loss_table'][index][epoch] = tmp['loss'] 
                self.table['level_table'][index][epoch] = tmp['level']
                self.table['level_accmulation'][index][epoch] = tmp['level']
                
        else:
            for id, loss in enumerate(loss):
                index=Batchmeta['index'][id]
                difficult=self.table['difficult_table'][index][epoch-1]
                forget=self.table['forget_table'][index][epoch-1]
                level=self.table['level_table'][index][epoch-1]
                tmp={
                    'index':index,
                    'memory_difficult':difficult,
                    'forget_degree':forget,
                    'loss':loss,
                    'level':level}
                        #找到该样本在索引表中储存的META数据
                            #tmp['level']=1
                Batchmeta_list.append(tmp)

        return Batchmeta_list

    def Update_MiniBatch_MetaData(self,old_minibatchmeta):
        """input: old minibatch metadata
           update minibatch metadata and update the self.MetaData
           output：new minibatch metadata and minibatch-backwardloss
        """
        self.Gradient_Backward_Loss=0
        self.Gradient_Backward_num=0
        minibatch_backward_loss=0
        new_minibatchmeta=[]
        #assert len(old_minibatchmeta)==self.batchsize
        
        old_minibatchmeta=sorted(old_minibatchmeta,key=lambda tmp:tmp['loss'],reverse=True)
        #logger.info("==> OldBatchmeta has been sorted and Batchmeta contain {} instances".format(len(old_minibatchmeta)))
        
        for id, tmp in enumerate(old_minibatchmeta):            
            # in each batch, pick these large loss examples in CSR ratio range
            # and examine each difficult value 
            if id + 1 <= len(old_minibatchmeta)*self.CSR: 
                if tmp['memory_difficult']>= self.difficult_threshold: # 难
                    tmp['level']=2
                    # discard hard instances, meaning that their losses are not contributed to gradient backward
                    minibatch_backward_loss  += 0
                    # then we should make its forget_degree larger because the model doesn't learn it
                    tmp['forget_degree'] = tmp['forget_degree']+ 2*self.forget_degree_incrase   
                    #tmp['forget_degree'] += 2*self.forget_degree_incrase          
                else: # 中
                    tmp['level']=1
                    minibatch_backward_loss += tmp['loss']
                    self.Gradient_Backward_num +=1
                    tmp['forget_degree'] = tmp['forget_degree'] - self.forget_degree_incrase    
                    #tmp['forget_degree'] -= self.forget_degree_incrase                 
            else: # 易
                tmp['level']=0
                minibatch_backward_loss += tmp['loss']
                self.Gradient_Backward_num +=1
                
                tmp['forget_degree'] =  tmp['forget_degree'] - self.forget_degree_incrase 
                #tmp['forget_degree'] -= self.forget_degree_incrase 
            # final operation: 
            # we should redefine the difficult for each batch example
            if tmp['forget_degree'] < 0:              
                tmp['forget_degree'] = 0               
            if tmp['forget_degree'] > 100:  
                tmp['forget_degree'] = 100 

            tmp['memory_difficult'] = int((tmp['memory_difficult']+tmp['forget_degree'])/2)     
            new_minibatchmeta.append(tmp)
            
        self.Gradient_Backward_Loss=minibatch_backward_loss
        return new_minibatchmeta

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
               'level_accmulation':Tabel_Index_Level_accmulation,
               }
        return Table
    
    def Update_Table_Index(self,epoch,meta_batch):
        """update the values of table with epoch increases"""
        Table = self.table 
        if epoch > 0:         
            for id in range(len(meta_batch)):
                index=meta_batch[id]['index']
                #Table['index_table'][index][epoch] = meta_batch[id]['index']
                Table['difficult_table'][index][epoch] = meta_batch[id]['memory_difficult']
                Table['forget_table'][index][epoch] = meta_batch[id]['forget_degree']
                Table['loss_table'][index][epoch] = meta_batch[id]['loss'] 
                Table['level_table'][index][epoch] = meta_batch[id]['level']
                Table['level_accmulation'][index][epoch] = Table['level_accmulation'][index][epoch-1]+meta_batch[id]['level']
            self.table=Table

    def hard_example_learning(self,meta,epoch):
        """when the  model has learn the general pattern from datas with some easy examples 
        and the training steps into robust learning stage. 
        We should feed more hard example into model training.
        How to define the 'hard' example? 如何定义“困难”样本？ 查询 难度累积表 即可！
        The Tabel_Index_Level_Accmulation accmulates the historical difficult for each example, 
        we define a example with large Level_Accmulation values as 'hard' example , then we feed them to the model"""
        if epoch > self.hard_learning_epoch_begin:
            
            for tmp in meta:
                if tmp['level'] == 2: # means hard example
                    Level_Accmulation=self.table['level_accmulation'][tmp['index']][epoch]
                    if (Level_Accmulation/epoch) > 1 and (Level_Accmulation/epoch) < 1.5: # 如果每个周期的平均难度 超过中等（level=1）且不是特别难 level<1.5
                        #定义其为'hard' example
                        self.Gradient_Backward_Loss += tmp['loss'] #把那些难度比较大的样本的loss 加入有效梯度中
                        self.Gradient_Backward_num +=1
                        #这样做依旧让原来的规则有效，更加强调了困难样本的学习！

    def API_LOSS(self,loss,meta,epoch):
        """
        1.加载数据每个mini-batch中每个样本的的meta数据：
            if 第一次遇见该样本： 从dataloader中加载mini-batch中的meta
            else：从索引表中 加载minibatch中的meta数据
        2.根据当前mini-batch中在当前迭代中各个样本的loss值，更新mini-batch中的meta数据，计算最佳回传梯度
        3.根据新的mini-batch中的meta数据，更新索引表
        4.if 周期进行到困难样本学习阶段：对于mini-batch中没有对梯对回传贡献的样本，考察其历史平均难度，对难样本考虑其梯度
        4.根据更新后的mini-batch中的meta数据，更新全局动态索引表中meta数据

        """   
        #logger.info('==> t = {}, CSR = {}'.format(epoch,self.CSR))
        
        Batchmeta = self.MiniBatch_MetaData_Loader(loss,meta,epoch) 
        
        new_Batchmeta = self.Update_MiniBatch_MetaData(Batchmeta)
        
        if epoch>0:
            self.Update_Table_Index(epoch,new_Batchmeta)

        #if epoch>self.hard_learning_epoch_begin:
        #    self.hard_example_learning(new_Batchmeta,epoch)
        self.CSR=self.CSR_Decay(epoch)

        best_backward_gradient=self.Gradient_Backward_Loss/(self.Gradient_Backward_num+1)
        return best_backward_gradient
        
    def Add_Meta_To_Trainset(self,trainset):
        trainset_meta=trainset_add_meta(trainset)
        return trainset_meta
    
    def Random_Label_To_Trainset(self,trainset,rand_possibility=0.3):
        trainset_randY=trainset_random_label(trainset,rand_possibility)
        return trainset_randY

    def Output_CSV_Table(self,table_dir_name='dynamic_table'):
        isExists=os.path.exists(table_dir_name)
        if not isExists:
            os.makedirs(table_dir_name) 
        columns=list(map(lambda k:'t={}'.format(k), [k for k in range(self.total_epoch)]))

        
        Difficult_Table=self.table['difficult_table']
        table1=Difficult_Table.numpy().astype(int)
        data1 = pd.DataFrame(table1,columns=columns)
        logger.info('==> difficult table is \n{}\n====================='.format(data1))
        data1.to_csv(os.path.join(table_dir_name,'difficult_talbe.csv'))

        Forget_Table=self.table['forget_table']
        table_F=Forget_Table.numpy().astype(int)
        data_f = pd.DataFrame(table_F,columns=columns)
        logger.info('==> forget table is \n{}\n====================='.format(data_f))
        data_f.to_csv(os.path.join(table_dir_name,'forget_talbe.csv'))

        
        Level_Table=self.table['level_table']
        table2=Level_Table.numpy().astype(int)
        data2 = pd.DataFrame(table2,columns=columns)
        logger.info('==> level table is \n{}\n====================='.format(data2))
        data2.to_csv(os.path.join(table_dir_name,'level_talbe.csv'))

        # record the statistics of  hard or easy or medium
        level_ratio_table=torch.zeros(size=(3,self.total_epoch),)
        for i in range(3):
            hard=torch.eq(Level_Table,i*torch.ones(Level_Table.size()))
            level_ratio_table[i]=torch.sum(hard, 0)
        table4=level_ratio_table.numpy().astype(int)/self.total_num
        
        data4 = pd.DataFrame(table4,index=['easy','medi','hard'],columns=columns).round(2)
        logger.info('==> percentage table of easy,medium and hard examples \n{}\n====================='.format(data4))
        data4.to_csv(os.path.join(table_dir_name,'level_num.csv'))

        
        Loss_Table=self.table['loss_table']
        table3=Loss_Table.detach().numpy()
        data3 = pd.DataFrame(table3,columns=columns).round(5)
        logger.info('==> loss table is \n{}\n====================='.format(data3))
        data3.to_csv(os.path.join(table_dir_name,'loss_talbe.csv'))

        logger.info('CSR_Decay_Table={}'.format(self.CSR_Decay_Table))

        
        

class trainset_add_meta(torch.utils.data.Dataset):
    
    def __init__(self,trainset):
           self.trainset=trainset
    def __len__(self):
        return len(self.trainset)
    def __getitem__(self,id):
        item=self.trainset.__getitem__(id)
        meta={'index':id,
          'memory_difficult':50,
          'forget_degree':100}
        item=list(item)
        item.append(meta)
        item=tuple(item)
        return item

class trainset_random_label(torch.utils.data.Dataset):
    
    def __init__(self,trainset,rand_possibility=0.3):
           self.trainset=trainset
           self.rand_possibility=rand_possibility
    def __len__(self):
        return len(self.trainset)
    def __getitem__(self,id):
        item=self.trainset.__getitem__(id)
        if np.random.rand(1)<=self.rand_possibility:  #改变lable的概率
            item=list(item)
            item[1]=np.random.randint(10) #改变lable标签
            item=tuple(item)
        return item
        
        
class Meta_dataset(Dataset):
    def __init__(self,l):
       self.len=l
    def __len__(self):
        return self.len

    def __getitem__(self, id):
        tmp={'index':id,
            'memory_difficult':60,
            'forget_degree':100,
            'loss':5*np.random.rand(1)}
            
        return tmp['loss'],tmp

        

def train(train_loader,MetaData_Container, epoch):
    
    M_C=MetaData_Container
    logger.info('==> t = {}, CSR = {}'.format(epoch,M_C.CSR))
    M_C.CSR_Decay_Table.append(round(M_C.CSR,2))

    for i, (loss, meta) in enumerate(train_loader):
        
        Best_Loss = M_C.API_LOSS(loss,meta,epoch) 
        

def main():
    logger.info("############# experiment {} ################".format(datetime.datetime.now()))
    batchsize=16
    total_epoch=16
    meta_dataset=Meta_dataset(64)
    train_loader = torch.utils.data.DataLoader(
        meta_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    datasets_num=len(meta_dataset)
    print(datasets_num)

    #在训练周期开始前，初始定义meta数据容器，随意起名
    SEU_YS=MetaData_Container(datasets_num,batchsize,total_epoch)
    
    for epoch in range(total_epoch):
        train(train_loader,SEU_YS,epoch)

    #print(SEU_YS.MetaData_Dict)
        SEU_YS.Output_CSV_Table()

if __name__ =='__main__':
    main()
