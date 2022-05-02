from config import train_config
from train import distributed_train,main_worker
from evaluation import all_eval
import argparse
import fire
import torch
import subprocess
#torch.autograd.set_detect_anomaly(True)

def pretrain():
    name='Efb4new40'
    url='tcp://127.0.0.1:27015'
    Config=train_config(name,['ff-all-c40','efficientnet-b4'],url=url,attention_layer='b5',feature_layer='logits',epochs=3,batch_size=8,AGDA_loss_weight=0)
    Config.mkdirs()
    print('in')
    distributed_train(Config) 
    print('out')
    procs=[subprocess.Popen(['/bin/bash','-c','CUDA_VISIBLE_DEVICES={} python main.py test {} {}'.format(i,name,j)]) for i,j in enumerate(range(-3,0))]
    for i in procs:
        i.wait()

## do pretrain first!
def aexp():
    name='a1_b5_b2'  
    url='tcp://127.0.0.1:27016'
    Config=train_config(name,['ff-all-c23','efficientnet-b4'],url=url,attention_layer='b5',feature_layer='b2',epochs=50,batch_size=15,\
        ckpt='checkpoints/Efb4/ckpt_19.pth',inner_margin=[0.2,-0.8],margin=0.8)
    Config.mkdirs()
    distributed_train(Config) 
    procs=[subprocess.Popen(['/bin/bash','-c','CUDA_VISIBLE_DEVICES={} python main.py test {} {}'.format(i,name,j)]) for i,j in enumerate(range(-3,0))]
    for i in procs:
        i.wait()


def resume(name,epochs=0):
    Config=train_config.load(name)
    Config.epochs+=epochs
    Config.reload()
    Config.resume_optim=True
    distributed_train(Config) 
    for i in range(-3,0):
        all_eval(name,i)

def test(name,ckpt=None):
    # all_eval(name,ckpt)
    print(torch.cuda.device_count())
    all_eval(name, ckpt, test_sets=['celeb'])

if __name__=="__main__":
    fire.Fire()
