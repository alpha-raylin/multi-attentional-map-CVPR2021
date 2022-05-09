import os 
import json
import random
dfdcroot=''
celebroot='../../Celeb_img/Celeb_img/' 
ffpproot='../../data-x/g15/ffpp-faces/'
jpegroot='../../data-x/g15/ffpp-faces/'

deeperforensics_root=''
def load_json(name):
    with open(name) as f:
        a=json.load(f)
    return a

print(os.path.exists(celebroot+'celeb_synthesis'))
with open ('celeb.json', 'w') as f:
    fake = [['celeb_synthesis/'+i, 1] for i in os.listdir(celebroot+'celeb_synthesis')]
    real = [['celeb_original/'+i, 0] for i in os.listdir(celebroot+'celeb_original')]
    youtube = [['youtube_original/'+i, 0] for i in os.listdir(celebroot+'youtube_original')]
    
    total = youtube+real+fake
    json.dump(total, f)


def catdir(dir,label):
    l=os.listdir(dir)
    return [[os.path.join(dir,i),label] for i in l]

def FF_dataset(tag='Origin',codec='c0',part='train'):
    assert(tag in ['Origin','Deepfakes','NeuralTextures','FaceSwap','Face2Face','FaceShifter'])
    assert(codec in ['c0','c23','c40','all'])
    assert(part in ['train','val','test','all'])
    if part=="all":
        return FF_dataset(tag,codec,'train')+FF_dataset(tag,codec,'val')+FF_dataset(tag,codec,'test')
    if codec=='all':
        return FF_dataset(tag,'c0',part)+FF_dataset(tag,'c23',part)+FF_dataset(tag,'c40',part)
    path=ffpproot+'%s/%s/larger_images/'%(tag,codec)
    metafile=load_json(ffpproot+'ffpp/'+part+'.json')
    files=[]
    if tag=='Origin':
        for i in metafile:
            files.append([path+i[0],0])
            files.append([path+i[1],0])
    else:
        for i in metafile:
            files.append([path+i[0]+'_'+i[1],1])
            files.append([path+i[1]+'_'+i[0],1])
    return files

Celeb_test=list(map(lambda x:[os.path.join(celebroot,x[0]),x[1]],load_json('celeb.json')))

def jpeg_FF_dataset(tag='Origin',codec='c0',part='train'):
    assert(tag in ['Origin','Deepfakes','NeuralTextures','FaceSwap','Face2Face','FaceShifter'])
    assert(codec in ['c0','c23','c40','all'])
    assert(part in ['train','val','test','all'])
    if part=="all":
        return jpeg_FF_dataset(tag,codec,'train')+jpeg_FF_dataset(tag,codec,'val')+jpeg_FF_dataset(tag,codec,'test')
    if codec=='all':
        return jpeg_FF_dataset(tag,'c0',part)+jpeg_FF_dataset(tag,'c23',part)+jpeg_FF_dataset(tag,'c40',part)
    path=ffpproot+'%s/%s/jpeg_10_images/'%(tag,codec)
    metafile=load_json(ffpproot+'ffpp/'+part+'.json')
    files=[]
    if tag=='Origin':
        for i in metafile:
            files.append([path+i[0],0])
            files.append([path+i[1],0])
    else:
        for i in metafile:
            files.append([path+i[0]+'_'+i[1],1])
            files.append([path+i[1]+'_'+i[0],1])
    return files

def make_balance(data):
    tr=list(filter(lambda x:x[1]==0,data))
    tf=list(filter(lambda x:x[1]==1,data))
    if len(tr)>len(tf):
        tr,tf=tf,tr
    rate=len(tf)//len(tr)
    res=len(tf)-rate*len(tr)
    tr=tr*rate+random.sample(tr,res)
    return tr+tf


def dfdc_dataset(part='train'):
    assert(part in ['train','val','test'])
    lf=load_json(dfdcroot+'DFDC.json')
    if part=='train':
        path=dfdcroot+'dfdc/'
        files=make_balance(lf['train'])
    if part=='test':
        path=dfdcroot+'dfdc-test/'
        files=lf['test']
    if part=='val':
        path=dfdcroot+'dfdc-val/'
        files=lf['val']
    files=[[path+i[0],i[1]] for i in files]
    return files


def deeperforensics_dataset(part='train'):
    a=os.listdir(deeperforensics_root)
    d=dict()
    for i in a:
        d[i.split('_')[0]]=i
    l=lambda x:[deeperforensics_root+d[x],1]
    metafile=load_json(ffpproot+part+'.json')
    files=[]
    for i in metafile:
        files.append(l(i[0]))
        files.append(l(i[1]))
    return files

