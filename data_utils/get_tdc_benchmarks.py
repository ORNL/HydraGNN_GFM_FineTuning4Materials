import pandas as pd
import numpy as np

from tdc import utils
from tdc.benchmark_group import admet_group

import yaml

names = utils.retrieve_benchmark_names('ADMET_Group')

total_num_molecules = 0
for name in names:
    group = admet_group()
    benchmark = group.get(name)
    train_val = benchmark['train_val']
    # split into train and val
    msk = np.random.rand(len(train_val)) < 0.8
    train = train_val[msk] 
    train['split'] = 'train'
    val = train_val[~msk]
    val['split'] = 'val'
    test = benchmark['test']
    test['split'] = 'test'
    df = pd.concat([train,val,test],ignore_index=True)
    df.drop('Drug_ID',inplace=True, axis=1)
    df.rename({'Drug':'smiles','Y':f'{name}'}, axis=1, inplace=True)
    if ~df[f'{name}'].isin([0.0, 1.0]).all():
        task = 'regression'
    else:
        task = 'binary'

    path = "../datasets/tdc/"
    
    df.to_csv(path+name+'.csv')
    total_num_molecules += len(df)

    meta = {'name':name, 'smiles':'smiles', 'graph_tasks':[{'name':name,'type':task}],
            'split':'split','source':'https://tdcommons.ai'}
    with open(path+name+'.yaml','w') as f: 
        yaml.dump(meta, f, default_flow_style=False)
    
    print(name,task) 

print(total_num_molecules)
