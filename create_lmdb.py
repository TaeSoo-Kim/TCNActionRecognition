import lmdb
import numpy as np
import deepdish as dd
from keras.utils import np_utils

import pdb



#basic setting
mode = ['train','test']
batch_size = 256
num_train_chunks = 18
num_test_chunks = 13

for m in mode:
  data_root = '/home/tk/dev/data/nturgbd/subjects_split/'
  lmdb_file_x = '/home/tk/dev/data/nturgbd/subjects_split/X%s_lmdb'%m
  lmdb_file_y = '/home/tk/dev/data/nturgbd/subjects_split/Y%s_lmdb'%m


  max_len = 300
  n_classes = 60

  # create the lmdb file
  lmdb_env_x = lmdb.open(lmdb_file_x, map_size=int(1e12))
  lmdb_env_y = lmdb.open(lmdb_file_y, map_size=int(1e12))
  lmdb_txn_x = lmdb_env_x.begin(write=True)
  lmdb_txn_y = lmdb_env_y.begin(write=True)



  item_id = -1
  if m == "train":
    count = 18
  else:
    count = 13

  for c in range(0,count):
    loaded = dd.io.load(data_root+'Xy_%s_%03d'%(m,c)+'.h5')
    print 'loaded %sing'%m, c

    #data = (loaded['X_train'],loaded['y_train'])
    X = loaded['X_%s'%m]
    y = loaded['y_%s'%m]
    Y = np_utils.to_categorical(y, n_classes)
    for i in range(0,len(X)):

      item_id += 1
      keystr = '{:0>8d}'.format(item_id)
      lmdb_txn_x.put( keystr, X[i].tobytes() )
      lmdb_txn_y.put( keystr, Y[i].tobytes() )

      # write batch
      if(item_id + 1) % batch_size == 0:
        lmdb_txn_x.commit()
        lmdb_txn_x = lmdb_env_x.begin(write=True)
        lmdb_txn_y.commit()
        lmdb_txn_y = lmdb_env_y.begin(write=True)
        print (item_id + 1)
      #pdb.set_trace()

  # write last batch
  if (item_id+1) % batch_size != 0:
    lmdb_txn_x.commit()
    lmdb_txn_y.commit()
    print 'last batch'
    print (item_id + 1)



