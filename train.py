"""
Train Res-TCN model NTURGBD skeleton dataset.
Pre-reqs: GPU mode, TensorFlow backend with Keras, plus all standard python libs such as numpy, lmdb

Tae Soo Kim
April, 2017
"""


import Models
from keras.utils import np_utils
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2,l1l2,l1
import pdb
import numpy as np
import lmdb
import threading
import os
from keras.callbacks import ReduceLROnPlateau


# seed 1234 is used for reproducibility
np.random.seed(seed=1234)

## raw means skeletons are not normalized as in the original Sharoudy et. al.
raw = 1

## 0:CrossView, 1:CrossSubject
subject_split = 1                                                             ## CHECK THIS!!!!!!!!!

## SET UP THE DATA
if subject_split:
  if raw:
    data_root = '/home-4/tkim60@jhu.edu/scratch/dev/nturgbd/data_raw/'        ## CHECK THIS!!!!!!!!!
  else:
    data_root = '/home-4/tkim60@jhu.edu/scratch/dev/nturgbd/data/'            ## CHECK THIS!!!!!!!!!
else:
  if raw:
    data_root = '/home-4/tkim60@jhu.edu/scratch/dev/nturgbd/data_cv_raw/'     ## CHECK THIS!!!!!!!!!
  else:
    data_root = '/home-4/tkim60@jhu.edu/scratch/dev/nturgbd/data_cv/'         ## CHECK THIS!!!!!!!!!

## LMDBs
lmdb_file_train_x = os.path.join(data_root,'Xtrain_lmdb')
lmdb_file_train_y = os.path.join(data_root,'Ytrain_lmdb')
lmdb_file_test_x = os.path.join(data_root,'Xtest_lmdb')
lmdb_file_test_y = os.path.join(data_root,'Ytest_lmdb')

## OPTIMIZER PARAMS
loss = 'categorical_crossentropy'
lr = 0.01
momentum = 0.9

## MODEL CHOICE
# 0: no downsample, 1: downsample, 2: multiscale, 3:resnet, 4:resnet_gap, 5:resnet_v2_gap, 6:resnet_v3_gap, 7:resnet_v4_gap
model_choice = 6                                                              ## CHECK THIS!!!!!!!!!
out_dir_name = 'tktcn_Dr0.5_L9_F8_resnet_v3_raw2'                             ## CHECK THIS!!!!!!!!!
activation = "relu"                                                           ## CHECK THIS!!!!!!!!!
optimizer = SGD(lr=lr, momentum=momentum, decay=0.0, nesterov=True)           ## CHECK THIS!!!!!!!!!
#optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)  ## CHECK THIS!!!!!!!!!
dropout = 0.5                                                                 ## CHECK THIS!!!!!!!!!
reg = l1(1.e-4)                                                               ## CHECK THIS!!!!!!!!!
## AUGMENTATION PARAMS
augment = 0                                                                   ## CHECK THIS!!!!!!!!!
shift_limit = 10
factor_denom = 2

## TRAINING PARAMS
batch_size = 126
nb_epoch = 200
verbose = 1
shuffle = 1

if subject_split:
  samples_per_epoch = 33094
  samples_per_validation = 23185
  if raw:
    train_x_mean = 0.60351
  else:
    train_x_mean = 0.197766
else:
  samples_per_epoch = 37462
  samples_per_validation = 18817
  if raw:
    train_x_mean = 0.582937574873
  else:
    train_x_mean = 0.237390121854

if augment:
  num_training_samples = samples_per_epoch*4
else:
  num_training_samples = samples_per_epoch

initial_epoch = 0

num_train_chunks = 18
num_test_chunks = 13

max_len = 300
#max_len = compute_max_len()



## SET UP THE MODEL
n_nodes = [128,256,512]                                                       ## CHECK THIS!!!!!!!!!    
conv_len = 25
feat_dim = 150


n_classes = 60


tr_chunks = list()
te_chunks = list()


class threadsafe_iter:
  """Takes an iterator/generator and makes it thread-safe by
  serializing call to the `next` method of given iterator/generator.
  """
  def __init__(self, it):
    self.it = it
    self.lock = threading.Lock()

  def __iter__(self):
    return self

  def next(self):
    with self.lock:
      return self.it.next()

def threadsafe_generator(f):
  """A decorator that takes a generator function and makes it thread-safe.
  """
  def g(*a, **kw):
      return threadsafe_iter(f(*a, **kw))
  return g

def compute_max_len():
  current_max = -1
  for i in range(0,num_train_chunks):
    loaded = dd.io.load(data_root+'Xy_train_%03d'%i+'.h5')
    x_tr = loaded['X_train']
    for x in x_tr:
      if len(x) > current_max:
        current_max = len(x)
    print current_max
  return current_max


@threadsafe_generator
def nturgbd_train_datagen(augmentation=1):
  lmdb_env_x = lmdb.open(lmdb_file_train_x)
  lmdb_txn_x = lmdb_env_x.begin()
  lmdb_cursor_x = lmdb_txn_x.cursor()

  lmdb_env_y = lmdb.open(lmdb_file_train_y)
  lmdb_txn_y = lmdb_env_y.begin()
  lmdb_cursor_y = lmdb_txn_y.cursor()
  
  X = np.zeros((batch_size,max_len,feat_dim))
  Y = np.zeros((batch_size,n_classes))
  batch_count = 0
  while True:
    indices = range(0,samples_per_epoch)
    np.random.shuffle(indices)
    
    for index in indices:
      value = np.frombuffer(lmdb_cursor_x.get('{:0>8d}'.format(index)))
      label = np.frombuffer(lmdb_cursor_y.get('{:0>8d}'.format(index)),dtype=np.float32)

      ## THIS IS MEAN SUBTRACTION
      x = value.reshape((max_len,feat_dim))
      nonzeros = np.where(np.array([np.sum(x[i])>0 for i in range(0,x.shape[0])])==False)[0]
      #value.reshape((max_len,feat_dim))
      if len(nonzeros) == 0:
        last_time = 0
      else:
        last_time = nonzeros[0]
      x.setflags(write=1)
      x[:last_time] = x[:last_time] - train_x_mean

      ## ORIGINAL
      X[batch_count] =  x
      Y[batch_count] = label
      batch_count += 1

      if augmentation:
        ## TEMPORAL SHIFT
        shift_range = np.random.randint(10)+1
        shift_augment_x = np.zeros(x.shape)
        end_point = min(shift_range+last_time,max_len)
        shift_augment_x[shift_range:end_point] = x[:last_time]
        X[batch_count] =  shift_augment_x
        Y[batch_count] = label
        batch_count += 1

        if batch_count == batch_size:
          ret_x = X
          ret_y = Y
          X = np.zeros((batch_size,max_len,feat_dim))
          Y = np.zeros((batch_size,n_classes))
          batch_count = 0
          yield (ret_x,ret_y)

        ## TEMPORAL STRETCH
        if last_time > 0:
          factor = np.random.random()/factor_denom + 1
          new_length = min(int(last_time*factor),max_len)
          stretch_augment_x = np.zeros(x.shape)
          stretched = np.resize(x[:last_time],(new_length,feat_dim))
          #stretched = cv2.resize(x[:last_time],(feat_dim,new_length),interpolation=cv2.INTER_LINEAR)
          stretch_augment_x[:new_length] = stretched
          X[batch_count] =  stretch_augment_x
          Y[batch_count] = label
          batch_count += 1

        if batch_count == batch_size:
          ret_x = X
          ret_y = Y
          X = np.zeros((batch_size,max_len,feat_dim))
          Y = np.zeros((batch_size,n_classes))
          batch_count = 0
          yield (ret_x,ret_y)

        ## TEMPORAL SHRINK
        if last_time > 0:
          factor = 1 - np.random.random()/factor_denom 
          new_length = max(int(last_time*factor),1)
          shrink_augment_x = np.zeros(x.shape)
          shrinked = np.resize(x[:last_time],(new_length,feat_dim))
          #shrinked = cv2.resize(x[:last_time],(feat_dim,new_length),interpolation=cv2.INTER_LINEAR)
          shrink_augment_x[:new_length] = shrinked
          X[batch_count] =  shrink_augment_x
          Y[batch_count] = label
          batch_count += 1

      if batch_count == batch_size:
        ret_x = X
        ret_y = Y
        X = np.zeros((batch_size,max_len,feat_dim))
        Y = np.zeros((batch_size,n_classes))
        batch_count = 0
        yield (ret_x,ret_y)


@threadsafe_generator
def nturgbd_test_datagen():
  lmdb_env_x = lmdb.open(lmdb_file_test_x)
  lmdb_txn_x = lmdb_env_x.begin()
  lmdb_cursor_x = lmdb_txn_x.cursor()

  lmdb_env_y = lmdb.open(lmdb_file_test_y)
  lmdb_txn_y = lmdb_env_y.begin()
  lmdb_cursor_y = lmdb_txn_y.cursor()
  
  X = np.zeros((batch_size,max_len,feat_dim))
  Y = np.zeros((batch_size,n_classes))
  batch_count = 0
  while True:
    indices = range(0,samples_per_validation)
    np.random.shuffle(indices)
    batch_count = 0
    for index in indices:
      value = np.frombuffer(lmdb_cursor_x.get('{:0>8d}'.format(index)))
      label = np.frombuffer(lmdb_cursor_y.get('{:0>8d}'.format(index)),dtype=np.float32)
      
      ## THIS IS MEAN SUBTRACTION
      x = value.reshape((max_len,feat_dim))
      nonzeros = np.where(np.array([np.sum(x[i])>0 for i in range(0,x.shape[0])])==False)[0]
      #value.reshape((max_len,feat_dim))
      if len(nonzeros) == 0:
        last_time = 0
      else:
        last_time = nonzeros[0]
      x.setflags(write=1)
      x[:last_time] = x[:last_time] - train_x_mean

      ##



      X[batch_count] =  x
      Y[batch_count] = label

      batch_count += 1

      if batch_count == batch_size:
        ret_x = X
        ret_y = Y
        X = np.zeros((batch_size,max_len,feat_dim))
        Y = np.zeros((batch_size,n_classes))
        batch_count = 0
        yield (ret_x,ret_y)


def train():
  
  model_vanila = Models.TK_TCN(n_nodes, 
         conv_len, 
         n_classes, 
         feat_dim,
         max_len=max_len,
         channel_wise=0,
         dropout=0.5,
         activation=activation)
  model_downsample = Models.TK_TCN_downsample(n_nodes, 
         conv_len, 
         n_classes, 
         feat_dim,
         max_len=max_len,
         gap=0,
         dropout=dropout,
         W_regularizer=reg,
         activation=activation)
  model_multiscale = Models.TK_TCN_downsample_multiscale( 
         n_classes, 
         feat_dim,
         max_len=300,
         dropout=dropout,
         gap=0,
         W_regularizer=reg, 
         activation=activation)
  model_resnet = Models.TK_TCN_resnet( 
         n_classes, 
         feat_dim,
         max_len=300,
         gap=0,
         dropout=dropout,
         W_regularizer=reg,
         activation=activation)
  model_resnet_gap = Models.TK_TCN_resnet( 
         n_classes, 
         feat_dim,
         max_len=300,
         gap=1,
         dropout=dropout,
         W_regularizer=reg,
         activation=activation)
  model_resnet_gap_v2 = Models.TK_TCN_resnet_v2( 
         n_classes, 
         feat_dim,
         max_len=300,
         gap=1,
         dropout=dropout,
         W_regularizer=reg,
         activation=activation)
  model_resnet_gap_v3 = Models.TK_TCN_resnet_v3( 
         n_classes, 
         feat_dim,
         max_len=300,
         gap=1,
         dropout=dropout,
         W_regularizer=reg,
         activation=activation)
  model_resnet_gap_v4 = Models.TK_TCN_resnet_v4( 
         n_classes, 
         feat_dim,
         max_len=300,
         gap=1,
         dropout=dropout,
         W_regularizer=reg,
         activation=activation)
  models = [model_vanila, model_downsample, model_multiscale, model_resnet,model_resnet_gap,model_resnet_gap_v2,model_resnet_gap_v3,model_resnet_gap_v4]
  model = models[model_choice]

  model.compile(loss=loss, 
                 optimizer=optimizer,  
                 metrics=['accuracy'])
                   #sample_weight_mode="temporal",

  if not os.path.exists('weights/'+out_dir_name):
    os.makedirs('weights/'+out_dir_name) 
  weight_path = 'weights/'+out_dir_name+'/{epoch:03d}_{val_acc:0.3f}.hdf5'
  checkpoint = ModelCheckpoint(weight_path, 
                               monitor='val_acc', 
                               verbose=1, 
                               save_best_only=True, mode='max')
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                factor=0.1,
                                patience=10, 
                                verbose=1,
                                mode='auto',
                                cooldown=3,
                                min_lr=0.0001)

  callbacks_list = [checkpoint,reduce_lr]


  model.fit_generator(nturgbd_train_datagen(augment),
                      samples_per_epoch=num_training_samples,
                      nb_epoch=nb_epoch,
                      verbose=1,
                      callbacks=callbacks_list,
                      validation_data=nturgbd_test_datagen(),
                      nb_val_samples=samples_per_validation,
                      nb_worker=1,
                      initial_epoch=0
                      )


def compute_dataset_mean():
  lmdb_env_x = lmdb.open(lmdb_file_train_x)
  lmdb_txn_x = lmdb_env_x.begin()
  lmdb_cursor_x = lmdb_txn_x.cursor()

  lmdb_env_y = lmdb.open(lmdb_file_train_y)
  lmdb_txn_y = lmdb_env_y.begin()
  lmdb_cursor_y = lmdb_txn_y.cursor()
  
  X = []
  batch_count = 0

  indices = range(0,samples_per_epoch)
  np.random.shuffle(indices)

  total_count = 0
  total_sum = 0
  for index in indices:
    value = np.frombuffer(lmdb_cursor_x.get('{:0>8d}'.format(index)))
    label = np.frombuffer(lmdb_cursor_y.get('{:0>8d}'.format(index)),dtype=np.float32)
    X.append(value)
    x = value.reshape((max_len,feat_dim))
    
    nonzeros = np.where(np.array([np.sum(x[i])>0 for i in range(0,x.shape[0])])==False)[0]
    
    if len(nonzeros) == 0:
      continue

    last_time = nonzeros[0]
    total_count += last_time
    total_sum += np.sum(x[:last_time])

    print total_sum

  final_mean = total_sum / total_count
  final_mean = final_mean / feat_dim
  print final_mean
  pdb.set_trace()


if __name__ == "__main__":
  train()
