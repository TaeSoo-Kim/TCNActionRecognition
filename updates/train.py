import sys
sys.path.append('/cm/shared/apps/python/3.6.0/lib/python3.6/site-packages/lmdb')
sys.path.append('/cm/shared/apps/python/3.6.0/lib/python3.6/site-packages/keras')
sys.path.append('/home-2/jhou16@jhu.edu/.local/lib/python3.6/site-packages') 
import Models
from keras.utils import np_utils
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2,l1
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
subject_split = 1

## SET UP THE DATA
if subject_split:
  if raw:
    data_root = '/home-2/jhou16@jhu.edu/data/nturgbd/subjects_split_raw/'
  else:
    data_root = '/home-2/jhou16@jhu.edu/data/nturgbd/subjects_split_rot_norm_quat/'
else:
  if raw:
    data_root = '/home-2/jhou16@jhu.edu/data/nturgbd/views_split_raw/'
  else:
    data_root = '/home-2/jhou16@jhu.edu/data/nturgbd/views_split_rot_norm_quat/'

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
# 0:TCN_simple, 1:TCN_plain, 2:TCN_resnet, 3:TCN_simple_resnet
model_choice = 2
out_dir_name = 'TCN_raw_resnet_L10'
activation = "relu"
optimizer = SGD(lr=lr, momentum=momentum, decay=0.0, nesterov=True)
#optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
dropout = 0.5
reg = l1(1.e-4)
## AUGMENTATION PARAMS
augment = 0
shift_limit = 10
factor_denom = 2

## TRAINING PARAMS
batch_size = 128
epochs = 200
verbose = 1
shuffle = 1

if subject_split:
  samples_per_epoch = 39889
  samples_per_validation = 16390
  if raw:
    train_x_mean = 0.590914884877
  else:
    train_x_mean = 0.189002480981
else:
  samples_per_epoch = 37462
  samples_per_validation = 18817
  if raw:
    train_x_mean = 0.582937574873               ##### need to be changed
  else:
    train_x_mean = 0.237390121854               ##### need to be changed

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
n_nodes = [64]    
conv_len = 8
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

  def __next__(self):
    with self.lock:
      return self.it.__next__()

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
    print(current_max)
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
    indices = list(range(0,samples_per_epoch))
    np.random.shuffle(indices)
    
    for index in indices:
      value = np.frombuffer(lmdb_cursor_x.get('{:0>8d}'.format(index).encode()))
      label = np.frombuffer(lmdb_cursor_y.get('{:0>8d}'.format(index).encode()))

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
      X[batch_count] =  x.reshape(max_len,feat_dim,1)
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
  
  X = np.zeros((batch_size,max_len,feat_dim,1))
  Y = np.zeros((batch_size,n_classes))
  batch_count = 0
  while True:
    indices = list(range(0,samples_per_validation))
    np.random.shuffle(indices)
    batch_count = 0
    for index in indices:
      value = np.frombuffer(lmdb_cursor_x.get('{:0>8d}'.format(index).encode()))
      label = np.frombuffer(lmdb_cursor_y.get('{:0>8d}'.format(index).encode()))
      
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



      X[batch_count] =  x.reshape(max_len,feat_dim,1)
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
  model_TCN_simple = Models.TCN_simple(
        n_classes,
        feat_dim,
        max_len,
        gap=1,
        dropout=dropout,
        kernel_regularizer=l2(1.e-4),
        activation=activation)
  model_TCN_plain = Models.TCN_plain(
        n_classes,
        feat_dim,
        max_len,
        gap=1,
        dropout=dropout,
        kernel_regularizer=reg,
        activation=activation)
  model_TCN_resnet = Models.TCN_resnet(
        n_classes,
        feat_dim,
        max_len,
        gap=1,
        dropout=dropout,
        kernel_regularizer=reg,
        activation=activation) 
  model_TCN_simple_resnet = Models.TCN_simple_resnet(
        n_classes,
        feat_dim,
        max_len,
        gap=1,
        dropout=dropout,
        kernel_regularizer=reg,
        activation=activation)   

  models = [model_TCN_simple, model_TCN_plain, model_TCN_resnet, model_TCN_simple_resnet]
  model = models[model_choice]
  
  model.compile(loss=loss,
                optimizer=optimizer,
                metrics=['accuracy'])

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
                      steps_per_epoch=num_training_samples/batch_size+1,
                      epochs=epochs,
                      verbose=1,
                      callbacks=callbacks_list,
                      validation_data=nturgbd_test_datagen(),
                      validation_steps=samples_per_validation/batch_size+1,
                      workers=1,
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

  indices = list(range(0,samples_per_epoch))
  np.random.shuffle(indices)

  total_count = 0
  total_sum = 0
  for index in indices:
    value = np.frombuffer(lmdb_cursor_x.get('{:0>8d}'.format(index).encode()))
    label = np.frombuffer(lmdb_cursor_y.get('{:0>8d}'.format(index).encode()))
    X.append(value)
    x = value.reshape((max_len,feat_dim))
    
    nonzeros = np.where(np.array([np.sum(x[i])>0 for i in range(0,x.shape[0])])==False)[0]
    
    if len(nonzeros) == 0:
      continue

    last_time = nonzeros[0]
    total_count += last_time
    total_sum += np.sum(x[:last_time])
 
    print(total_sum)

  final_mean = total_sum / total_count
  final_mean = final_mean / feat_dim
  print(total_count)
  print(final_mean)

if __name__ == "__main__":
  train()
