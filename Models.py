import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, merge, Lambda
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.recurrent import *
from keras.regularizers import l2,l1l2,l1
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras import backend as K

from keras.activations import relu
from functools import partial


def channel_normalization(x):
  # Normalize by the highest activation
  max_values = K.max(K.abs(x), 2, keepdims=True)+1e-5
  out = x / max_values
  return out

def WaveNet_activation(x):
  tanh_out = Activation('tanh')(x)
  sigm_out = Activation('sigmoid')(x)  
  return Merge(mode='mul')([tanh_out, sigm_out])

def ED_TCN(n_nodes, conv_len, n_classes, n_feat, max_len, 
            loss='categorical_crossentropy', causal=False, 
            optimizer="rmsprop", activation='norm_relu',
            return_param_str=False):
  n_layers = len(n_nodes)

  inputs = Input(shape=(max_len,n_feat))
  model = inputs

  # ---- Encoder ----
  for i in range(n_layers):
    # Pad beginning of sequence to prevent usage of future data
    if causal: model = ZeroPadding1D((conv_len//2,0))(model)
    model = Convolution1D(n_nodes[i], conv_len, border_mode='same')(model)
    if causal: model = Cropping1D((0,conv_len//2))(model)

    model = SpatialDropout1D(0.3)(model)
    
    if activation=='norm_relu': 
      model = Activation('relu')(model)            
      model = Lambda(channel_normalization, name="encoder_norm_{}".format(i))(model)
    elif activation=='wavenet': 
      model = WaveNet_activation(model) 
    else:
      model = Activation(activation)(model)            
    
    model = MaxPooling1D(2)(model)

  # ---- Decoder ----
  for i in range(n_layers):
    model = UpSampling1D(2)(model)
    if causal: model = ZeroPadding1D((conv_len//2,0))(model)
    model = Convolution1D(n_nodes[-i-1], conv_len, border_mode='same')(model)
    if causal: model = Cropping1D((0,conv_len//2))(model)

    model = SpatialDropout1D(0.3)(model)

    if activation=='norm_relu': 
      model = Activation('relu')(model)
      model = Lambda(channel_normalization, name="decoder_norm_{}".format(i))(model)
    elif activation=='wavenet': 
      model = WaveNet_activation(model) 
    else:
      model = Activation(activation)(model)

  # Output FC layer
  model = TimeDistributed(Dense(n_classes, activation="softmax" ))(model)

  model = Model(input=inputs, output=model)
  model.compile(loss=loss, optimizer=optimizer, sample_weight_mode="temporal", metrics=['accuracy'])

  if return_param_str:
    param_str = "ED-TCN_C{}_L{}".format(conv_len, n_layers)
    if causal:
      param_str += "_causal"

    return model, param_str
  else:
    return model

def TK_TCN(n_nodes, 
           conv_len, 
           n_classes, 
           feat_dim,
           max_len,
           channel_wise=1,
           dropout=0.0,
           W_regularizer=l2(1.e-4),
           activation="relu"):
  
  
  if K.image_dim_ordering() == 'tf':
    ROW_AXIS = 1
    CHANNEL_AXIS = 2
  else:
    ROW_AXIS = 2
    CHANNEL_AXIS = 1

  input = Input(shape=(max_len,feat_dim))
  model = input

  ## CONV LAYERS
  for width in n_nodes:
    model = Convolution1D(width, 
                          conv_len,
                          init="he_normal",
                          border_mode="same",
                          W_regularizer=W_regularizer)(model)

    model = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(model)
    model = Activation(activation)(model)

    model = Dropout(dropout)(model)

  ## CLASSIFER 
  if channel_wise:
    dense = TimeDistributed(Dense(1, activation="softmax" ))(model)
    dense = Flatten()(dense)
    dense = Dense(output_dim=n_classes,
                  init="he_normal",
                  activation="softmax")(dense)
  else:
    flatten = Flatten()(model)
    dense = Dense(output_dim=n_classes,
          init="he_normal",
          activation="softmax")(flatten)
  
    
  
  model = Model(input=input, output=dense)
  
  return model
    
def TK_TCN_downsample(n_nodes, 
           conv_len, 
           n_classes, 
           feat_dim,
           max_len,
           gap=1,
           dropout=0.0,
           W_regularizer=l2(1.e-4),
           activation="relu"):
  
  
  if K.image_dim_ordering() == 'tf':
    ROW_AXIS = 1
    CHANNEL_AXIS = 2
  else:
    ROW_AXIS = 2
    CHANNEL_AXIS = 1

  input = Input(shape=(max_len,feat_dim))
  model = input

  ## CONV LAYERS
  prev_width = n_nodes[0]
  for width in n_nodes:

    if width == prev_width:
      stride = 1
    else:
      stride = 2

    model = Convolution1D(width, 
                          conv_len,
                          init="he_normal",
                          border_mode="same",
                          subsample_length=stride,
                          W_regularizer=W_regularizer)(model)

    model = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(model)
    model = Activation(activation)(model)

    model = Dropout(dropout)(model)

    prev_width = width

  ## CLASSIFER 
    
  if gap:
    pool_window_shape = K.int_shape(model)
    gap = AveragePooling1D(pool_window_shape[ROW_AXIS],
                           stride=1)(model)
    flatten = Flatten()(gap)
  else:
    flatten = Flatten()(model)
  dense = Dense(output_dim=n_classes,
        init="he_normal",
        activation="softmax")(flatten)
  model = Model(input=input, output=dense)
  
  return model

def TK_TCN_downsample_pool(n_nodes, 
           conv_len, 
           n_classes, 
           feat_dim,
           max_len,
           channel_wise=1,
           dropout=0.0,
           W_regularizer=l2(1.e-4),
           activation="relu"):
  
  
  if K.image_dim_ordering() == 'tf':
    ROW_AXIS = 1
    CHANNEL_AXIS = 2
  else:
    ROW_AXIS = 2
    CHANNEL_AXIS = 1

  input = Input(shape=(max_len,feat_dim))
  model = input

  ## CONV LAYERS
  prev_width = n_nodes[0]
  for width in n_nodes:

    if width == prev_width:
      down_sample = 1
    else:
      down_sample = 2

    model = Convolution1D(width, 
                          conv_len,
                          init="he_normal",
                          border_mode="same",
                          W_regularizer=W_regularizer)(model)
    if down_sample != 1:
      model = MaxPooling1D(pool_length=down_sample, stride=None, border_mode='valid')(model)

    model = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(model)
    model = Activation(activation)(model)
    model = Dropout(dropout)(model)

    prev_width = width

  ## CLASSIFER 
  if channel_wise:
    dense = TimeDistributed(Dense(1, activation="softmax" ))(model)
    dense = Flatten()(dense)
    dense = Dense(output_dim=n_classes,
                  init="he_normal",
                  activation="softmax")(dense)
  else:
    flatten = Flatten()(model)
    dense = Dense(output_dim=n_classes,
          init="he_normal",
          activation="softmax")(flatten)
  
    
  
  model = Model(input=input, output=dense)
  
  return model

def TK_TCN_downsample_multiscale(
           n_classes, 
           feat_dim,
           max_len,
           gap=1,
           dropout=0.0,
           W_regularizer=l1(1.e-4),
           activation="relu"):


  if K.image_dim_ordering() == 'tf':
    ROW_AXIS = 1
    CHANNEL_AXIS = 2
  else:
    ROW_AXIS = 2
    CHANNEL_AXIS = 1

  input = Input(shape=(max_len,feat_dim))
  model = input

  config = [ [(1,8,64), (1,16,64)],
             [(2,8,128), (2,16,128)],
             [(2,8,256), (2,16,256)]
           ]

           
  for depth in range(0,len(config)):
    blocks = []
    for stride,filter_dim,num in config[depth]:
      conv = Convolution1D(num, 
                          filter_dim,
                          init="he_normal",
                          border_mode="same",
                          subsample_length=stride,
                          W_regularizer=W_regularizer)(model)
      bn = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(conv)
      relu = Activation(activation)(bn)
      dr = Dropout(dropout)(relu)
      blocks.append(dr)
    model = merge(blocks,mode='concat',concat_axis=CHANNEL_AXIS)

  ## CLASSIFER 
  if gap:
    pool_window_shape = K.int_shape(model)
    gap = AveragePooling1D(pool_window_shape[ROW_AXIS],
                           stride=1)(model)
    flatten = Flatten()(gap)
  else:
    flatten = Flatten()(model)
  dense = Dense(output_dim=n_classes,
          init="he_normal",
          activation="softmax")(flatten)  
    
  
  model = Model(input=input, output=dense)
  return model
 
def TK_TCN_resnet(
           n_classes, 
           feat_dim,
           max_len=300,
           gap=1,
           dropout=0.0,
           W_regularizer=l1(1.e-4),
           activation="relu"):
  if K.image_dim_ordering() == 'tf':
    ROW_AXIS = 1
    CHANNEL_AXIS = 2
  else:
    ROW_AXIS = 2
    CHANNEL_AXIS = 1
  
  initial_conv_len = 8
  initial_conv_num = 64
  """
  config = [ 
             [(1,8,64)],
             [(1,8,64)],
             [(1,8,64)],
             [(2,8,128)],
             [(1,8,128)],
             [(1,8,128)],
             [(2,8,256)],
             [(1,8,256)],
             [(1,8,256)],
           ]
  """
  config = [ 
             [(1,8,64)],
             [(1,8,64)],
             [(1,8,64)],
             [(1,8,64)],
             [(1,8,64)],
             [(1,8,64)],
             [(2,8,128)],
             [(1,8,128)],
             [(1,8,128)],
             [(1,8,128)],
             [(1,8,128)],
             [(1,8,128)],
             [(2,8,256)],
             [(1,8,256)],
             [(1,8,256)],
             [(1,8,256)],
             [(1,8,256)],
             [(1,8,256)],
           ]


  input = Input(shape=(max_len,feat_dim))
  model = input

  model = Convolution1D(initial_conv_num, 
                              initial_conv_len,
                              init="he_normal",
                              border_mode="same",
                              subsample_length=1,
                              W_regularizer=W_regularizer)(model)

  for depth in range(0,len(config)):
    blocks = []
    for stride,filter_dim,num in config[depth]:
      ## residual block
      bn = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(model)
      relu = Activation(activation)(bn)
      dr = Dropout(dropout)(relu)
      conv = Convolution1D(num, 
                              filter_dim,
                              init="he_normal",
                              border_mode="same",
                              subsample_length=stride,
                              W_regularizer=W_regularizer)(dr)
      #dr = Dropout(dropout)(conv)


      ## potential downsample
      conv_shape = K.int_shape(conv)
      model_shape = K.int_shape(model)
      if conv_shape[CHANNEL_AXIS] != model_shape[CHANNEL_AXIS]:
        model = Convolution1D(num, 
                              1,
                              init="he_normal",
                              border_mode="same",
                              subsample_length=2,
                              W_regularizer=W_regularizer)(model)

      ## merge block
      model = merge([model,conv],mode='sum',concat_axis=CHANNEL_AXIS)

  ## final bn+relu
  bn = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(model)
  model = Activation(activation)(bn)


  if gap:
    pool_window_shape = K.int_shape(model)
    gap = AveragePooling1D(pool_window_shape[ROW_AXIS],
                           stride=1)(model)
    flatten = Flatten()(gap)
  else:
    flatten = Flatten()(model)

  dense = Dense(output_dim=n_classes,
        init="he_normal",
        activation="softmax")(flatten)
  model = Model(input=input, output=dense)
  return model

def TK_TCN_resnet_v2(
           n_classes, 
           feat_dim,
           max_len,
           gap=1,
           dropout=0.0,
           W_regularizer=l1(1.e-4),
           activation="relu"):
  if K.image_dim_ordering() == 'tf':
    ROW_AXIS = 1
    CHANNEL_AXIS = 2
  else:
    ROW_AXIS = 2
    CHANNEL_AXIS = 1
  

  config = [ 
             [(1,8,64)],
             [(2,8,128)],
             [(2,8,256)],
           ]
  initial_conv_len = 8
  initial_conv_num = 64

  input = Input(shape=(max_len,feat_dim))
  model = input

  model = Convolution1D(initial_conv_num, 
                              initial_conv_len,
                              init="he_normal",
                              border_mode="same",
                              subsample_length=1,
                              W_regularizer=W_regularizer)(model)

  for depth in range(0,len(config)):
    blocks = []
    for stride,filter_dim,num in config[depth]:
      ## residual block
      bn_a = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(model)
      relu_a = Activation(activation)(bn_a)
      dr_a = Dropout(dropout)(relu_a)
      conv_a = Convolution1D(num, 
                              filter_dim,
                              init="he_normal",
                              border_mode="same",
                              subsample_length=1,
                              W_regularizer=W_regularizer)(dr_a)

      bn_b = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(conv_a)
      relu_b = Activation(activation)(bn_b)
      dr_b = Dropout(dropout)(relu_b)
      conv_b = Convolution1D(num, 
                              filter_dim,
                              init="he_normal",
                              border_mode="same",
                              subsample_length=1,
                              W_regularizer=W_regularizer)(dr_b)

      bn_c = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(conv_b)
      relu_c = Activation(activation)(bn_c)
      dr_c = Dropout(dropout)(relu_c)
      conv_c = Convolution1D(num, 
                              filter_dim,
                              init="he_normal",
                              border_mode="same",
                              subsample_length=stride,
                              W_regularizer=W_regularizer)(dr_c)
      #dr = Dropout(dropout)(conv)


      ## potential downsample
      conv_shape = K.int_shape(conv_c)
      model_shape = K.int_shape(model)
      if conv_shape[CHANNEL_AXIS] != model_shape[CHANNEL_AXIS]:
        model = Convolution1D(num, 
                              1,
                              init="he_normal",
                              border_mode="same",
                              subsample_length=stride,
                              W_regularizer=W_regularizer)(model)

      ## merge block
      model = merge([model,conv_c],mode='sum',concat_axis=CHANNEL_AXIS)

  ## final bn+relu
  bn = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(model)
  model = Activation(activation)(bn)


  if gap:
    pool_window_shape = K.int_shape(model)
    gap = AveragePooling1D(pool_window_shape[ROW_AXIS],
                           stride=1)(model)
    flatten = Flatten()(gap)
  else:
    flatten = Flatten()(model)
  dense = Dense(output_dim=n_classes,
        init="he_normal",
        activation="softmax")(flatten)
  model = Model(input=input, output=dense)
  return model

def TK_TCN_resnet_v3(
           n_classes, 
           feat_dim,
           max_len,
           gap=1,
           dropout=0.0,
           W_regularizer=l1(1.e-4),
           activation="relu"):

  if K.image_dim_ordering() == 'tf':
    ROW_AXIS = 1
    CHANNEL_AXIS = 2
  else:
    ROW_AXIS = 2
    CHANNEL_AXIS = 1
  
  
  config = [ 
             [[(1,8,32)],[(1,16,32)]],
             [[(1,8,32)],[(1,16,32)]],
             [[(1,8,32)],[(1,16,32)]],
             [[(2,8,64)],[(2,16,64)]],
             [[(1,8,64)],[(1,16,64)]],
             [[(1,8,64)],[(1,16,64)]],
             [[(2,8,128)],[(2,16,128)]],
             [[(1,8,128)],[(1,16,128)]],
             [[(1,8,128)],[(1,16,128)]],
           ]
  """
  config = [ 
             [[(1,8,32)],[(1,16,32)]],
             [[(1,8,32)],[(1,16,32)]],
             [[(1,8,32)],[(1,16,32)]],
             [[(1,8,32)],[(1,16,32)]],
             [[(1,8,32)],[(1,16,32)]],
             [[(1,8,32)],[(1,16,32)]],
             [[(2,8,64)],[(2,16,64)]],
             [[(1,8,64)],[(1,16,64)]],
             [[(1,8,64)],[(1,16,64)]],
             [[(1,8,64)],[(1,16,64)]],
             [[(1,8,64)],[(1,16,64)]],
             [[(1,8,64)],[(1,16,64)]],
             [[(2,8,128)],[(2,16,128)]],
             [[(1,8,128)],[(1,16,128)]],
             [[(1,8,128)],[(1,16,128)]],
             [[(1,8,128)],[(1,16,128)]],
             [[(1,8,128)],[(1,16,128)]],
             [[(1,8,128)],[(1,16,128)]],
           ]
  """
  initial_conv_len = 8
  initial_conv_num = 64

  input = Input(shape=(max_len,feat_dim))
  model = input

  model = Convolution1D(initial_conv_num, 
                              initial_conv_len,
                              init="he_normal",
                              border_mode="same",
                              subsample_length=1,
                              W_regularizer=W_regularizer)(model)

  for depth in range(0,len(config)):
    blocks = []
    num_filters_this_layer = 0
    bn = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(model)
    relu = Activation(activation)(bn)
    dr = Dropout(dropout)(relu)
    for multiscale in config[depth]:
      for stride,filter_dim,num in multiscale:
        ## residual block
        conv = Convolution1D(num, 
                                filter_dim,
                                init="he_normal",
                                border_mode="same",
                                subsample_length=stride,
                                W_regularizer=W_regularizer)(dr)
        blocks.append(conv)
        num_filters_this_layer += num
    res = merge(blocks,mode='concat',concat_axis=CHANNEL_AXIS)
      #dr = Dropout(dropout)(conv)


    ## potential downsample
    conv_shape = K.int_shape(res)
    model_shape = K.int_shape(model)
    if conv_shape[CHANNEL_AXIS] != model_shape[CHANNEL_AXIS]:
      model = Convolution1D(num_filters_this_layer, 
                            1,
                            init="he_normal",
                            border_mode="same",
                            subsample_length=stride,
                            W_regularizer=W_regularizer)(model)

    ## merge block
    model = merge([model,res],mode='sum',concat_axis=CHANNEL_AXIS)

  ## final bn+relu
  bn = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(model)
  model = Activation(activation)(bn)


  if gap:
    pool_window_shape = K.int_shape(model)
    gap = AveragePooling1D(pool_window_shape[ROW_AXIS],
                           stride=1)(model)
    flatten = Flatten()(gap)
  else:
    flatten = Flatten()(model)
  dense = Dense(output_dim=n_classes,
        init="he_normal",
        activation="softmax")(flatten)
  model = Model(input=input, output=dense)
  return model

def TK_TCN_resnet_v4(
           n_classes, 
           feat_dim,
           max_len,
           gap=1,
           dropout=0.0,
           W_regularizer=l1(1.e-4),
           activation="relu"):

  if K.image_dim_ordering() == 'tf':
    ROW_AXIS = 1
    CHANNEL_AXIS = 2
  else:
    ROW_AXIS = 2
    CHANNEL_AXIS = 1
  

  config = [ 
             [(1,8,150)],
             [(1,8,150)],
             [(1,8,150)],
           ]
  initial_conv_len = 8
  initial_conv_num = 64

  input = Input(shape=(max_len,feat_dim))
  model = input

  #model = Convolution1D(initial_conv_num, 
  #                            initial_conv_len,
  #                            init="he_normal",
  #                            border_mode="same",
  #                            subsample_length=1,
  #                            W_regularizer=W_regularizer)(model)

  for depth in range(0,len(config)):
    blocks = []
    for stride,filter_dim,num in config[depth]:
      ## residual block
      conv = Convolution1D(num, 
                              filter_dim,
                              init="he_normal",
                              border_mode="same",
                              subsample_length=stride,
                              W_regularizer=W_regularizer)(model)

      ## potential downsample
      conv_shape = K.int_shape(conv)
      model_shape = K.int_shape(model)
      if conv_shape[CHANNEL_AXIS] != model_shape[CHANNEL_AXIS]:
        model = Convolution1D(num, 
                              1,
                              init="he_normal",
                              border_mode="same",
                              subsample_length=stride,
                              W_regularizer=W_regularizer)(model)

      ## merge block
      model = merge([model,conv],mode='sum',concat_axis=CHANNEL_AXIS)
    bn = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(model)
    nonlin = Activation(activation)(bn)
    model = Dropout(dropout)(nonlin)

  if gap:
    pool_window_shape = K.int_shape(model)
    gap = AveragePooling1D(pool_window_shape[ROW_AXIS],
                           stride=1)(model)
    flatten = Flatten()(gap)
  else:
    flatten = Flatten()(model)
  dense = Dense(output_dim=n_classes,
        init="he_normal",
        activation="softmax")(flatten)
  model = Model(input=input, output=dense)
  return model
