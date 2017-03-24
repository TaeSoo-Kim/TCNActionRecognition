import Models
import pdb

import matplotlib.pyplot as plt
import numpy as np
import os

from process_skeleton import classes,normalize_skeleton, vids_with_missing_skeletons, draw_skeleton
from keras import backend as K
import imageio


n_nodes = [128,128,256,256,512,512]
conv_len = 25
n_classes = 60
feat_dim = 150
max_len = 300

weights = '/home/tk/dev/tksrc/icu/weights/tk_tcn_xyz_d50_c25_128x2_256x2_512x2/105_0.592.hdf5'

def to_joint_dict(feature):
  # assumes 150-D feature,
  joints1 = {}
  joints2 = {}
  step = 3
  if feat_dim == 350:
    step = 7

  joint_id = 0

  body1 = feature[:feat_dim/2]
  body2 = feature[feat_dim/2:]
  for index in range(0,feat_dim/2,step):
    joints1[joint_id] = body1[index:index+step]
    joint_id += 1
  joint_id = 0
  for index in range(0,feat_dim/2,step):
    joints2[joint_id] = body2[index:index+step]
    joint_id += 1
  return joints1, joints2

def extract_feature_from_file_index(indx):
  feat_dim = 150
  bad_files = vids_with_missing_skeletons()
  skeleton_dir_root = "/home/tk/dev/data/nturgbd/nturgb+d_skeletons"
  skeleton_files = os.listdir(skeleton_dir_root)
  file_name = skeleton_files[indx]

  if file_name in bad_files:
      return None
  action_class = int(file_name[file_name.find('A')+1:file_name.find('A')+4])
  subject_id = int(file_name[file_name.find('S')+1:file_name.find('S')+4])

  sf = open(os.path.join(skeleton_dir_root,file_name),'r')
  num_frames = int(sf.readline())
  #vid_info = dict() ## key=frame_num,  value=body info dicts

  feature = np.zeros((num_frames, feat_dim))
  for n in range(0,num_frames):
    body_count = int(sf.readline())
    #print body_count

    if body_count > 2:
      return None
    else:
      
      binfo = dict()
      norm_dist = 0
      anchor = None
      right_to_left = None
      spine_to_top = None
      for b in range(0,body_count):
        body_info = sf.readline()
        bsp = body_info.split()
        
        body_id = bsp[0]
        cliped_edges = bsp[1]
        lefthand_confidence = bsp[2]
        lefthand_state = bsp[3]
        righthand_confidence = bsp[4]
        righthand_state = bsp[5]
        is_restricted = bsp[6]
        lean_x = bsp[7]
        lean_y = bsp[8]
        body_tracking_state = bsp[9]

        #binfo[b] = bsp
        joint_count = int(sf.readline()) ## ASSUMING THIS IS ALWAYS 25


        jinfo = dict()
        
        for j in range(0,joint_count):
          joint_info = sf.readline()
          jsp = joint_info.split()
          x = float(jsp[0])
          y = float(jsp[1])
          z = float(jsp[2])
          depth_x = float(jsp[3])
          depth_y = float(jsp[4])
          rgb_x = float(jsp[5])
          rgb_y = float(jsp[6])
          rw = float(jsp[7])
          rx = float(jsp[8])
          ry = float(jsp[9])
          rz = float(jsp[10])
          joint_tracking_state = jsp[11]

          jinfo[j] = (x,y,z,rw,rx,ry,rz)
        ## END JOINT LOOP
        
        
        norm_jinfo,anchor,norm_dist,right_to_left,spine_to_top = normalize_skeleton(jinfo,anchor=anchor,norm_dist=norm_dist,right_to_left=right_to_left,spine_to_top=spine_to_top)
        
        binfo[b] = norm_jinfo
    
      ## END BODY LOOP
      sample_ind = 0
      sample = np.zeros(feat_dim)
      
      ## CONSTRUCT THE FEATURE FOR THIS N-th FRAME
      for bind, body in binfo.iteritems():
        for jind, joint in body.iteritems():
          sample[sample_ind] =   joint[0] #x
          sample_ind += 1
          sample[sample_ind] =   joint[1] #y
          sample_ind += 1
          sample[sample_ind] =   joint[2] #z
          sample_ind += 1
      feature[n] = sample
  return feature,action_class-1,classes[action_class-1],os.path.join(skeleton_dir_root,file_name)

def get_activations(model, layer, X_batch):
  get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
  activations = get_activations([X_batch,0])
  return activations

def vis_filters():
  
  model = Models.TK_TCN(n_nodes, 
           conv_len, 
           n_classes, 
           feat_dim,
           max_len=max_len,
           channel_wise=0,
           dropout=0.5,
           activation="relu")
  model.load_weights(weights)

  conv1 = model.layers[1]

  conv1_weights = conv1.get_weights()[0][:,0]
  conv1_b = conv1.get_weights()[1]

  num_filters = conv1_weights.shape[-1]


  #for filter_ind in range(0,num_filters):
    #filter = conv1_weights[:,:,filter_ind]
#    plt.imshow(np.transpose(filter),cmap="jet")
#    plt.savefig("./filter_vis/tk_tcn_xyz_d50_c25_128x2_256x2_512x2/conv1_"+str(filter_ind)+"png",bbox_inches='tight')
#    top_values = sorted(filter.flatten())[-3:]
#    temporal_max = []
#    response_max  = []
#    for val in top_values:
#      max_time, max_dim =  np.where(filter == val)
#      temporal_max.append(max_time[0])
#      response_max.append(max_dim[0])

    #pdb.set_trace()

  #filter1 = conv1_weights[:,:,1]
  filter = conv1_weights[:,:,0]
  for frame in range(0,filter.shape[0]):
    joints1,joints2 = to_joint_dict(filter[frame])  
    draw_skeleton([joints1,joints2],show=1,save=0,autoscale=1)
    pdb.set_trace()

def vis_skeleton_movement():
  model = Models.TK_TCN(n_nodes, 
           conv_len, 
           n_classes, 
           feat_dim,
           max_len=max_len,
           channel_wise=0,
           dropout=0.5,
           activation="relu")
  model.load_weights(weights)

  layer_ind = 1 # 1- conv1, 3-relu1
  
  x,y,y_name, file_path = extract_feature_from_file_index(0)
  X = np.zeros((max_len,feat_dim))
  X[:x.shape[0]] = x
  #pdb.set_trace()
  X = X.reshape((1,X.shape[0],X.shape[1]))
  prediction = np.argmax(model.predict(X))

  print "Ground Truth: ", y,"(%s)"%y_name
  print "Prediction: ", prediction,"(%s)"%classes[prediction]

  conv1_filters = model.layers[1].get_weights()[0][:,0]
  conv1_activation = get_activations(model,layer_ind, X)[0]
  
  new_set = 0
  if new_set:
    outdir = "./tmp/"
    for frame in range(0,x.shape[0]):
      joints1,joints2 = to_joint_dict(x[frame])
      draw_skeleton([joints1,joints2],show=0,save=1,outdir=outdir,outname=str(frame))
    
    files = os.listdir(outdir)
    files = sorted(files,key=lambda x: int(x[:x.find('.')]))
    images = []
    for fn in files:
      images.append(imageio.imread(outdir+fn))
    imageio.mimsave(outdir+"vis.gif",images)
  

def activate_joint_by_filter():
  model = Models.TK_TCN(n_nodes, 
           conv_len, 
           n_classes, 
           feat_dim,
           max_len=max_len,
           channel_wise=0,
           dropout=0.5,
           activation="relu")
  model.load_weights(weights)

  layer_ind = 1 # 1- conv1, 3-relu1
  
  x,y,y_name, file_path = extract_feature_from_file_index(0)
  X = np.zeros((max_len,feat_dim))
  X[:x.shape[0]] = x
  #pdb.set_trace()
  X = X.reshape((1,X.shape[0],X.shape[1]))
  prediction = np.argmax(model.predict(X))

  print "Ground Truth: ", y,"(%s)"%y_name
  print "Prediction: ", prediction,"(%s)"%classes[prediction]

  conv1_filters = model.layers[1].get_weights()[0][:,0]
  conv1_activation = get_activations(model,layer_ind, X)[0]

  # Lets just look at the first body, 
  initial_config = to_joint_dict(x[0])[0]
  scale = 2
  ## MOVE INITIAL_CONFIG BY CONV1 FILTER
  for f in range(0,conv1_filters.shape[-1]):
    outdir = "./tmp/" + str(f) + "/" 
    if not os.path.exists(outdir):
      os.makedirs(outdir)
    a_conv1_filter = conv1_filters[:,:,f]
    for t in range(0,a_conv1_filter.shape[0]):
      new_joint_pos = {}
      for jnum, pos in initial_config.iteritems():
        delta_pos = a_conv1_filter[t][(jnum*3):(jnum*3+3)]*pos*scale
        new_joint_pos[jnum] = pos + delta_pos
      draw_skeleton([new_joint_pos],show=0,save=1,outdir=outdir,outname=str(t))
      plt.close("all")

if __name__ == "__main__":
  #vis_filters()
  #vis_skeleton_movement()
  activate_joint_by_filter()