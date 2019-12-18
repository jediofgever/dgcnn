import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from model import *
import indoor3d_util
from plyfile import PlyData, PlyElement
import numpy as np
import open3d as o3d
import copy
import rospy


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--model_path', required=True, help='model checkpoint file path')
parser.add_argument('--dump_dir', required=True, help='dump folder path')
parser.add_argument('--output_filelist', required=True, help='TXT filename, filelist, each line is an output for a room')
parser.add_argument('--room_data_filelist' , help='TXT filename, filelist, each line is a test room data label file.')
parser.add_argument('--no_clutter', action='store_true', help='If true, donot count the clutter class')
parser.add_argument('--visu', action='store_true', help='Whether to output OBJ file for prediction visualization.')
parser.add_argument('--inference_dir',type= str, default="/home/atas/catkin_ws/src/ROS_FANUC_LRMATE200ID/inference/",help="directory where ply files are dumped by ROS")

FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
DUMP_DIR = FLAGS.dump_dir
INFERENCE_DIR = FLAGS.inference_dir

if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
#ROOM_PATH_LIST = [os.path.join(ROOT_DIR,line.rstrip()) for line in open(FLAGS.room_data_filelist)]

NUM_CLASSES = 2

def log_string(out_str):
  LOG_FOUT.write(out_str+'\n')
  LOG_FOUT.flush()
  print(out_str)

def room2blocks_plus_normalized(data_label, num_point, block_size, stride,
                random_sample, sample_num, sample_aug):
  """ room2block, with input filename and RGB preprocessing.
    for each block centralize XYZ, add normalized XYZ as 678 channels
  """

 

  new_data_batch = np.zeros((1, num_point, 9))
  new_label_batch = np.zeros((1,num_point))  

  print(new_label_batch.shape)

  new_data_batch[:, :, 0:9] = data_label[:,0:9]

  for i in range(0,len(data_label[0])):
    new_label_batch[0,i] = data_label[i,9:10]
  
  return new_data_batch, new_label_batch



def ply2npy(ply_file_path):
  test_pcd_frame_as_npy_array = np.zeros((NUM_POINT,10),dtype=float)

  plydata = PlyData.read(ply_file_path)

  for j in range(0, NUM_POINT): 
    label = 0
    if(plydata['vertex']['blue'][j] > 0 or plydata['vertex']['red'][j] > 0 or plydata['vertex']['green'][j] > 0 ):
        label = 1

    else:
        label = 0
                        

    MAX_X, MAX_, MAX_Z = 4,4,4
    MIN_X, MIN_Y, MIN_Z = -4,-4,-4
 
    test_pcd_frame_as_npy_array[j] = [plydata['vertex']['x'][j], plydata['vertex']['y'][j], plydata['vertex']['z'][j], 
                  plydata['vertex']['red'][j]/255.0, plydata['vertex']['green'][j]/255.0,plydata['vertex']['blue'][j]/255.0, 
                 (plydata['vertex']['x'][j]-MIN_X)/8.0,(plydata['vertex']['y'][j]-MIN_Y)/8.0,(plydata['vertex']['z'][j]-MIN_Z)/8.0,label]
  return test_pcd_frame_as_npy_array




def room2blocks_wrapper_normalized(data_label_npy, num_point, block_size=1.0, stride=1.0,
                   random_sample=False, sample_num=None, sample_aug=1):
  data_label = data_label_npy

  return room2blocks_plus_normalized(data_label, num_point, block_size, stride,
                     random_sample, sample_num, sample_aug)
 

 
from sensor_msgs.msg import PointCloud2 
import sensor_msgs.point_cloud2 as pc2
class PointNet_Ros_Node:
  def __init__(self):
    '''initiliaze  ros stuff '''
    self.cloud_pub = rospy.Publisher("output/pointnet/segmented",PointCloud2)
    self.cloud_sub = rospy.Subscriber("/corrected_cloud_ros_frame_convention", PointCloud2,self.callback,queue_size=1)
  
  def pointcloud2_to_array(self,cloud_msg):
    ''' 
    Converts a rospy PointCloud2 message to a numpy recordarray 
    
    Assumes all fields 32 bit floats, and there is no padding.
    '''
    dtype_list = [(f.name, np.float32) for f in cloud_msg.fields]
    cloud_arr = np.fromstring(cloud_msg.data, dtype_list)
    return  cloud_arr  
  
  def callback(self, ros_point_cloud):
    xyz = np.array([[0,0,0]])

    for p in pc2.read_points(ros_point_cloud, field_names = ("x", "y", "z"), skip_nans=True):
      n = np.array([p[0],p[1],p[2]])
      xyz = np.append(xyz,[[p[0],p[1],p[2]]], axis = 0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    #
    # pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud("/home/atas/test.ply", pcd)
    self.evaluate("/home/atas/test.ply","/home/atas/results.ply")
    print("spinning")



  def evaluate(self,ply_file_path, out_ply_file_path):
    is_training = False
    
    with tf.device('/gpu:'+str(GPU_INDEX)):
      pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
      is_training_pl = tf.placeholder(tf.bool, shape=())

      pred = get_model(pointclouds_pl, is_training_pl)
      loss = get_loss(pred, labels_pl)
      pred_softmax = tf.nn.softmax(pred)
  
      saver = tf.train.Saver()
      
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
        'labels_pl': labels_pl,
        'is_training_pl': is_training_pl,
        'pred': pred,
        'pred_softmax': pred_softmax,
        'loss': loss}
    
    total_correct = 0
    total_seen = 0
    fout_out_filelist = open(FLAGS.output_filelist, 'w')
    

    out_data_label_filename = os.path.basename(ply_file_path)[:-4] + '_pred.txt'
    out_data_label_filename = os.path.join(DUMP_DIR, out_data_label_filename)
    out_gt_label_filename = os.path.basename(ply_file_path)[:-4] + '_gt.txt'
    out_gt_label_filename = os.path.join(DUMP_DIR, out_gt_label_filename)

    print(ply_file_path, out_data_label_filename)
      # Evaluate room one by one.
    a, b = self.eval_one_epoch(sess, ops, ply_file_path,out_ply_file_path, out_data_label_filename, out_gt_label_filename)
    total_correct += a
    total_seen += b
    fout_out_filelist.write(out_data_label_filename+'\n')
    
    fout_out_filelist.close()
    #log_string('all room eval accuracy: %f'% (total_correct / float(total_seen)))

  def eval_one_epoch(self,sess, ops, ply_file_path, out_ply_file_path, out_data_label_filename, out_gt_label_filename):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    if FLAGS.visu:
      fout = open(os.path.join(DUMP_DIR, os.path.basename(ply_file_path)[:-4]+'_pred.obj'), 'w')
      fout_gt = open(os.path.join(DUMP_DIR, os.path.basename(ply_file_path)[:-4]+'_gt.obj'), 'w')
      fout_real_color = open(os.path.join(DUMP_DIR, os.path.basename(ply_file_path)[:-4]+'_real_color.obj'), 'w')
    fout_data_label = open(out_data_label_filename, 'w')
    fout_gt_label = open(out_gt_label_filename, 'w')

    out_pcd = o3d.geometry.PointCloud()
    xyz = np.zeros((NUM_POINT, 3))
    colors = np.zeros((NUM_POINT, 3))

    
    data_label_npy = ply2npy(ply_file_path)
    current_data, current_label = room2blocks_wrapper_normalized(data_label_npy, NUM_POINT)

    current_data = current_data[:,0:NUM_POINT,:]
    # Get room dimension..
  
    max_room_x = 8
    max_room_y = 8
    max_room_z = 8
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    print(file_size)

    
    for batch_idx in range(num_batches):
      start_idx = batch_idx * BATCH_SIZE
      end_idx = (batch_idx+1) * BATCH_SIZE
      cur_batch_size = end_idx - start_idx
      
      feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
            ops['labels_pl']: current_label[start_idx:end_idx,:],
            ops['is_training_pl']: is_training}
      loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],
                      feed_dict=feed_dict)

      if FLAGS.no_clutter:
        pred_label = np.argmax(pred_val[:,:,0:12], 2) # BxN
      else:
        pred_label = np.argmax(pred_val, 2) # BxN
      
      # Save prediction labels to OBJ file
      for b in range(BATCH_SIZE):
        pts = current_data[start_idx+b, :, :]
        l = current_label[start_idx+b,:]
        pts[:,6] *= max_room_x
        pts[:,7] *= max_room_y
        pts[:,8] *= max_room_z
        pts[:,3:6] *= 255.0
        pred = pred_label[b, :]
        for i in range(NUM_POINT):
          color = indoor3d_util.g_label2color[pred[i]]
          color_gt = indoor3d_util.g_label2color[current_label[start_idx+b, i]]
          if FLAGS.visu:
            fout.write('v %f %f %f %d %d %d\n' % (pts[i,0], pts[i,1], pts[i,2], color[0], color[1], color[2]))
            fout_gt.write('v %f %f %f %d %d %d\n' % (pts[i,0], pts[i,1], pts[i,2], color_gt[0], color_gt[1], color_gt[2]))

            xyz[i, 0], xyz[i, 1],xyz[i, 2] = pts[i,0], pts[i,1], pts[i,2]
            colors[i,0],colors[i,1],colors[i,2] = color[0], color[1], color[2]


          fout_data_label.write('%f %f %f %d %d %d %f %d\n' % (pts[i,0], pts[i,1], pts[i,2], pts[i,3], pts[i,4], pts[i,5], pred_val[b,i,pred[i]], pred[i]))
          fout_gt_label.write('%d\n' % (l[i]))
      
      correct = np.sum(pred_label == current_label[start_idx:end_idx,:])
      total_correct += correct
      total_seen += (cur_batch_size*NUM_POINT)
      loss_sum += (loss_val*BATCH_SIZE)
      for i in range(start_idx, end_idx):
        for j in range(NUM_POINT):
          l = current_label[i, j]
          #total_seen_class[l] += 1
          #total_correct_class[l] += (pred_label[i-start_idx, j] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    fout_data_label.close()
    fout_gt_label.close()

    out_pcd.points = o3d.utility.Vector3dVector(xyz)
    out_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_ply_file_path,out_pcd)

    if FLAGS.visu:
      fout.close()
      fout_gt.close()
    return total_correct, total_seen







      
if __name__=='__main__':
  
  ic = PointNet_Ros_Node()
  rospy.init_node('ros_point_cloud',anonymous=True)
  rospy.spin()
  '''
  while(True):
    with tf.Graph().as_default():

      ply_file_path = INFERENCE_DIR+ "latest_raw.ply"
      out_ply_file_path = INFERENCE_DIR+ "latest_segmented.pcd"

      #evaluate(ply_file_path, out_ply_file_path)      
      # '''