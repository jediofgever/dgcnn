import h5py
import os, os.path
import numpy as np
from plyfile import PlyData, PlyElement
import glob
import open3d as o3d

data = np.zeros((193, 4096, 6), dtype = np.float32)
label = np.zeros((193,  4096),dtype = np.uint8)
 	
f = h5py.File('real_data.h5', 'w')

labeled_data_dir = '/home/atas/POINTNET_LABELED_REAL_DATASET/'
rgb_data_dir = '/home/atas/POINTNET_REAL_DATASET/'
i = -1

for filepath in glob.iglob(labeled_data_dir+'*.ply'):
    if(i == 192):
        break; 
    i = i+1
    xyz_label_ply = PlyData.read(filepath)
       
    filepath = os.path.relpath(filepath,labeled_data_dir)
    file_no = filepath[:-4]
 

    colors = np.zeros((4096, 3))
    rgb_pcd_path = rgb_data_dir + file_no + '.pcd'
    rgb_pcd = o3d.io.read_point_cloud(rgb_pcd_path)
    colors = np.asarray(rgb_pcd.colors)
    print(colors)
 

    for j in range(0, 4096):

       data[i, j] = [xyz_label_ply['vertex']['x'][j], xyz_label_ply['vertex']['y'][j], xyz_label_ply['vertex']['z'][j], 
                     colors[j,0], colors[j,1],colors[j,2]]
            
      
       label[i,j] = xyz_label_ply['vertex']['label'][j]

f.create_dataset('data', data = data)
f.create_dataset('label', data = label)
                    

