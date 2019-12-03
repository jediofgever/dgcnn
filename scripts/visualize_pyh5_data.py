import numpy as np 
import open3d 
import open3d as o3d
 


from plyfile import PlyData, PlyElement
import h5py
f = h5py.File('data0.h5','r')

print(f.keys())
data = f['data']
label = f['label']

xyz = np.zeros((len(data[1]), 3))
colors = np.zeros((len(data[1]), 3))

xyz[:, 0] = data[900][:,0]
xyz[:, 1] = data[900][:,1]
xyz[:, 2] = data[900][:,2]
colors[:, 0] = data[900][:,3]
colors[:, 1] = data[900][:,4]
colors[:, 2] = data[900][:,5]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.io.write_point_cloud("/home/atas/test.ply", pcd)

pcd_load = o3d.io.read_point_cloud('/home/atas/test.ply')
o3d.visualization.draw_geometries([pcd_load])

