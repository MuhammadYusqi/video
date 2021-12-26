import numpy as np
import tensorflow as tf 
import keras.backend as K
  
def Grid_to_Ann(output_cnn, size_image, grid, th_obj = 0.5):
  #Mengubah nilai Obj Score dengan th = 0.5 
  layerObj = output_cnn[:, :, :, 0]
  layerObj[layerObj < th_obj]  = 0
  layerProb = np.copy(layerObj)
  layerObj[layerObj >= th_obj] = 1

  output_cnn[:, :, :, 0] = layerObj

  size_grid = size_image / grid
  obj = output_cnn[...,0:1]
  g_xy = output_cnn[...,1:3]
  g_wh = output_cnn[...,3:]

  a = K.arange(0, grid, dtype=tf.float32)
  a = tf.repeat(a, grid)
  b = K.reshape(a, (grid,grid))
  b = K.flatten(K.transpose(b))
  a = K.reshape(a, (grid * grid, 1))
  b = K.reshape(b, (grid * grid, 1))
  g_temp = K.concatenate((a,b), axis=1)
  g_temp = K.reshape(g_temp, (grid, grid, 2))
  mask = g_xy <= 0
  xy_mask  = tf.where(mask, 0., g_temp)

  g_xy = g_xy * (size_grid - 1) + (xy_mask * size_grid)   
  g_wh = g_wh * size_image

  XY_min = g_xy - (g_wh / 2)
  XY_max = g_xy + (g_wh / 2)
  ano = K.concatenate((XY_min, XY_max), axis=3)
  ano = np.array(tf.cast(ano, dtype=tf.int32))

  #Mengubah ukuran anotasi citra dari (n, 3, 3, 4) => (n, 9, 4)
  ano = ano.reshape((-1, grid**2, 4)) #n, 9, 4
  #Mengubah ukuran Obj score dari (n, 3, 3) => (n, 9)
  layerObj = layerObj.reshape((-1, grid**2))
  layerProb = layerProb.reshape((-1, grid**2))
  #Menghilangkan anotasi yang Obj score = 0
  result = []
  prob = []
  for i in range(len(layerObj)):
    temp1 = []
    temp2 = []
    for j in range(len(layerObj[i])):
      if layerObj[i][j] == 1:
        temp1.append(ano[i][j])
        temp2.append(layerProb[i][j])
    result.append(temp1)
    prob.append(temp2)
  result = np.array(result)
  prob = np.array(prob)
  return result, prob

