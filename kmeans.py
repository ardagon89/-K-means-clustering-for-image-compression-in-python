#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np 
import matplotlib.image as img 
from scipy import misc
import os 

def convert_img_to_arr(img_path):
    img = misc.imread(img_path)
    rows, cols, colors = img.shape
    rgb_arr = np.reshape(img, (rows*cols, colors))
    #rgb_arr = rgb_arr/255
    return rows, cols, colors, rgb_arr, os.stat(img_path).st_size

def find_closest_points(rgb_arr, centroids, k):
    dist = np.zeros((rgb_arr.shape[0], k))
    for i in range(k):
        dist[:, i] = np.sqrt(np.sum((rgb_arr-centroids[i])**2, axis = 1))
        
    return np.argmin(dist, axis=1)

def update_centroid(k, centroids, rgb_arr, closest_points):
    for i in range(k):
        centroids[i] = np.mean(rgb_arr[np.where(closest_points == i),:], axis=1)[0]
    #print(centroids)
        
def train_kmeans(iterations, rgb_arr, centroids, k):
    total_points = rgb_arr.shape[0]
    points_swithced_cluster = total_points
    i = 0
    while points_swithced_cluster > 0 and i < iterations:
        i += 1
        new_closest_points = find_closest_points(rgb_arr, centroids, k)
        update_centroid(k, centroids, rgb_arr, new_closest_points)
        if i > 1:
            points_swithced_cluster = np.sum(closest_points != new_closest_points)
            closest_points = new_closest_points
        else:
            closest_points = new_closest_points
        #print('# of points updated centroid in round', i, ':', points_swithced_cluster)
        
    return new_closest_points
        
def save_compressed_img(k, closest_points, centroids, rows, cols, colors, img, output=None, iteration=''):
    img_arr = np.zeros((rows*cols, colors))
    for i in range(k):
        img_arr[np.where(closest_points == i),:] = centroids[i]
    
    compressed_img = np.reshape(img_arr, (rows, cols, colors))
    if not output:
        img_ls = img.split('.')
        misc.imsave(str(k)+'k_compressed_' + ''.join(img_ls[:-1]) + iteration + '.' + img_ls[-1], compressed_img)
        return os.stat(str(k)+'k_compressed_' + ''.join(img_ls[:-1]) + iteration+ '.' + img_ls[-1]).st_size
    else:
        output_ls = output.split('.')
        misc.imsave(''.join(output_ls[:-1]) + iteration + '.' + output_ls[-1], compressed_img)
        return os.stat(''.join(output_ls[:-1]) + iteration + '.' + output_ls[-1]).st_size
    
def kmeanspp(k, rgb_arr):
    centroids = np.zeros((k, rgb_arr.shape[1]))
    centroids[0, :] = rgb_arr[np.random.randint(0, rgb_arr.shape[0]), :]
    selected_list = []
    for j in range(1, k):
        dist = np.zeros((rgb_arr.shape[0], k))
        for i in range(k):
            dist[:, i] = np.sqrt(np.sum((rgb_arr-centroids[i])**2, axis = 1))
        
        min_dist = np.min(dist, axis=1)
        #print('min_dist', min_dist)
        for val in selected_list:
            min_dist[val] = 0.0
        #print('min_dist', min_dist)
        p = min_dist/np.sum(min_dist)
        #print('p',p)
        selected_list.append(np.random.choice(rgb_arr.shape[0], size=None, replace=False, p=p))

        #centroids[j, :] = rgb_arr[np.argmax(np.min(dist, axis=1)), :]
        centroids[j, :] = rgb_arr[selected_list[-1], :]
    #print('selected_list', selected_list)
    return centroids

def getcentroids(k, rgb_arr):
    centroids = np.zeros((k, rgb_arr.shape[1]))
    selected_list = np.random.choice(rgb_arr.shape[0], size=k, replace=False, p=None)
    for n in range(k):
        centroids[n, :] = rgb_arr[selected_list[n], :]
    
    return centroids

def calc_full(max_iter, img_list=None, k_list=None, output=None):
    iterations = max_iter
    if not img_list:
        img_list = ['Penguins.jpg', 'Koala.jpg']
    if not k_list:
        k_list = [2, 5, 10, 15, 20]
    compression_list = []
    for m in range(10):
        for img in img_list:
            #print("for img:", img)
            rows, cols, colors, rgb_arr, img_size = convert_img_to_arr(img)
            #print("rows:", rows, "cols:", cols, "colors:", colors, "rgb_arr:", rgb_arr)
            for k in k_list:    
                #centroids = kmeanspp(k, rgb_arr)   
                centroids = getcentroids(k, rgb_arr)   
                closest_points = train_kmeans(iterations, rgb_arr, centroids, k)
                img_resize = save_compressed_img(k, closest_points, centroids, rows, cols, colors, img, output, str(m+1))    
                compression_list.append([img, k, m, img_size, img_resize])
        print('Iteration', m+1, ': Image size reduced from', compression_list[-1][3], 'bytes to', compression_list[-1][4], 'bytes')
    np.savetxt("compression.csv", np.array(compression_list), fmt='%s', delimiter=",")
    compression_perc = calc_avg_var(compression_list)
    np.savetxt("compression_perc.csv", np.array(compression_perc), fmt='%s', delimiter=",")
    print('Please find the output images and csv files compression.csv & compression_perc.csv')
    return compression_perc

def calc_avg_var(compression_list):
    compression_arr = np.array(compression_list)
    compression_perc = []
    for img1 in np.unique(compression_arr[:, 0]):
        for k1 in np.unique(compression_arr[:, 1]):
            size_vals = compression_arr[np.where((compression_arr[:, 0]==img1) & (compression_arr[:, 1]==k1))]
            size_dec_perc = 100-(size_vals[:,4].astype(float)/size_vals[:,3].astype(float))*100
            compression_perc.append([img1, k1, np.average(size_dec_perc), np.var(size_dec_perc)])
    return compression_perc
        
if __name__ == "__main__":
    import sys

    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
        
    #img = 'Penguins.jpg'
    #print(calc_full(max_iter))

    max_iter = 100
    if len(sys.argv) < 4:
        print("Usage:python kmeans.py <input-image> <k> <output-image>")
        img = 'Penguins.jpg'
        k = 2
        output = 'output.jpg'
        result = calc_full(max_iter=100, img_list=[img], k_list=[k], output='output.jpg')[0]
        print('Average compression achieved :', result[2], '%')
        print('Variance in compression :', result[3])
    else:
        img = sys.argv[1]
        k = int(sys.argv[2])
        output = sys.argv[3]
        
        result = calc_full(max_iter=100, img_list=[img], k_list=[k], output=output)[0]
        print('Average compression achieved :', result[2], '%')
        print('Variance in compression :', result[3])
                
        #print(img, k , output)
        #rows, cols, colors, rgb_arr, img_size = convert_img_to_arr(img)
        #centroids = kmeanspp(k, rgb_arr)
        #closest_points = train_kmeans(max_iter, rgb_arr, centroids, k)
        #save_compressed_img(k, closest_points, centroids, rows, cols, colors, img, output)  
        #print(img, 'compressed with k =', k,'. New image is', output)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[81]:





# In[ ]:





# In[ ]:




