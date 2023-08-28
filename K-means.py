import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

#Nguồn tham khảo: https://machinelearningcoban.com/2017/01/01/kmeans/

def kmeans_init_centroid(img_1d, k_cluster, type='random'):
    if type == 'in_pixels':
        return img_1d[np.random.choice(img_1d.shape[0], k_cluster, replace=False)]
    if type == 'random':
        return np.random.randint(0, 256, size = (k_cluster, img_1d.shape[1]))
    

def kmeans_assign_labels(img_1d, centroid):
    distances = cdist(img_1d, centroid, metric='euclidean')
    return np.argmin(distances, axis=1)


def  kmeans_update_centers(img_1d, label, k_cluster, channel):
    centroid = np.zeros((k_cluster,channel))
    for i in range(k_cluster):
        cluster_k = img_1d[label == i]
        if cluster_k.shape[0] > 0:
            centroid[i] = np.mean(cluster_k, axis = 0)
    return centroid

def should_stop(centroid, new_centroid, epsilon=1e-5):
    return np.all(np.abs(centroid - new_centroid) <= epsilon)

def update_picture(img_1d,k_clusters,label,centroids):
    new_img = np.zeros((img_1d.shape[0],img_1d.shape[1]))
    for k in range(k_clusters):
        new_img[label == k,]  += centroids[k]
    return new_img

def kmeans(img_1d, k_clusters, max_iter=100, init_centroids='random'):
    row = img_1d.shape[0]
    column = img_1d.shape[1]
    channel = img_1d.shape[2]
    #reshape image into 2D
    img_1d = img_1d.reshape(row * column, channel)
    centroid = [kmeans_init_centroid(img_1d, k_clusters, init_centroids)]
    label = []
    iterations = 0
    while iterations < max_iter:
        label.append(kmeans_assign_labels(img_1d, centroid[-1]))
        new_centroid =  kmeans_update_centers(img_1d, label[-1], k_clusters, channel)
        if should_stop(centroid[-1], new_centroid):
            break
        iterations += 1
        centroid.append(new_centroid)
    new_img = update_picture(img_1d, k_clusters, label[-1], centroid[-1])
    new_img = new_img.reshape(row, column, channel) 

    return centroid[-1], new_img

if __name__ == "__main__":
    print('Input file of name to compress + extension (E.G: test1.jpg): ')
    name  = input()
    img_1d = Image.open(name) 
    img_1d = np.asarray(img_1d)
    k_clusters = 4
    row = img_1d.shape[0]
    column = img_1d.shape[1]
    #init_centroids = 'in_pixels' OR init_centroids = 'random'
    new_centroid, new_img = kmeans(img_1d, k_clusters,init_centroids = 'random')
    print('new centroid = ', new_centroid)
    print('new img = ', new_img)
    print('Input type of output (png or pdf): ')
    type = input()
    if type == 'png':
        img = Image.fromarray(new_img.astype(np.uint8))
        img.save('output2.png')
    elif type == 'pdf':
        img = Image.fromarray(new_img.astype(np.uint8)).convert('RGB')
        img.save('output.pdf', format='PDF')
        
          
 