import pandas as pd
import numpy as np
import random

df=pd.read_csv("Mall_Customers.csv")
print (df.shape) 
print ("\nSample information\n\n",df.head)
print ("\n\nSample stats\n\n",df.describe)
x_lab=df.iloc[:,[3,4]].values
for item in x_lab:
    print (item)
y_lab=[]

def random_centers(dim,k): 
    centers=[] 
    for i in range(k):
        center=[] 
        for d in range(dim):
            rand=random.randint(0,110)
            center.append(rand)
        centers.append(center) 
    return centers

def point_clustering(data, centers, dim, first_cluster=False): 
    for point in data:
        nearest_center, nearest_center_distance = 0, None 
        for i in range(0,len(centers)):
            euc_distance=0 
            for d in range(0,dim):
                dist=abs(point[d] - centers[i][d])
                euc_distance+=dist
            euc_distance=np.sqrt(euc_distance)
            if nearest_center_distance == None or nearest_center_distance > euc_distance:
                nearest_center_distance = euc_distance 
                nearest_center = i 
        if first_cluster:
            point.append(nearest_center) 
        else:
            point[-1]=nearest_center 
    return data 
                
def mean_center(data,centers,dim): 
    print ("Centers : ", centers,"\nDim : ",dim)
    new_centers=[]
    for i in range(len(centers)):
        new_center, total_of_points, number_of_points = [], [] , 0 
        for point in data:
            if point[-1] == i:
                number_of_points+=1 
                for d in range(0,dim):
                    if d < len(total_of_points):
                        total_of_points[d]+=point[d]
                    else:
                        total_of_points.append(point[d]) 
        if len(total_of_points) != 0:
            for d in range(0,dim):
                print ("Point total : ",total_of_points,"Dim : ",d)
                new_center.append(total_of_points[d]/number_of_points)
            new_centers.append(new_center)
        else:
            new_centers.append(centers[i])
    return new_centers
def train_k_mean_clustering(data, k, epochs):
    dims=len(data[0])
    print ("Data[0] : ",data[0])
    centers=random_centers(dims,k) 
    
    clustered_data=point_clustering(data,centers,dims,first_cluster=True)
    
    for i in range(epochs):
        centers=mean_center(clustered_data, centers, dims) 
        clustered_data=point_clustering(data, centers, dims, first_cluster=False) 
    
    return centers
          
new_x_lab=x_lab #processing the data to suit the algorithm format
new_list = [] 
for i in range(len(new_x_lab)):
    temp=[]
    temp.append(new_x_lab[i][0])
    temp.append(new_x_lab[i][1])
    new_list.append(temp) 

centers = train_k_mean_clustering(new_list,5,1000) 

print ("\nCalculated Centers : ",centers)
