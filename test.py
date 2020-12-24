import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

print(os.listdir("E:/3rd Year/AI/project/Face-Recognition-by-Random-forest/olivetti_faces.npy"))
pics=np.load("E:/3rd Year/AI/project/Face-Recognition-by-Random-forest/olivetti_faces.npy/olivetti_faces.npy")
labels= np.load("E:/3rd Year/AI/project/Face-Recognition-by-Random-forest/olivetti_faces.npy/olivetti_faces_target.npy")
print("pics: ", pics.shape)
print("labels: ", labels.shape)


fig = plt.figure(figsize=(20, 10))
columns = 10
rows = 4
for i in range(1,columns*rows+1):
    img = pics[10*(i-1),:,:]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img, cmap = plt.get_cmap('gray'))
    plt.title("person {}".format(i-1), fontsize=16)
    plt.axis('off')
    
plt.suptitle("There are 40 distinct people in the dataset", fontsize=22)
plt.show()
Xdata = pics # store images in Xdata
Ydata = labels.reshape(-1,1) # store labels in Ydata


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(Xdata, Ydata, test_size = 0.241, random_state=45)

print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)



from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=400,random_state=1)
rf.fit(x_train, y_train)
RF_accuracy = round(rf.score(x_test, y_test)*100,2)
print("RF_accuracy is %", RF_accuracy)


image=mpimg.imread("E:/3rd Year/AI/project/Face-Recognition-by-Random-forest/images/image-94.png")
image=image.reshape(1,-1)
print(image.shape)
y_pred= rf.predict(image)
print("Person",y_pred)
fig = plt.figure(figsize=(4,5))
img = pics[10*(y_pred[0]),:,:]
plt.imshow(img, cmap = plt.get_cmap('gray'))  
plt.axis('off')   
plt.title("person {}".format(y_pred), fontsize=16)
plt.show()




    
    
    
    
    
    
