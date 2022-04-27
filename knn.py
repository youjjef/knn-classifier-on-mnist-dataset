from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from keras.datasets import mnist
import numpy as np

(train_X, train_y), (test_X, test_y) = mnist.load_data()
y=train_y[:10000]
X=train_X[:10000]

def split(array, nrows, ncols):

    r, h = array.shape

    return (array.reshape(h//nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))

def get_center_of_graviyt(matrix):
    y = 0
    x = 0
    counter = 1
    my_mat=[]
    for i in range(14):
        for j in range(14):
            if matrix[i][j] != 0:
                list = [[j, i]]
                my_mat.append([j, 14 - i])

    for i in my_mat :
        x = x + i[0]
        y=y+i[1]
        counter+=1

    return x/counter, y/counter


load_digits = load_digits()



new_data=np.array([[0,0,0,0,0,0,0,0]])

for phote in X:
    #phote=phote.reshape(28,28)
    a,b,c,d=split(phote,14,14)
    x1,y1=get_center_of_graviyt(a)
    x2,y2 = get_center_of_graviyt(b)
    x3,y3= get_center_of_graviyt(c)
    x4,y4 = get_center_of_graviyt(d)

    row = np.array([x1, x2,x3,x4,y1,y2,y3,y4])
    new_data=np.vstack([new_data,row])
new_data=np.delete(new_data,0,0)

[X_train, X_test, y_train, y_test] = train_test_split(new_data,  y, test_size=0.1, random_state=44, shuffle =True)

#KNN
KNNClassifierModel = KNeighborsClassifier(n_neighbors= 6,weights ='distance' , # it can be distance
                                          algorithm='auto') # it can be ball_tree, kd_tree,brute
KNNClassifierModel.fit(X_train, y_train)

#Calculating Details
print('KNNClassifierModel Train Score is : ' , KNNClassifierModel.score(X_train, y_train))
print('KNNClassifierModel Test Score is : ' , KNNClassifierModel.score(X_test, y_test))
print("accuracy : ",KNNClassifierModel.score(X_test, y_test)*100)