# knn-classifier-on-mnist-dataset
The MNIST dataset is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.
knn classifier uses the k-nearest neighbours to vote to which class the data belongs.
low values of k can be noisy and subject to the effects of outliers, large values of k are smooth over things but too big values can make categories be out voted.

 here i used keras library to load the minst dataset .
scikitlearn library includes the knnclassifier which i used to fit the data .
i divided the images using center of gravity .
the testing sample was 10% the training sample.
I tried various sample sizes and values of k. and here is some of the trials:
at training sample = 50000 , at k=1 ,accuarracy=80.72
                             at k=2 ,accuarracy=80.72
                             at k=7 ,accuarracy=84.88

at training sample = 10000 , at k=1 ,accuarracy=81.1
                             at k=3 ,accuarracy=82.6
                             at k=6 ,accuarracy=85.3
                             at k=24 ,accuarracy=84

at training sample = 1000 , at k=1 ,accuarracy=83
                             at k=4 ,accuarracy=83
                             at k=10 ,accuarracy=86
                             at k=50 ,accuarracy=80

my conclusion :
at smaller samples , it tends to have a bit higher accuracy .
at larger values of k the accuarracy is better. accuracy increases as k increases until the value of k  becomes too big so accuracy
decreases because categories gets outvoted.

