# Credit-Card-Fraud-Detection
## Dataset
[creditcard.csv](https://www.kaggle.com/mlg-ulb/creditcardfraud)

[FraudDetectionUsingText](https://assets.datacamp.com/production/repositories/2162/datasets/94f2356652dc9ea8f0654b5e9c29645115b6e77f/chapter_4.zip)

[Statlog (Australian Credit Approval) Data Set (SOM)](http://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval))
## Fraud Detection by SOM (Unsupervised Learning)
Each customer is a input observation point which is going to be mapped to a new output space. Each neuron inside the neural network being initialized as a vector of weights, that is a vector of 15 elements (customer ID + 14 attributes). The output of a customer will be the closest neuron (winning node) to the customer.

Use a neighborhood function (e.g Galch neighborhood function) to update the weight of the neighbors of the winning node to move them closer to the winning node. And we do this for all the customers in the input space, and repeat it again and again until it reaches a point where the neighborhood stops decreasing. Each time we repeat it, the output space decreases and loses dimensions.

The outliers in the output space can be defined as frauds.

**How to detect the outlier neurons?**
* Compute the mean of the Euclidean distance between this neuron and the neurons in its neighborhood.

**How to identify which customer originally in the input space is associated to the winning node?**
* Use an inverse mapping function.