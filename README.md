# `tSNE`
This class is an implementation of the tSNE. The class includes two main  callablemethods: fit and predict.

## `__init__`
Initiallizing parameters perplexity and dimensions.

## `grid_search`
Using perplexity parameter finding the best sigmas for each iteration for p_ij

## `fit`

The fit method computes p_ij, pij and does Gradient Descent Search by initialising values of y, finding q_ij and then updating both on each step using GSD algorithm.

## `predict`
Predict method outputs the new Y matrix with reduced dimensionality. 

Below you can see an example of a code you can use to run this model for your data X.
```
tsne = tSNE(perplexity=30, dimensions=2)
tsne.fit(X)
Y = tsne.predict()
print(Y)
```

