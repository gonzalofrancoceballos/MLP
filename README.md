# MLP project

This is a work-in-progress repository. Simple and light-weight implementation of a Multi-Layer Perceptron library using only Numpy


## Code example

The following lines showcase the use of this library to run a simple train-predict task


#### Instantiate a regression  model

```python
mlp = MLP(X, hidden_layers=[5,4,2], activation="tanh", optimizer="adam")
```

#### Instantiate a binary classificaton  model

```python
mlp = MLP(X, 
          hidden_layers=[3,3,2], 
          activation="swish", 
          optimizer="adam", 
          problem="binary_classification",
          loss = "logloss")

```

#### Instantiate a quantile regression  model

```python
mlp = MLP(X,
          hidden_layers=[5,5,5],
          activation="tanh", optimizer="adam", 
          problem="quantile",
          loss="quantile",
          q=0.01)
```

#### Train on test, dev data
```python
mlp.train(X,y,
          X_dev=X_dev, 
          y_dev=y_dev,
          n_epoch=130,
          batch_size=256)
```      

#### Compute predictions on new data
```python
predictions = mlp.predict(X_pred)
```            

#### Save and load a model
```python
mlp.save("model.json")

mlp = MLP()
mlp.load("model.json")
```      



## Authors

* **Gonzalo Franco** - [Github](https:///github.com/gonzalofrancoceballos)


## License

This project is licensed under the GNU  linse- see the [LICENSE.md](https://github.com/gonzalofrancoceballos/MLP/blob/master/LICENSE) file for details