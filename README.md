# MLP project

This is a work-in-progress repository. Simple and light-weight implementation of a Multi-Layer Perceptron library using only Numpy


## Code example

The following lines showcase the use of this library to run a simple train-predict task


#### Instantiate a new model

```python
mlp = MLP(X, hidden_layers=[5,4,2], activation="tanh", optimizer="adam")
```

#### Train on test, dev data
```python
mlp.train(X,y,
          X_dev=X_dev, 
          y_dev=y_dev,
          n_epoch=100,
          n_stopping_rounds=30)
```      

#### Compute predictions on new data
```python
predictions = mlp.predict(X_pred)
```            
          

## Authors

* **Gonzalo Franco** - [Github](https:///github.com/gonzalofrancoceballos)


## License

This project is licensed under the GNU  linse- see the [LICENSE.md](https://github.com/gonzalofrancoceballos/MLP/blob/master/LICENSE) file for details