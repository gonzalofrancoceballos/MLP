# MLP project

This is a work-in-progress repository. Simple and light-weight implementation of a Multi-Layer Perceptron library using only Numpy


## Code example

The following lines showcase the use of this library to run a simple train-predict task


#### Instantiate a new empty model

```python
from models import BasicMLP
model = BasicMLP()
```

#### Add layers

```python
from layers import Dense
from activations import Sigmoid, Relu
n_features = 10
model.add(Dense(units=32, activation=Relu(), input_dim=n_features))
model.add(Dense(units=64, activation=Relu()))
model.add(Dense(units=8, activation=Relu()))
model.add(Dense(units=1, activation=Sigmoid()))

```

#### Compile model

```python
model.compile()
```

#### Train a quantile model
```python
from losses import Quantile
params = {
    "learning_rate": 0.001, 
    "n_epoch": 100,
    "print_rate": 10
    }

loss = Quantile(0.5)
model.train(loss, train_data=[X, y], params=params)
```      

#### Train using train funciton
```python
#### Train a quantile model
from train import ModelTrain
params = {"n_epoch": 1000}
trainer = ModelTrain(params)
loss = Quantile(0.5)

trainer.train(model, loss=loss, train_data=[X, y])
```            

#### Save and load a model
```python
model.save("model.json")
mlp = BasicMLP()
mlp.load("model.json")
```      



## Authors

* **Gonzalo Franco** - [Github](https:///github.com/gonzalofrancoceballos)


## License

This project is licensed under the GNU  linse- see the [LICENSE.md](https://github.com/gonzalofrancoceballos/MLP/blob/master/LICENSE) file for details