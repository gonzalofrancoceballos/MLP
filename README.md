# MLP project
Simple and light-weight implementation of a Multi-Layer Perceptron library using only Numpy


## Code example
The following lines showcase the use of this library to run a simple train-predict task


#### Instantiate a new empty model
```python
from models import BasicMLP
model = BasicMLP()
```

#### Add layers
Only first layer needs input_dim specified. Rest of layers will infer from layer that feeds into them
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
This is simply to mimic the behavior of `Keras` library. All this does is initialize the layers
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

## Train parameters:
- `n_epoch`: number of epochs (default: 10)
- `batch_size`: batch size (default: 128)
- `n_stopping_rounds`: N of consecutive epochs without improvement for early-stopping (default: 10)
- `learning_rate`:1,  # learning rate (default: 0)
- `reg_lambda`:  # regularization factor for gradients (default: 0)
- `verbose`: flag to plot train  results (default: True)
- `print_rate`: print train results every print_rate epochs (default: 5)
- `early_stopping`: flag to use early-stopping (default: False)

## Authors
* **Gonzalo Franco** - [Github](https:///github.com/gonzalofrancoceballos)


## License
This project is licensed under the GNU  linse- see the [LICENSE.md](https://github.com/gonzalofrancoceballos/MLP/blob/master/LICENSE) file for details

## Collaborating
Here's a list of ideas to be inplemented in the future. If you are interested in collaborateing, feel free to pick any of these! If there is anything else that you would like to contribute with, please open an [issue](https://github.com/gonzalofrancoceballos/MLP/issues)
- **Save train log**: save train log along with the model
- **Auto-save best train when using cross-validation**: store best train iteration
- **Keep trainlog**: when model train is stopped by hand, train log resets
- **2D Conv layer**: add a 2D convolutional layer
- **3D Conv layer** add a 3D convolutional layer
- **Flatten layer**: add a flatten layer to turn convolutional into dense
- **Concat layer**: add concat layer so convolutionals and denses can be combined
- **Autograd**: implement autograd library
- **Logging**: add possibility to pass a logger to train task