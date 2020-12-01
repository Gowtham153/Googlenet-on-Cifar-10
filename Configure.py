# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE

model_configs = {
	"name": 'MyModel',
	"save_dir": '../saved_models/',
	"depth": 2
	# ...
}

training_configs = {
	"max_epoch" : 128,
    "learning_rate" : 0.01,
    "batch_size" : 100,
    "momentum" : 0.9,
    "weight_decay" : 5e-4,
    "save_interval" : 10,
    "name": "MyModel",
    "save_dir":"/.saved_models/"
}

### END CODE HERE