To run from the start:
python main.py ./data/cifar-10-batches-py train 

To run from the best epoch:
python main.py ./data/cifar-10-batches-py train --resume

To test from the best epoch:
python main.py  ./data/cifar-10-batches-py test --resume

To predict from the checkpoint:
python main.py ./data/private-test-images.npy predict --resume


attaching my checkpoint for the architecture1. Since I cannot upload many such checkpoints.
network architecture 1 is active for now, architecture 2 and 3 are commented. Feel free to explore them.

--------Thanks----