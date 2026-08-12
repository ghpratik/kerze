minitorch-lib/ # top-level repo (rename per your final name choice)
├── README.md # project pitch, install, quickstart, benchmark numbers
├── LICENSE # MIT is standard for this kind of project
├── pyproject.toml # package metadata for PyPI (name, version, deps=[])
├── .github/
│ └── workflows/
│ └── test.yml # CI: run pytest on push
│
├── src/
│ └── minitorch/ # actual importable package
│ ├── **init**.py # exposes public API: Tensor, nn, optim
│ │
│ ├── ndarray.py # NDArray: flat buffer + shape + strides
│ │ # - get/set, reshape, transpose (stride tricks)
│ │
│ ├── tensor.py # Tensor: autograd wrapper around NDArray
│ │ # - data, grad, requires_grad, \_backward, \_prev
│ │ # - backward() — topological sort + chain rule
│ │
│ ├── ops.py # forward+backward pairs, the core math
│ │ # - add, mul, matmul, sum, mean, reshape, transpose
│ │ # - relu, sigmoid, exp, log
│ │ # - broadcast_shapes(), unbroadcast() helpers
│ │
│ ├── nn/
│ │ ├── **init**.py
│ │ ├── module.py # base Module class (params, zero_grad)
│ │ ├── linear.py # Linear layer
│ │ ├── activations.py # ReLU, Sigmoid as Module wrappers
│ │ └── losses.py # MSELoss, CrossEntropyLoss
│ │
│ └── optim.py # SGD, Adam optimizers
│
├── tests/
│ ├── test_ndarray.py # shape/stride correctness, get/set, transpose
│ ├── test_ops.py # forward correctness for each op
│ ├── test_gradients.py # numerical gradient checking — YOUR KEY PROOF FILE
│ ├── test_broadcast.py # dedicated tests for the hardest part
│ └── test_nn.py # Linear layer, loss functions end-to-end
│
├── examples/
│ ├── mnist_train.py # the flagship demo — trains an MLP on MNIST
│ ├── xor_toy.py # tiny sanity-check example (fast, good for README gif/demo)
│ └── benchmark.py # timing comparison vs numpy/torch (optional, for README numbers)
│
└── docs/
└── design_notes.md # your own write-up: strides, broadcasting derivation, # matmul backward derivation — this doubles as interview prep

## Commands to upload package

`pip install build twine --break-system-packages
python3 -m build
twine upload --repository testpypi dist/*   # test first
twine upload dist/*                          # then the real thing`
