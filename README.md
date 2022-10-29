# Smart control algorithm
This repository contains a concise implementation of the smart control algorithm proposed in a research paper by Philippe de Bekker. Based on analysing various designed cases, the paper defined the heuristics of a smart control algorithm that matches power supply and demand optimally. The core of improvement of the smart control algorithm is exploiting future knowledge, which can be realized by current state-of-the-art forecasting techniques, to effectively store and trade energy. In addition, a simulation environment is provided.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to run this repository.

```bash
pip install -r requirements.txt
```

## Usage
Create an instance of `OptimizedModel` (i.e. provide data and a `Battery` instance) and call `run()`.


## License
[No License](https://choosealicense.com/no-permission/) (= No Permission)
