# Exploiting Heterogeneous Program Knowledge Graph for Register-related Instruction Vulnerability Prediction (VP-HPKG)
VP-HPKG explores the random propagation of instruction-stream errors by representing the source program as a multi-layer heterogeneous program knowledge graph, and accurately locates the instructions prone to errors.

## Result on WebAssembly programs(C++)


| methods | generate_Fib          | float-mm |
|:-------:|:---------------------:|:--------:|
| ours    | A:0.89 P:0.98 F1:0.80 |A:0.85 P:0.90 F1:0.82|
| PrograML| A:0.85 P:0.93 F1:0.80 |A:0.82 P:0.85 F1:0.78|
| MVD     | A:0.67 P:0.50 F1:0.40 |A:0.63 P:0.56 F1:0.67|


## Overview
Each folder is a benchmark and contains the program data.
The most important files in each folder are as follow:
- data_test.py: Data acquisition and processing module.
  - `def get_node` : Identifying entities and relations from the information obtained by LLVM.
  - `def get_Ins_SDC` : Getting the attribute information of each entity.
- main.py: Data division, model building, and training module.
  - `def train_model` : Building a knowledge graph. Model building and training.
  - `G` : Heterogeneous Graph of instrcution layer.
  - `BB_G` : Graph of basic block layer. 
  
## Setup

To run the code, you need the following dependencies:
- [Pytorch 1.10.2](https://pytorch.org/)
- [DGL 0.9.0](https://www.dgl.ai/pages/start.html)
- [sklearn](https://github.com/scikit-learn/scikit-learn)
- [numpy 1.18.5](https://numpy.org/)
  
## DataSet
Our current experiments are conducted on data obtained by LLVM and LLFI.
- `Ins_g.dot` : The text and structure information of instructions.
- `F_B_I.dot` : The position number of the instruction in the program.
- `cycle_result.txt, result_other.txt` : The result of fault injection.

## Usage
Execute the following scripts to train on node classification task:

```bash
python main.py
```

