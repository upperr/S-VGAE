# Efficient Spiking Variational Graph Autoencoders for Unsupervised Graph Representation Learning Tasks

This is the source code of our work "Efficient Spiking Variational Graph Autoencoders for Unsupervised Graph Representation Learning Tasks".

## Requirements

python 3.7.6

torch 1.11.0

torch-geometric 2.0.3

spikingjelly 0.0.0.0.12

## Examples

python train_link.py --dataset Cora --encoder_layer 64 --learning_rate 0.03 --flops

python train_link.py --dataset CiteSeer --encoder_layer 64 --T 3 --dropout 0.4 --learning_rate 0.015 --batch_size 256

python train_generate.py --dataset MUTAG --model SGAE --flops

python train_generate.py --dataset PTC_MR --model SGAE --flops