#!bin/bash

working_folder=/home/Container/bgnn
cur_folder=$(pwd)
dataset=house
model_folder=model_folder
option='NODE'
device='cuda'
task='regression'
seed_num=5
nepochs=10

function fit {
	python3 -m pdb run.py \
	--dataset "$working_folder/datasets/$dataset/X.csv" \
	--labels "$working_folder/datasets/$dataset/y.csv" \
	--save_folder "./$model_folder/$dataset/model$1" \
	--graph "$working_folder/datasets/$dataset/graph.graphml" \
	--config "$cur_folder/config.ini" \
	--config_option $option \
	--task $task \
	--gnn_passes_per_epoch 1 \
	--device $device \
	--seed_num $seed_num \
	--stochastic True \
	--nepochs $nepochs \
	--masks "$working_folder/datasets/$dataset/masks.json"
}

fit
