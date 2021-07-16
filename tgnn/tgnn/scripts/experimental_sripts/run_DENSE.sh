#!bin/bash

working_folder='/home/jovyan/Alexey_jr/geo2vec/bgnn'
cur_folder=$(pwd)
dataset=house
model_folder=results_DENSE
option=Dense
device=cuda
task=regression
seed_num=5
nepochs=2000

mkdir $model_folder
mkdir $model_folder/$dataset

function fit {
	python3 -m pdb -c continue   run.py \
	--dataset "$working_folder/datasets/$dataset/X.csv" \
	--labels "$working_folder/datasets/$dataset/y.csv" \
	--save_folder "./$model_folder/$dataset/$option" \
	--graph "$working_folder/datasets/$dataset/graph.graphml" \
	--config "$cur_folder/hp_config.ini" \
	--config_option $option \
	--task $task \
	--device $device \
	--seed_num $seed_num \
	--stochastic True \
	--nepochs $nepochs \
	--masks "$working_folder/datasets/$dataset/masks.json"
}

fit

