#!bin/bash

nfolds=1
working_folder=/home/Container/bgnn
cur_folder=$(pwd)
dataset=house
model_folder=model_folder
option='NODE'
device='cuda'

function fit {
	./run_tgnn_experiment.py \
	-s "$working_folder/datasets/$dataset/X.csv" \
	-d "./$model_folder/$dataset/model$1" \
	-t $2 \
	-c "$cur_folder/config.ini" \
	--config_option $option \
	--masks "$working_folder/datasets/$dataset/masks.json" \
	--graph "$working_folder/datasets/$dataset/graph.graphml" \
	--seed $1 \
	--nepochs 200 \
	--device 'cuda'
}


function predict {
	./predict_script.py \
	-s "$working_folder/datasets/$dataset/X.csv" \
	-d "./$model_folder/$dataset/model$1/pred.pkl" \
	-t $2 \
	-c "$cur_folder/config_pred.ini" \
	--config_option  $option \
	--pretrained "./$model_folder/$dataset/model$1/weights" \
	--masks "$working_folder/datasets/$dataset/masks.json" \
	--graph "$working_folder/datasets/$dataset/graph.graphml" \
	--seed $1
}

#rm -r ./$model_folder/$dataset
#mkdir ./$model_folder/$dataset

for (( fold=0; fold<$nfolds; fold++ ))
do
#	mkdir "./$model_folder/$dataset/model${fold}"
#	fit $fold "$working_folder/datasets/$dataset/y.csv"
	predict $fold "$working_folder/datasets/$dataset/y.csv"
done	
