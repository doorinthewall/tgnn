#!bin/bash

nfolds=5
working_folder=~/upamo-storage/Alexey_jr/geo2vec/GCN/DGL_pipeline/CheckResults
model_folder=model_folder

function fit {
	./fit_downstream.py \
	-s "$working_folder/folds/fold$1" \
	-d "$working_folder/$model_folder/model$1_$2" \
	-t $2 \
	-f $working_folder/../selected_feats.pkl \
	--cat_cols $working_folder/../selected_cats.pkl \
	-a $working_folder/../../MTS_data_to_fit/snowball_part/adj.pkl \
	-c config.ini \
	--config_option 'DEFAULT'
}


function predict {
	./predict_downstream.py \
	-s "$working_folder/folds/test$1" \
	-d "$working_folder/$model_folder/model$1_$2/pred_$2.pkl" \
	-t $2 \
	-f $working_folder/../selected_feats.pkl \
	--cat_cols $working_folder/../selected_cats.pkl \
	-a $working_folder/../../MTS_data_to_fit/snowball_part/adj.pkl \
	-c config_pred.ini \
	--config_option 'DEFAULT' \
	--pretrained "$working_folder/$model_folder/model$1_$2/weights"
}

rm -r $working_folder/$model_folder
mkdir $working_folder/$model_folder

for (( fold=0; fold<$nfolds; fold++ ))
do
	for target in 'y_cred' 'y_comm' 'y_deps_delta'
	do
		mkdir "$working_folder/$model_folder/model${fold}_${target}"
		fit $fold $target
		predict $fold $target
	done
done	
