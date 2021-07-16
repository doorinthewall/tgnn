#!bin/bash

#python3 -m pdb -c continue fit_downstream.py \
./fit_downstream.py \
-s train_data.pkl \
-d models \
-t 'y_cred' \
-f ../selected_feats.pkl \
--cat_cols ../selected_cats.pkl \
-a ../../MTS_data_to_fit/snowball_part/adj.pkl \
-c config.ini \
--config_option 'DEFAULT'
