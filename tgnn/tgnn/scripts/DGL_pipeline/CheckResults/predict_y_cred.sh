#!bin/bash

#python3 -m pdb  predict_downstream.py \
./predict_downstream.py \
-s test_data.pkl \
-d pred_y_cred.pkl \
-t 'y_cred' \
-f ../selected_feats.pkl \
--cat_cols ../selected_cats.pkl \
-a ../../MTS_data_to_fit/snowball_part/adj.pkl \
-c config_pred.ini \
--config_option 'DEFAULT'
--pretrained models/weights
