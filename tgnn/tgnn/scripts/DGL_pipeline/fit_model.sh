#!bin/bash

#python3 -m pdb -c continue fit_model.py \
./fit_model.py \
-s train_dataset.pkl \
-d models \
-t ../MTS_data_to_fit/target_cols.pkl \
-f selected_feats.pkl \
--cat_cols selected_cats.pkl \
-a ../MTS_data_to_fit/snowball_part/adj.pkl \
-c config.ini \
--config_option 'DEFAULT'
