# CUDA_VISIBLE_DEVICES=3,0 python advr_model.py  --checkpoint ../output_confused --output_dir ../output_confused --train_file ../data/squad/train-v1.1.json --do_train --model_type bert  --do_predict --predict_file ../data/squad/dev-v1.1.json 
CUDA_VISIBLE_DEVICES=0,1 python new_test_adversarial.py  --checkpoint ../output_nyt_ner_adv --output_dir ../output_nyt_ner_adv --train_file /home/pritam/MRC/notebooks/squad_nyt.json --do_train --model_type roberta  --do_predict --predict_file /home/pritam/MRC/data/natural_shift/nyt_v1.0.json
