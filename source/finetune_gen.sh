# CUDA_VISIBLE_DEVICES=3,0 python advr_model.py  --checkpoint ../output_confused --output_dir ../output_confused --train_file ../data/squad/train-v1.1.json --do_train --model_type bert  --do_predict --predict_file ../data/squad/dev-v1.1.json 
CUDA_VISIBLE_DEVICES=0,1 python finetune_generator.py  --checkpoint ../output_nyt_ner_adv --output_dir ../FINETUNE_GEN1 --train_file ../data/squad/train-v1.1.json --do_train --model_type roberta  --do_predict --predict_file ../data/squad/dev-v1.1.json
