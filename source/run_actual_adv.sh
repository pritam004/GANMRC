# CUDA_VISIBLE_DEVICES=3,0 python advr_model.py  --checkpoint ../output_confused --output_dir ../output_confused --train_file ../data/squad/train-v1.1.json --do_train --model_type bert  --do_predict --predict_file ../data/squad/dev-v1.1.json 
CUDA_VISIBLE_DEVICES=0 python adversarial_training.py  --generator_path /home/pritam/KGQA/aristo --checkpoint ../out --output_dir ../OUT_squad --train_file /home/pritam/MRC/notebooks/squad_pseudo.json --do_train --model_type roberta  --do_predict --predict_file /home/pritam/MRC/data/squad/dev-v1.1.json --gp /home/pritam/MRC/FINETUNE_GEN/pytorch_model.bin --data_type squad
