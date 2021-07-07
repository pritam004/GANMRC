 CUDA_VISIBLE_DEVICES=0,1  python new_test.py  --checkpoint ../output_distil_roberta --output_dir ../output_distil_roberta --train_file ../data/squad/train-v1.1.json --do_train --model_type roberta  --do_predict --predict_file /home/pritam/MRC/data/natural_shift/reddit_v1.0.json

# CUDA_VISIBLE_DEVICES=0,1 python new_test.py  --checkpoint ../output_red --output_dir ../output_red --train_file /home/pritam/MRC/notebooks/squad_reddit.json --do_train --model_type bert-finetuned  --do_predict --predict_file /home/pritam/MRC/data/natural_shift/reddit_v1.0.json 

# CUDA_VISIBLE_DEVICES=0,1  python new_test.py --learning_rate 5e-6 --checkpoint ../output_medical_only --output_dir ../output_medical_only --train_file /home/pritam/MRC/notebooks/medical_pseudo.json --do_train --model_type bert-finetuned  --do_predict --predict_file /home/pritam/MRC/data/squad/dev-v1.1.json 

# CUDA_VISIBLE_DEVICES=0,1  python new_test.py --learning_rate 5e-5 --checkpoint ../output_medical --output_dir ../output_medical --train_file /home/pritam/MRC/notebooks/squad_medical.json --do_train --model_type bert  --do_predict --predict_file /home/pritam/MRC/data/squad/dev-v1.1.json 