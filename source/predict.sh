CUDA_VISIBLE_DEVICES=0,1 python new_test.py  --checkpoint ../output_xlnet --output_dir ../output_xlnet/ --train_file ../data/squad/train-v1.1.json  --do_predict --predict_file ../data/squad/dev-v1.1.json --model_type xlnet
python evaluate_squad.py ../data/squad/dev-v1.1.json ../output_xlnet/squadpredictions.json