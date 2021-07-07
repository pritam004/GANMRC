# python new_test.py  --checkpoint ../output_reddit --output ../output_reddit/ --train_file ../data/squad/train-v1.1.json  --do_predict --predict_file ../data/natural_shift/amazon_reviews_v1.0.json --model_type bert --data_type amazon
# python evaluate_squad.py ../data/natural_shift/amazon_reviews_v1.0.json ../output_reddit/amazonpredictions.json


# python new_test.py  --checkpoint ../output_reddit --output ../output_reddit/ --train_file ../data/squad/train-v1.1.json  --do_predict --predict_file ../data/natural_shift/nyt_v1.0.json --model_type bert --data_type nyt
# python evaluate_squad.py ../data/natural_shift/nyt_v1.0.json ../output_reddit/nytpredictions.json

# python new_test.py  --checkpoint ../output_distil_roberta --output ../output_distil_roberta/ --train_file ../data/squad/train-v1.1.json  --do_predict --predict_file ../data/natural_shift/reddit_v1.0.json --model_type roberta --data_type reddit
# python evaluate_squad.py ../data/natural_shift/reddit_v1.0.json ../output_distil_roberta/redditpredictions.json

# python new_test.py  --checkpoint ../output_reddit --output ../output_reddit/ --train_file ../data/squad/train-v1.1.json  --do_predict --predict_file ../data/natural_shift/new_wiki_v1.0.json --model_type bert --data_type new_wiki
# python evaluate_squad.py ../data/natural_shift/new_wiki_v1.0.json ../output_reddit/new_wikipredictions.json

python new_test.py  --checkpoint ../output --output ../output/ --train_file ../data/squad/train-v1.1.json  --do_predict --predict_file ../data/natural_shift/medical.json --model_type roberta --data_type medical
python evaluate_squad.py ../data/natural_shift/medical.json ../output/medicalpredictions.json