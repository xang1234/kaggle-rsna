model=model003_2_retrain
gpu=0
fold=3
conf=./conf/${model}.py

python -m src.cnn.main train ${conf} --fold ${fold} --gpu ${gpu}
