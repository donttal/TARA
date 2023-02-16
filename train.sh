#!/bin/bash
python3 ./scripts/train.py \
    --training_type ${training_type:-"mix"} \
    --template_type ${template_type:-"ptuning"} \ 
    --dataset ${dataset} \
    --data_condition ${data_condition} \
    --num_examples_per_label ${num_examples_per_label:-50} \
    --few_shot_train ${few_shot_train:-"true"} \ 
    --model ${model:-"BERT"} \
    --model_lr ${model_lr}  \
    --template_lr ${template_lr}  \
    --epoch_num ${epoch_num} \
    --batch_size ${batch_size} \
    --use_cuda ${use_cuda} \
    --bank_size ${bank_size:-32}


 

    
