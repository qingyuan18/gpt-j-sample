import os
import argparse
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    set_seed,
    GPTJForCausalLM,
    AutoModelForCausalLM,
    default_data_collator,
)


from datasets import load_from_disk
import torch
import torch.distributed as dist
import evaluate

import deepspeed
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import Trainer, TrainingArguments
import nltk


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument("--model_id", type=str, default="google/flan-t5-xl", help="Model id to use for training.")
    parser.add_argument("--train_dataset_path", type=str, help="Path to processed dataset stored by sageamker.")
    parser.add_argument("--test_dataset_path", type=str, help="Path to processed dataset stored by sageamker.")
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size to use for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size to use for testing.")
    parser.add_argument("--generation_max_length", type=int, default=140, help="Maximum length to use for generation")
    parser.add_argument("--generation_num_beams", type=int, default=4, help="Number of beams to use for generation.")
    parser.add_argument("--learning_rate", type=float, default=3e-3, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Whether to use gradient checkpointing.")
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_known_args()
    return args


def training_function(args):
    # set seed
    set_seed(args.seed)

    # load dataset from disk and tokenizer
    train_dataset = load_from_disk(args.train_dataset_path)
    eval_dataset = load_from_disk(args.test_dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    metric = evaluate.load("accuracy")
    # load model from the hub
    model = GPTJForCausalLM.from_pretrained(
        args.model_id,
        use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
        cache_dir = "/tmp/input/" # For instance storage instance such as p4d.24xlarge, you can put the file under /tmp which has enough storage space
    )


    # Define compute metrics function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Define training args
    #If you just want to save the best model weights, you can set the output_dir to temporary path such as '/tmp' on p4d.24xlarge;
    #And if you want to save all of the checkpoint during the training, you can set the output_dir to the checkponit local path (it will impact the train speed for multi-nodes training. Because SageMaker will upload the checkpoint to S3 nearly real-time, it will occupy the networking bandwidth and impact the communication efficiency between nodes in the cluster).
    output_dir = '/tmp'
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        label_names=["input_ids", "attention_mask"],
        #predict_with_generate=True,
        #generation_max_length=args.generation_max_length,
        #generation_num_beams=args.generation_num_beams,
        fp16=True,  # T5 overflows with fp16
        #bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps = 80,      
        deepspeed=args.deepspeed_config,
        save_on_each_node=True,    #By default, DeepSpeed expects that a multi-node environment uses a shared storage. If this is not the case and each node can only see the local filesystem，you need to set the parameter to true.
        gradient_checkpointing=args.gradient_checkpointing,
        # logging & evaluation strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=40,
        evaluation_strategy="steps",
        save_strategy="no",
        eval_steps=60,
        save_total_limit=2,
        load_best_model_at_end=False, #need to set it to false during deepspeed multiple nodes training.
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        #compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    # Start training
    trainer.train()

    #We now save the model assets to an intermediate path.
    #Note: plesae do not save the model into /opt/ml/model (because Sagemaker will tar and compress all of files under /opt/ml/model, and it will consume much time for LLM.)
    print("------saving model!-----")
    save_model_dir = '/tmp/output/asset/'
    tokenizer.save_pretrained(save_model_dir)
    trainer.save_model(save_model_dir)
    print("------model is saved!-----")
    
    #Note: we just use the rank 0 process to upload the trained model assets to S3 by s5cmd command.
    WORLD_RANK = int(os.environ['RANK'])
    if WORLD_RANK == 0:
        os.system("./configz_and_code/scripts/s5cmd sync {0} {1}".format(save_model_dir, os.environ['MODEL_S3_PATH']))
    
    #Note: we should sync with every ranker and ensure rank 0 uploading the model assets successfully. 
    torch.distributed.barrier()

def main():
    #Note: here the "_" is needed because parse_arge() return a tuple.
    args, _ = parse_arge()
      
    # Environment variables set by torch.distributed.launch
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])
    
    dist.init_process_group(backend='nccl', rank=WORLD_RANK, world_size=WORLD_SIZE)
    
    if LOCAL_RANK != 0:
        print("---------local rank {0}".format(LOCAL_RANK))
    else :
        print("------download and unzip nltk punkt for for local rank 0!-----")
        nltk.download("punkt", quiet=True)
    
    #Note: the barrier is used to ensure just only local rank 0 to download and unzip the punkt, otherwise it may fail the training job. 
    torch.distributed.barrier()
    
    training_function(args)


if __name__ == "__main__":
    main()
