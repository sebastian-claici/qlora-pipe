import subprocess
import json
import os
import argparse

from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from huggingface_hub import login
from huggingface_hub import snapshot_download

access_token = "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"
login(access_token)
training_env = json.loads(os.environ["SM_TRAINING_ENV"])


def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    # add model id and dataset path argument
    parser.add_argument("--model_id", type=str, help="Model id to use for training.")
    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    
    args = parser.parse_known_args()
    return args


def save_model(args):
    output = snapshot_download(repo_id=args.model_id, cache_dir=args.model_dir)
    subprocess.run(["mv", 
                    output,
                    os.path.join(args.model_dir, "base/")])
    subprocess.run(["rm -rf", output])


def save_dataset(args):
    dataset = load_from_disk(args.train_dir)
    dataset.to_json(os.path.join(args.train_dir, "train.json"))


if __name__ == "__main__":
    args, _ = parse_arge()
    save_dataset(args)
    save_model(args)

    subprocess.run('NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=8 train.py --deepspeed --deepspeed_config examples/ds_config.json --config examples/config.toml', shell=True, capture_output=True)