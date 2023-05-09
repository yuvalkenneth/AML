import torch
from mingpt.model import GPT
from mingpt.trainer import Trainer
import mingpt.bpe
import tqdm

TRAIN_ITERATIONS = 1000

TRAIN_BATCH_SIZE = 32

VOCAB_SIZE = 50257
BLOCK_SIZE = 64
LR = 5e-4


def init_model():
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-nano'
    model_config.vocab_size = VOCAB_SIZE  # TODO check if correct
    model_config.block_size = BLOCK_SIZE
    return GPT(model_config)


def init_trainer(model_to_train, data):
    train_config = Trainer.get_default_config()
    train_config.learning_rate = LR
    train_config.max_iters = TRAIN_ITERATIONS
    train_config.batch_size = TRAIN_BATCH_SIZE
    return Trainer(train_config, model_to_train, data)


if __name__ == '__main__':
    with open("alice_in_wonderland.txt") as f:
        dataset = f.read()
    # data = data.replace("\n", " ")
    e = mingpt.bpe.BPETokenizer()
    tokenized_data = e(dataset)
    model = init_model()
    trainer = init_trainer(model, tokenized_data)
    trainer.run()
