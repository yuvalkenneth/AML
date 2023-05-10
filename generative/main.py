import torch
from torch.utils.data import DataLoader, TensorDataset
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
    gpt = GPT(model_config)
    return gpt


def init_trainer(model_to_train, data):
    train_config = Trainer.get_default_config()
    train_config.learning_rate = LR
    train_config.max_iters = TRAIN_ITERATIONS
    train_config.batch_size = TRAIN_BATCH_SIZE

    def batch_end_callback(trainer):
        if trainer.iter_num % 100 == 0:
            print(
                f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss "
                f"{trainer.loss.item():.5f}")

    trainer = Trainer(train_config, model_to_train, data)
    trainer.set_callback('on_batch_end', batch_end_callback)
    return trainer


if __name__ == '__main__':
    with open("alice_in_wonderland.txt") as f:
        dataset = f.read()
    # data = data.replace("\n", " ")
    e = mingpt.bpe.BPETokenizer()
    tokenized_data = e(dataset)

    x = torch.stack([tokenized_data[0][i:i + BLOCK_SIZE] for i in range(len(tokenized_data[0]) - BLOCK_SIZE -
                                                                        1)])
    y = torch.tensor([tokenized_data[0][i + BLOCK_SIZE + 1] for i in range(len(tokenized_data[0]) -
                                                                           BLOCK_SIZE - 1)])
    dataset = TensorDataset(x, y)
    model = init_model()
    model_trainer = init_trainer(model, dataset)
    model_trainer.run()
