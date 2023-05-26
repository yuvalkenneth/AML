import os.path
from torch.nn import functional as F
import numpy as np
import torch
from torch.utils.data import Dataset

import mingpt.bpe
from mingpt.model import GPT
from mingpt.trainer import Trainer

TRAIN_ITERATIONS = 1000

TRAIN_BATCH_SIZE = 32

VOCAB_SIZE = 50257
BLOCK_SIZE = 64
LR = 5e-4


def init_model():
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-nano'
    model_config.vocab_size = VOCAB_SIZE
    model_config.block_size = BLOCK_SIZE
    gpt = GPT(model_config)
    return gpt


def init_trainer(model_to_train, data):
    train_config = Trainer.get_default_config()
    train_config.learning_rate = LR
    train_config.max_iters = TRAIN_ITERATIONS
    train_config.batch_size = TRAIN_BATCH_SIZE

    def batch_end_callback(train):
        if train.iter_num % 100 == 0:
            print(
                f"iter_dt {train.iter_dt * 1000:.2f}ms; iter {train.iter_num}: train loss "
                f"{train.loss.item():.5f}")

    trainer = Trainer(train_config, model_to_train, data)
    trainer.set_callback('on_batch_end', batch_end_callback)
    return trainer


def data_by_blocks(data, block_size):
    ind = 0
    tokens = []
    targets = []
    while ind < len(data) - block_size:
        tokens.append(data[ind:ind + block_size])
        targets.append(data[ind + 1:ind + block_size + 1])
        ind += 1
    return tokens, targets


def clean_string(input_string):
    output_string = ""
    for i in input_string:
        if i.isalnum() or i == " " or i in "!(),.:;-":
            output_string += i.lower()
    output_string = " ".join(output_string.split())
    return output_string


class TrainSet(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]


def perform_inversion(ar, sentence, embedding_dim, iterations=20000):
    # model.eval()
    # for param in ar.parameters():
    #     param.requires_grad = False
    vec = np.random.uniform(0, VOCAB_SIZE, (1, BLOCK_SIZE, embedding_dim))
    input_vec = torch.tensor(vec, dtype=torch.float, requires_grad=True)
    optimizer = torch.optim.Adam([input_vec], lr=1e-3)

    one_hot = torch.zeros(len(sentence[0]), VOCAB_SIZE)
    criterion = torch.nn.CrossEntropyLoss()
    for i in range(len(sentence[0])):
        one_hot[i][sentence[0][i]] = 1
    for _ in range(iterations):
        optimizer.zero_grad()
        probs = ar.generate(idx=None,input_vector=input_vec, max_new_tokens=8)
        # logits, model_loss = ar.forward(None, sentence, input_vec)
        # probs = F.softmax(logits, dim=-1)
        entropy_loss = criterion(probs, one_hot)
        entropy_loss.backward()
        optimizer.step()

    return input_vec


if __name__ == '__main__':
    with open("alice_in_wonderland.txt") as f:
        dataset = f.read()

    dataset = clean_string(dataset)
    e = mingpt.bpe.BPETokenizer()
    tokenized_data = e(dataset)
    vocab_size = tokenized_data.unique().shape[0]
    x, y = data_by_blocks(tokenized_data[0], BLOCK_SIZE)
    x = torch.stack(x)
    y = torch.stack(y)
    dataset = TrainSet(x, y)
    # dataset = TrainSet(x, y)

    model = init_model()
    model_trainer = init_trainer(model, dataset)
    if os.path.exists("ar_model_weights.pth"):
        model.load_state_dict(torch.load("ar_model_weights.pth", map_location=torch.device('cpu')))
    else:
        model_trainer.run()

    # torch.save(model.state_dict(), 'ar_model_weights.pth')
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # y = model.generate(e("for she had plenty of time as she went down"), 2)
    sentence_tokens = e("I am a little squirrel holding a walnut")
    inp = perform_inversion(model, sentence_tokens, 48, 1000)
    # logits, loss = model(None, None, inp)
    # tokenized_answer = torch.argmax(logits, dim=2)
    # prompt = prompt.to("cuda")

    print(5)
