
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn import functional as F

import mingpt.bpe
from mingpt.model import GPT
from mingpt.trainer import Trainer

TRAIN_ITERATIONS = 2500

TRAIN_BATCH_SIZE = 32

VOCAB_SIZE = 50257
BLOCK_SIZE = 64
LR = 5e-4


class TrainSet(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]


def init_model():
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt2'
    model_config.vocab_size = VOCAB_SIZE
    model_config.block_size = BLOCK_SIZE
    gpt = GPT(model_config)
    return gpt


def init_trainer(model_to_train, data):
    train_config = Trainer.get_default_config()
    train_config.learning_rate = LR
    train_config.max_iters = TRAIN_ITERATIONS
    train_config.batch_size = TRAIN_BATCH_SIZE
    train_config.num_workers = 2
    losses = []

    def batch_end_callback(train):
        if train.iter_num % 100 == 0:
            print(
                f"iter_dt {train.iter_dt * 1000:.2f}ms; iter {train.iter_num}: train loss "
                f"{train.loss.item():.5f}")
        losses.append(train.loss.item())

    trainer = Trainer(train_config, model_to_train, data)
    trainer.set_callback('on_batch_end', batch_end_callback)
    return trainer, losses


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


def perform_inversion(gpt, output_sentence, embedding_dim, context_size, iterations=500):
    vec = np.random.uniform(-1, 1, (1, context_size, embedding_dim))
    # vec = np.random.normal(size=(1, context_size, embedding_dim))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_vec = torch.tensor(vec, dtype=torch.float, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([input_vec], lr=0.01)
    gpt.to(device)
    output_sentence = sentence = [s.to(device) for s in output_sentence]
    inversion_loss = []
    for _ in tqdm(range(iterations)):
        inversion_loss.append(0)
        optimizer.zero_grad()
        losses = []
        output, logits = gpt.generate(idx=None, max_new_tokens=len(sentence[0]), input_vector=input_vec)
        losses = F.cross_entropy(logits, sentence[0], reduction='none')
        for loss in losses:
            inversion_loss[-1] += loss.item()
            loss.backward(retain_graph=True)
        optimizer.step()
    return input_vec, inversion_loss


if __name__ == '__main__':
    with open("alice_in_wonderland.txt") as f:
        dataset = f.read()
    # dataset = clean_string(dataset)
    e = mingpt.bpe.BPETokenizer()
    tokenized_data = e(dataset)
    vocab_size = tokenized_data.unique().shape[0]
    x, y = data_by_blocks(tokenized_data[0], BLOCK_SIZE)
    x = torch.stack(x)
    y = torch.stack(y)
    dataset = TrainSet(x, y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model()
    model_trainer, train_loss = init_trainer(model, dataset)
    if os.path.exists("gpt_model_weights.pth"):
        model.load_state_dict(torch.load("gpt_model_weights.pth", map_location=device))
    else:
        model_trainer.run()
        torch.save(model.state_dict(), "gpt_model_weights.pth")
        plt.plot(train_loss)
        plt.title("Train loss")
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.show()

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    model.to(device)

    sentence_tokens = e("I am a little squirrel holding a walnut").to(device)
    inp, losses = perform_inversion(model, sentence_tokens, 768, 20, iterations=1)

    plt.plot(losses)
    plt.title("Loss of inversion")
    plt.show()
    with torch.no_grad():

        output, logits = model.generate(idx=None, max_new_tokens=9, input_vector=inp)

        probs = F.softmax(logits, dim=-1)

        p, toks = torch.topk(probs, k=1, dim=-1)

        for t in toks:
            print(e.decode(t))

        # Q3
        y = model.generate(sentence_tokens, max_new_tokens=3)
        last_block = model.get_blocks()[-1]
        last_block_averaged_attention = last_block.get_attention_score().mean(dim=1)[0]
        print(last_block_averaged_attention[-1].sum())
        plt.figure(figsize=(8, 6))
        sns.heatmap(last_block_averaged_attention[-1].cpu().detach().unsqueeze(0),
                    annot=True, fmt=".2f", cmap="viridis")
        plt.xlabel('Token Index')
        plt.ylabel('11th Word')
        plt.title('Attention Scores Heatmap (11th Word) - Last Block')
        plt.show()

        # Q4
        first_block = model.get_blocks()[0]
        first_block_averaged_attention = first_block.get_attention_score().mean(dim=1)[0]
        plt.figure(figsize=(8, 6))
        print(first_block_averaged_attention[-1].sum())
        sns.heatmap(first_block_averaged_attention[-1].cpu().detach().unsqueeze(0),
                    annot=True, fmt=".2f", cmap="viridis")
        plt.xlabel('Token Index')
        plt.ylabel('11th Word')
        plt.title('Attention Scores Heatmap (11th Word) - First Block')
        plt.show()

        # Q5
        s, sentence_probabilities = model.generate(sentence_tokens[:, 0:3], max_new_tokens=6,
                                                   get_probs=True)[0]
        sentence_probabilities = sentence_probabilities[0].cpu().numpy()
        decoded_sentence = [e.decode(t) for t in s]
        print(f"log score of the sentence '{decoded_sentence}' is: {np.prod(np.log(sentence_probabilities))}")
