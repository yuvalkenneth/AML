import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

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
    sentence = [s.to(device) for s in output_sentence]
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


def analyze_attention(gpt_model, sentence, block_ind, y):
    last_block = gpt_model.get_blocks()[block_ind]
    last_block_averaged_attention = last_block.get_attention_score().mean(dim=1)[0]
    eleventh_word_attention = last_block_averaged_attention[-1]
    eleventh_word_attention = [round(w.item(), 3) for w in eleventh_word_attention]
    colored_sentence = colorize_text(e.decode(y[0])[:-1], eleventh_word_attention)

    print(colored_sentence)
    print(eleventh_word_attention)


def colorize_text(sentence, weights):
    # Written by ChatGPT
    colored_sentence = ""
    min_weight = min(weights)
    max_weight = max(weights)

    for word, weight in zip(sentence.split(), weights):
        # Normalize the weight to the range [0, 1]
        normalized_weight = (weight - min_weight) / (max_weight - min_weight)

        # Calculate the color based on the normalized weight
        red = int((1 - normalized_weight) * 255)
        green = int(normalized_weight * 255)
        blue = 0

        # Create the colored word using the RGB color values
        colored_word = f"\033[38;2;{red};{green};{blue}m{word}\033[0m"

        # Append the colored word to the colored sentence
        colored_sentence += colored_word + " "

    return colored_sentence.strip()


def run_questions():
    # Q2
    sentence_tokens = e("I am a little squirrel holding a walnut").to(device)
    inp, losses = perform_inversion(model, sentence_tokens, 768, 20, iterations=1500)
    plt.plot(losses)
    plt.title("Loss of inversion")
    plt.show()
    with torch.no_grad():
        output, logits = model.generate(idx=None, max_new_tokens=9, input_vector=inp)

        probs = F.softmax(logits, dim=-1)

        p, tokens = torch.topk(probs, k=1, dim=-1)

        for t in tokens:
            print(e.decode(t))

        # Q3
        sentence = e("and had just begun to dream that she was walking").to(device)
        generated_sentence = model.generate(sentence, max_new_tokens=2)

        analyze_attention(model, sentence, -1, generated_sentence)  # last block

        # Q4
        analyze_attention(model, sentence, 0,generated_sentence)  # first block

        # Q5
        s, sentence_probabilities = model.generate(sentence[:, :3], max_new_tokens=5,
                                                   get_probs=True)
        sentence_probabilities = sentence_probabilities[0].cpu().numpy()
        decoded_sentence = e.decode(s[0])
        print(f"log score of the sentence '{decoded_sentence}' is: {np.prod(np.log(sentence_probabilities))}")


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

    run_questions()
