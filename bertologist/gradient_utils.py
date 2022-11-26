import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


def get_salience_scores(data, model):

    grad_norm = []
    grad_dot_input = []

    for input_ids, token_type_ids, attention_mask, label in zip(
        data["input_ids"],
        data["token_type_ids"],
        data["attention_mask"],
        data["label"],
    ):

        model.zero_grad()

        preds = model(
            input_ids=torch.tensor([input_ids]),
            token_type_ids=torch.tensor([token_type_ids]),
            attention_mask=torch.tensor([attention_mask]),
            labels=torch.tensor([label]),
        )

        loss = preds["loss"]

        loss.backward()

        if "albert" in dir(model):
            embeddings = model.albert.embeddings
        elif "bert" in dir(model):
            embeddings = model.bert.embeddings
        else:
            print("new model type")

        input_ids = [
            input_id
            for input_id, mask in zip(input_ids, attention_mask)
            if mask == 1
        ]
        grad = (
            embeddings.word_embeddings.weight.grad[input_ids].detach().numpy()
        )
        weights = embeddings.word_embeddings.weight[input_ids].detach().numpy()

        # scores as in https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/gradient_maps.py

        # grad_norm
        grad_norm_ = np.linalg.norm(
            grad, axis=1
        )  # Frobenius Norm#math.sqrt(sum(grad[0]**2))
        grad_norm_ /= np.sum(grad_norm_)  # normalize
        grad_norm.append(grad_norm_)

        # grad_dot_input
        grad_dot_input_ = np.sum(grad * weights, axis=-1)
        grad_dot_input_ = (
            grad_dot_input_ + np.finfo(np.float32).eps
        )  # necessary? https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/citrus/utils.py
        grad_dot_input_ = grad_dot_input_ / np.abs(grad_dot_input_).sum(
            -1
        )  # normalize
        grad_dot_input.append(grad_dot_input_)

    return {"grad_norm": grad_norm, "grad_dot_input": grad_dot_input}


def plot_grad_norm(data, save_path):

    for i, (tokens, grad_norm) in enumerate(
        zip(data["tokens"], data["grad_norm"])
    ):

        fig, ax = plt.subplots()

        y_pos = np.arange(len(grad_norm))

        ax.barh(y_pos, grad_norm, align="center", tick_label=tokens)

        ax.set_yticks(y_pos)

        ax.invert_yaxis()

        plt.savefig(f"{save_path}grad_norm_{i}.png")


def plot_grad_dot_input(data, save_path):

    for i, (tokens, grad_dot_input) in enumerate(
        zip(data["tokens"], data["grad_dot_input"])
    ):

        fig, ax = plt.subplots()

        y_pos = np.arange(len(grad_dot_input))

        ax.barh(y_pos, grad_dot_input, align="center", tick_label=tokens)

        ax.set_yticks(y_pos)

        ax.invert_yaxis()

        plt.savefig(f"{save_path}grad_dot_input_{i}.png")
