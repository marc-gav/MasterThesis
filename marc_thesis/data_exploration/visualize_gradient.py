from transformers import BertTokenizer, BertForMaskedLM
import torch
import pickle
import re
import random
import tqdm
from loguru import logger
from file_manager import store_file_target, load_file_target


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
model.train()

target_word = input("Enter target word: ")
sentences = load_file_target(target_word, f"{target_word}_sentences.pickle")

# randomly select one sentence
sentence, sentence_idx = random.choice(sentences)

# substitue the target word with [MASK]
sentence = re.sub(r"\b" + target_word + r"\b", "[MASK]", sentence)

# tokenize the sentence
inputs = tokenizer(sentence, return_tensors="pt")

mask_id = tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
mask_pos = torch.where(inputs["input_ids"] == mask_id)[1]

# get the gradients of the input tokens
outputs = model(**inputs)
logits = outputs.logits

# logger.info the top 5 predictions for [MASK]
logger.info('The top 5 predictions for "[MASK]" are:')
logger.info(
    tokenizer.convert_ids_to_tokens(
        torch.topk(logits[0, mask_pos], 5).indices[0].tolist()
    )
)

target_word_id = tokenizer.convert_tokens_to_ids([target_word])[0]

target_word_logit = logits[0, mask_pos, target_word_id]

# backpropagate from the target_word_logit
# zero gradients
model.zero_grad()
target_word_logit.backward()
embeddings_gradient = model.bert.embeddings.word_embeddings.weight.grad

# from the embeddings_gradient extract only the entries for the input tokens
input_embeddings_gradient = embeddings_gradient[inputs["input_ids"][0]]

# use L1 norm of these vectors
input_embeddings_gradient_norm = torch.norm(
    input_embeddings_gradient, dim=1, p=1
)

# visualize them with respect to the subwords
word_gradient = []
for i, subword in enumerate(
    tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
):
    word_gradient.append((subword, input_embeddings_gradient_norm[i].item()))

# visualize in matplotlib
import matplotlib.pyplot as plt

plt.bar(
    range(len(word_gradient)),
    [x[1] for x in word_gradient],
    tick_label=[x[0] for x in word_gradient],
)
plt.xticks(rotation=90)
plt.show()
