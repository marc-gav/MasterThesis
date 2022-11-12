import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClusteredWordsDataset(Dataset):
    def __init__(self, data_dir: str):
        df = pd.read_csv(data_dir, sep="|", index_col=False)
        self.num_classes = len(df["cluster_label"].unique())
        self.salience_score = df["salience score"]
        data = df["word"]
        # Get word dict from data
        self.word_dict = {word: i for i, word in enumerate(data)}
        # One hot encodding representation of data using word_dict
        self.vocab_size = len(self.word_dict)
        self.data = torch.zeros(len(data), self.vocab_size)

        for i, word in enumerate(data):
            self.data[i][self.word_dict[word]] = 1
        self.labels = torch.tensor(
            df["cluster_label"].to_numpy(), dtype=torch.long
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_num_classes(self):
        return self.num_classes

    def get_salience_score(self):
        return self.salience_score

    def get_vocab_size(self):
        return self.vocab_size

    def get_word_dict(self):
        return self.word_dict


class ClusteredWordsClassifier(torch.nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, num_classes: int
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        self.linear = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = self.linear(x)
        return x


def train(
    model,
    train_loader,
    optimizer,
    criterion,
    num_epochs,
    device,
    save_path,
    save_every,
):
    model.train()
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(train_loader):
            data = data
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )
        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), f"{save_path}/model_{epoch}.pt")
    torch.save(model.state_dict(), f"{save_path}/model_final.pt")


def test(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(
            f"Accuracy of the model on the test set: {100 * correct / total:.2f}%"
        )


def main():
    data_dir = "data/plant_top_attention_words.csv"
    save_path = "models"
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.001
    save_every = 2
    dataset = ClusteredWordsDataset(data_dir)
    num_classes = dataset.get_num_classes()
    salience_score = dataset.get_salience_score()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    model = ClusteredWordsClassifier(
        num_embeddings=len(dataset),
        embedding_dim=dataset.get_vocab_size(),
        num_classes=num_classes,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(
        model,
        train_loader,
        optimizer,
        criterion,
        num_epochs,
        device,
        save_path,
        save_every,
    )
    test(model, test_loader, device)


if __name__ == "__main__":
    main()
