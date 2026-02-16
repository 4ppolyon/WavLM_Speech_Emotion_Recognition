import os
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
from MLPClassifier import MLPClassifier
import TransClassifier
from plot import plot_classifier_curve, plot_classifier_matrix

AUDIO_EXTENSIONS = ".wav"
POOLING_TYPES = ["mean", "max", "mean_std", "transformer"]

print(os.getcwd())

data_path = Path("./dataset/")

# Train Dev Test
train_path = data_path / "train"
dev_path = data_path / "dev"
test_path = data_path / "test"

# Embedding paths
train_embedding_path = train_path / "embedding"
dev_embedding_path = dev_path / "embedding"
test_embedding_path = test_path / "embedding"

# Pooled versions paths
train_pooled_path = train_path / "pooled"
dev_pooled_path = dev_path / "pooled"
test_pooled_path = test_path / "pooled"

saved_MLP_model_path = "./saved_MLP.pth"
saved_transformer_model_path = "./saved_transformer.pth"

LABELS = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Scared", "Disgusted", "Surprised"]

def find_emotion_T(name):
    if 'neutral' in name:
        return "01"
    elif 'happy' in name or 'joy' in name or 'positive' in name:
        return "03"
    elif 'sad' in name or 'sadness' in name or 'pain' in name:
        return "04"
    elif 'angry' in name or 'anger' in name:
        return "05"
    elif 'fear' in name:
        return "06"
    elif 'disgust' in name or 'negative' in name:
        return "07"
    elif 'ps' in name or 'surprise' in name:
        return "08"
    else:
        return "-1"


# 'emotions' list fix for classification purposes:
#     Classification values start from 0, Thus an 'n = n-1' operation has been executed for both RAVDESS and TESS databases:
def emotionfix(e_num):
    if e_num == "01":
        return 0  # neutral
    elif e_num == "02":
        return 1  # calm
    elif e_num == "03":
        return 2  # happy
    elif e_num == "04":
        return 3  # sad
    elif e_num == "05":
        return 4  # angry
    elif e_num == "06":
        return 5  # fear
    elif e_num == "07":
        return 6  # disgust
    else:
        return 7  # suprised

def load_split(file_list, base_path, is_transformer=False):
    embeddings, labels = [], []
    nb_par_classes = [0 for _ in range(len(LABELS))]

    for file_name in file_list:

        if not file_name.startswith("_") and file_name.endswith(".pt"):

            # 1) Charge l'embedding
            emb = torch.load(base_path / file_name)  # tensor

            # 2) Détecte l'émotion
            emo_code = find_emotion_T(file_name)
            if emo_code == "-1":
                emo_code = file_name[6:8]  # RAVDESS
            label = emotionfix(emo_code)
            nb_par_classes[label] += 1

            embeddings.append(emb)
            labels.append(label)
    if is_transformer:
        X = pad_sequence(embeddings, batch_first=True)
    else:
        X = torch.stack(embeddings)
    y = torch.tensor(labels)
    print("\t - ",nb_par_classes)

    return X, y, nb_par_classes

paths = {}
for name in POOLING_TYPES:
    if name != "transformer":
        paths[name] = [train_pooled_path / f"_{name}.pt", dev_pooled_path / f"_{name}.pt", test_pooled_path / f"_{name}.pt"]
    else:
        paths[name] = [train_embedding_path / f"_{name}.pt", dev_embedding_path / f"_{name}.pt", test_embedding_path / f"_{name}.pt"]

batch_size = 16

pooling_type = "mean"  # "max", "mean_std", "transformer"
train_MLP_path, dev_MLP_path, test_MLP_path = paths[pooling_type]

train_files = torch.load(train_MLP_path)
dev_files = torch.load(dev_MLP_path)
test_files = torch.load(test_MLP_path)

print("Number of example for each class\nTrain :")
X_train, y_train, nb_train = load_split(train_files, train_MLP_path.parents[0] / "mean")
print("dev  :")
X_dev, y_dev, nb_dev = load_split(dev_files, dev_MLP_path.parents[0] / "mean")
print("Test  :")
X_test, y_test, nb_test = load_split(test_files, test_MLP_path.parents[0] / "mean")

print(f"[MLP] Loaded splits:")
print("Train :", X_train.shape, y_train.shape)
print("dev   :", X_dev.shape, y_dev.shape)
print("Test  :", X_test.shape, y_test.shape)

MLP_train_dataset = TensorDataset(X_train, y_train)
MLP_dev_dataset = TensorDataset(X_dev, y_dev)
MLP_test_dataset = TensorDataset(X_test, y_test)

MLP_train_loader = DataLoader(MLP_train_dataset, batch_size=batch_size, shuffle=True)
MLP_dev_loader = DataLoader(MLP_dev_dataset, batch_size=batch_size)
MLP_test_loader = DataLoader(MLP_test_dataset, batch_size=batch_size)

input_dim = X_test.shape[1]
mlp_model = MLPClassifier(input_dim=input_dim, num_classes=len(LABELS))

#### If you want to train

train_losses, dev_losses, train_acc, dev_acc = mlp_model.train_mlp(
    MLP_train_loader, MLP_dev_loader,
    num_epochs=30, lr=1e-4, weight_decay=1e-3
)
plot_classifier_curve(
    mlp_model.name(),
    train_losses, dev_losses,
    train_acc, dev_acc
)

#### If you want to load

# mlp_model.load(saved_MLP_model_path)

test_loss, test_acc, test_preds, test_yb = mlp_model.evaluate_MLP(MLP_test_loader)
print(f"[{mlp_model.name()}] Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
conf_matrix = confusion_matrix(test_yb.numpy(), test_preds.numpy())
plot_classifier_matrix(mlp_model.name(), conf_matrix, LABELS)
print(classification_report(test_yb, test_preds, target_names=[l for l in LABELS]))
mlp_model.save(saved_MLP_model_path)

############################
## Transformer Classifier ##
############################

train_transformer_path, dev_transformer_path, test_transformer_path = paths["transformer"]

train_files = torch.load(train_transformer_path)
dev_files = torch.load(dev_transformer_path)
test_files = torch.load(test_transformer_path)

print("Number of example for each class\nTrain :")
X_train, y_train, nb_train = load_split(train_files, train_transformer_path.parents[0], is_transformer=True)
print("dev  :")
X_dev, y_dev, nb_dev = load_split(dev_files, dev_transformer_path.parents[0], is_transformer=True)
print("Test  :")
X_test, y_test, nb_test = load_split(test_files, test_transformer_path.parents[0], is_transformer=True)

print(f"[TRANSFORMER] Loaded splits:")
print("Train :", X_train.shape, y_train.shape)
print("dev   :", X_dev.shape, y_dev.shape)
print("Test  :", X_test.shape, y_test.shape)

transformer_train_loader = DataLoader(TransClassifier.PadDataset(X_train, y_train), batch_size=batch_size, shuffle=True, collate_fn=TransClassifier.collate_fn)
transformer_dev_loader = DataLoader(TransClassifier.PadDataset(X_dev, y_dev), batch_size=batch_size, collate_fn=TransClassifier.collate_fn)
transformer_test_loader = DataLoader(TransClassifier.PadDataset(X_test, y_test), batch_size=batch_size, collate_fn=TransClassifier.collate_fn)

transformer_model = TransClassifier.TransformerClassifier(
    input_dim=X_train.shape[2],  # nombre de features par timestep
    num_outputs=len(LABELS),               # classification en x classes
    d_model=64,
    n_layers=2,
    n_heads=4,
    ff_dim=128,
    dropout=0.1
)

#### If you want to train

train_losses, dev_losses, train_acc, dev_acc = transformer_model.train_transformer(
    transformer_train_loader, transformer_dev_loader,
    num_epochs=30, lr=1e-4, weight_decay=1e-3
)
plot_classifier_curve(
    transformer_model.name(),
    train_losses, dev_losses,
    train_acc, dev_acc
)

#### If you want to load

# transformer_model.load(saved_transformer_model_path)

test_loss, test_acc, test_preds, test_yb = transformer_model.evaluate_transformer(transformer_test_loader)
print(f"[{transformer_model.name()}] Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
conf_matrix = confusion_matrix(test_yb.numpy(), test_preds.numpy())
plot_classifier_matrix(transformer_model.name(), conf_matrix, LABELS)
print(classification_report(test_yb, test_preds, target_names=[l for l in LABELS]))
transformer_model.save(saved_transformer_model_path)