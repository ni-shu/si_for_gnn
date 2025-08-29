from tqdm import tqdm
import onnx
import onnxoptimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import torch.optim as optim

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_utils import synth_data

class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features, activation=None):
        super(GCNLayer, self).__init__()
        self.activation = activation
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, Dinv_A, X):

        Dinv_A_X = torch.matmul(Dinv_A, X)
        Dinv_A_X_W = torch.matmul(Dinv_A_X, self.weight)

        if self.activation:
            return self.activation(Dinv_A_X_W)
        else:
            return Dinv_A_X_W


class GAPLayer(nn.Module):

    def forward(self, X):
        # Average the node features across all nodes
        return torch.mean(X, dim=1)


class CAMLayer(nn.Module):
    def __init__(self, weights):
        super(CAMLayer, self).__init__()
        self.cam_weights = nn.Parameter(
            weights.t(), requires_grad=False
        )  # Transpose weights and set requires_grad=False

    def forward(self, F):
        cam = F @ self.cam_weights
        cam_output = torch.relu(cam)
        return cam_output


def create_model(num_features=100, gcn_layers=3):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            weight_size = 10
            self.gcn_list = nn.ModuleList(
                [
                    GCNLayer(num_features, weight_size, activation=F.relu),
                ]
            )
            assert gcn_layers > 0
            for i in range(1, gcn_layers):
                self.gcn_list.append(
                    GCNLayer(weight_size, weight_size, activation=F.relu)
                )
            self.gap = GAPLayer()
            self.fc = nn.Linear(weight_size, 2, bias=False)

        def forward(self, Dinv_A, X):
            for gcn in self.gcn_list:
                X = gcn(Dinv_A, X)
            X = self.gap(X)
            X = self.fc(X)
            return X

    return Model()


def create_cam_model(original_model):
    class CAMModel(nn.Module):
        def __init__(self, original_model):
            super(CAMModel, self).__init__()
            self.gcn_list = original_model.gcn_list
            self.fc = original_model.fc
            self.cam = CAMLayer(original_model.fc.weight)
            self.gap = original_model.gap

        def forward(self, Dinv_A, X):
            for gcn in self.gcn_list:
                X = gcn(Dinv_A, X)
            X1 = self.gap(X)
            return self.cam(X), self.fc(X1)

    return CAMModel(original_model)


def load_init_model(num_features):
    torch.manual_seed(1)
    model = create_model(num_features)
    cam_model = create_cam_model(model)
    return model, cam_model


def load_pretrained_model(model_path, num_features, layer_size):
    model = create_model(num_features, layer_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_pretrained_cam_model(cam_model_path, num_features, layer_size):
    model = create_model(num_features, layer_size)
    cam_model = create_cam_model(model)
    cam_model.load_state_dict(torch.load(cam_model_path))
    cam_model.eval()
    return cam_model


def train(
    model, train_loader, val_loader, num_epochs=10, chunk_length=0, chunk_slide_width=0
):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            (A, X), labels = batch
            if chunk_length > 0:
                A, X = synth_data.chunk_time_series(A, X, chunk_length, chunk_slide_width)
            A, X, labels = A.to(device), X.to(device), labels.to(device)
            index_labels = torch.argmax(labels, dim=1)  # one-hot to index
            optimizer.zero_grad()
            outputs = model(A, X)
            loss = criterion(outputs, index_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            probas = F.softmax(outputs)
            predicted = (probas > 0.5).float()
            correct_train += (predicted == labels).all(dim=1).sum().item()
            total_train += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"
            ):
                (A, X), labels = batch
                if chunk_length > 0:
                    A, X = eeg_data.chunk_time_series(
                        A, X, chunk_length, chunk_slide_width
                    )
                A, X, labels = A.to(device), X.to(device), labels.to(device)
                index_labels = torch.argmax(labels, dim=1)  # one-hot to index
                optimizer.zero_grad()
                outputs = model(A, X)
                loss = criterion(outputs, index_labels)
                val_loss += loss.item()
                probas = F.softmax(outputs)
                predicted = (probas > 0.5).float()
                correct_val += (predicted == labels).all(dim=1).sum().item()
                total_val += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
        )

    results = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
    }
    return model, results


def save_as_onnx(model, output_path, num_features, save_cam=False):
    model.eval()

    num_nodes = 5

    dummy_A = torch.ones(1, num_nodes, num_nodes, dtype=torch.float32)
    dummy_X = torch.ones(1, num_nodes, num_features, dtype=torch.float32)

    if save_cam:
        torch.onnx.export(
            model,
            (dummy_A, dummy_X),
            output_path,
            dynamic_axes={
                # Make batch size and number of nodes variable (dynamic)
                "input_A": {0: "batch_size", 1: "num_nodes", 2: "num_nodes"},
                "input_X": {0: "batch_size", 1: "num_nodes"},
                "output_cam": {0: "batch_size"},
                "output_logits": {0: "batch_size"},
            },
            input_names=["input_A", "input_X"],
            output_names=["output_cam", "output_logits"],
            verbose=False,
        )
    else:
        torch.onnx.export(
            model,
            (dummy_A, dummy_X),
            output_path,
            dynamic_axes={
                # Make batch size and number of nodes variable (dynamic)
                "input_A": {0: "batch_size", 1: "num_nodes", 2: "num_nodes"},
                "input_X": {0: "batch_size", 1: "num_nodes"},
                "output": {0: "batch_size"},
            },
            input_names=["input_A", "input_X"],
            output_names=["output"],
            verbose=False,
        )

    onnx_model = onnx.load(output_path)
    # remove path info (if verbose option is set to True, the path info will be included in the model)
    for node in onnx_model.graph.node:
        node.doc_string = ""

    passes = ["eliminate_identity"]
    optimized_model = onnxoptimizer.optimize(onnx_model, passes)
    onnx.save(optimized_model, output_path)

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)


if __name__ == "__main__":
    pass

