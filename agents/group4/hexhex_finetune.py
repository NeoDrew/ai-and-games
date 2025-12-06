import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import torch.nn.functional as F
import copy
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from agents.group4.HexHexMinimaxAgent import PreTrainedModel

BOARD_SIZE = 11
DATASET_PATH = "agents/group4/hex_dataset.pt"
OUTPUT_MODEL_PATH = "agents/group4/hexhex_finetuned.pt"
BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class HexDataset(Dataset):
    """PyTorch Dataset for Hex state -> next move"""
    def __init__(self, data_list, augment=True):
        self.augment = augment
        # Store raw data first
        if not isinstance(data_list, list):
            raise ValueError("Expected a list of (board_tensor, move_index) tuples")
        self.data = self._augment_data(data_list) if augment else data_list

    def _augment_data(self, data):
        """Augment data by rotating and flipping boards"""
        augmented = []
        for board, move in data:
            #Original
            augmented.append((board, move))
            #90-degree rotations
            for k in [1, 2, 3]:
                rot_board = torch.rot90(board, k=k, dims=[1,2])
                row, col = divmod(move, BOARD_SIZE)
                new_row, new_col = col, BOARD_SIZE - 1 - row
                rot_move = new_row * BOARD_SIZE + new_col
                augmented.append((rot_board, rot_move))
            #Horizontal flip
            flip_board = torch.flip(board, [2])
            row, col = divmod(move, BOARD_SIZE)
            flip_move = row * BOARD_SIZE + (BOARD_SIZE - 1 - col)
            augmented.append((flip_board, flip_move))
        return augmented

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#Load dataset
print("Loading dataset...")
dataset = torch.load(DATASET_PATH, weights_only=False)
dataset = dataset[:1000] #Taking a subset for debugging
dataset = HexDataset(dataset, augment=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Loaded {len(dataset)} examples (including augmentations).")


#Load pre-trained model
print("Loading pre-trained HexHex model...")
hexhex_path = "agents/group4/pretrained_hexhex.pt"
hexhex_info = torch.load(hexhex_path, map_location="cpu", weights_only=False)
original_config = hexhex_info["config"]

model = PreTrainedModel(
    board_size=hexhex_info['config'].getint('board_size'),
    layers=hexhex_info['config'].getint('layers'),
    intermediate_channels=hexhex_info['config'].getint('intermediate_channels'),
    reach=hexhex_info['config'].getint('reach'),
    export_mode=False
)

#Load weights
state = hexhex_info['model_state_dict']
new_state = {}
for k, v in state.items():
    if k.startswith('internal_model.'):
        new_state[k[len('internal_model.'):]] = v
    else:
        new_state[k] = v
model.load_state_dict(new_state)
model.train()
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)


#Fine-tune model
print("Starting fine-tuning...")
for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    for batch in dataloader:
        boards, moves = batch
        boards = boards.to(DEVICE)
        moves = moves.to(DEVICE)

        optimiser.zero_grad()
        logits = model(boards)
        loss = criterion(logits, moves)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * boards.size(0)

    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

#Save fine-tuned model
torch.save({
    "model_state_dict": model.state_dict(),
    "config": original_config
}, OUTPUT_MODEL_PATH)
print(f"Fine-tuned model saved to {OUTPUT_MODEL_PATH}")
