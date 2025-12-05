from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR" #Supress NNPACK warnings
import torch
from torch.nn import Module, Conv2d, BatchNorm2d, ModuleList, Parameter

class SkipLayerBias(Module):
    """
        A "skip layer" module for residual connection, to overcome
        the vanishing gradient problem.
    """
    def __init__(self, channels, reach, scale=1):
        super(SkipLayerBias, self).__init__()
        self.conv = Conv2d(channels, channels, kernel_size=reach*2+1, padding=reach, bias=False)
        self.bn = BatchNorm2d(channels)
        self.scale = scale

    def forward(self, x):
        z = x + self.scale * self.bn(self.conv(x))
        swished = z * torch.sigmoid(z)
        return swished

class PreTrainedModel(Module):
    """
        A pre-trained HexHex model.
    """
    def __init__(self, board_size, layers, intermediate_channels, reach, export_mode):
        super().__init__()
        self.board_size = board_size
        self.conv = Conv2d(2, intermediate_channels, kernel_size=2*reach+1, padding=reach-1)
        self.skiplayers = ModuleList([SkipLayerBias(intermediate_channels, 1) for _ in range(layers)])
        self.policyconv = Conv2d(intermediate_channels, 1, kernel_size=2*reach+1, padding=reach, bias=False)
        self.bias = Parameter(torch.zeros(board_size**2))
        self.export_mode = export_mode

    def forward(self, x):
        x_sum = torch.sum(x[:, :, 1:-1, 1:-1], dim=1).view(-1,self.board_size**2)
        x = self.conv(x)
        for skiplayer in self.skiplayers:
            x = skiplayer(x)
        if self.export_mode:
            return self.policyconv(x).view(-1, self.board_size ** 2) + self.bias

        #Illegal moves given huge negative bias, so they are never selected for play
        illegal = x_sum * torch.exp(torch.tanh((x_sum.sum(dim=1)-1)*1000)*10).unsqueeze(1).expand_as(x_sum) - x_sum
        return self.policyconv(x).view(-1, self.board_size**2) + self.bias - illegal

class PreTrainedRotatedModel(Module):
    """
        A HexHex model which plays on a rotated board. During the
        models forward pass, moves are flipped.
    """
    def __init__(self, model):
        super(PreTrainedRotatedModel, self).__init__()
        self.board_size = model.board_size
        self.internal_model = model

    def forward(self, x):
        x_flip = torch.flip(x, [2, 3])
        y_flip = self.internal_model(x_flip)
        y = torch.flip(y_flip, [1])
        return (self.internal_model(x) + y) / 2

class HexHexAgent(AgentBase):
    """
        A HexHex-based agent (https://github.com/harbecke/HexHex) for playing Hex.

        Loads in the pre-trained model configuration, for the intention of 
        downstream fine-tuning.
    """
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.colour = colour
        hexhex_path = "agents/group4/hexhex.pt"
        hexhex_info = torch.load(hexhex_path, weights_only=False, map_location=torch.device('cpu'))
        
        #Strip wrapper prefixes
        state = hexhex_info['model_state_dict']
        new_state = {}
        for k, v in state.items():
            if k.startswith('internal_model.'):
                new_state[k[len('internal_model.'):]] = v
            else:
                new_state[k] = v

        self.model = PreTrainedModel(
            board_size=hexhex_info['config'].getint('board_size'),
            layers=hexhex_info['config'].getint('layers'),
            intermediate_channels=hexhex_info['config'].getint('intermediate_channels'),
            reach=hexhex_info['config'].getint('reach'),
            export_mode=False
        )

        if hexhex_info['config'].getboolean('rotation_model'):
            #Model was trained on a rotated board. Re-rotate to 
            #account for this
            self.model = PreTrainedRotatedModel(self.model)

        if isinstance(self.model, PreTrainedRotatedModel):
            self.model.internal_model.load_state_dict(new_state)
        else:
            self.model.load_state_dict(new_state)

        self.model.eval() #Enable inference


    def _convert_board_format(self, board: Board, rotate: bool) -> torch.Tensor:
        """
            Converts the board layout to a Tensor.

            :param board: The Board object representing the current game board.
            :return x: A Tensor representing the current game board.
        """
        size = board.size
        if self.colour == Colour.RED:
            opp_colour = Colour.BLUE
        else:
            opp_colour = Colour.RED

        #Create board structure, with the following format:
        # [batch size, player channels, num rows (with padding), num cols (with padding)]
        x = torch.zeros((1, 2, size + 2, size + 2), dtype=torch.float32)

        for i in range(size):
            for j in range(size):
                tile = board.tiles[i][j]

                if tile.colour == self.colour:
                    x[0, 0, i+1, j+1] = 1.0
                elif tile.colour == opp_colour:
                    x[0, 1, i+1, j+1] = 1.0

        if rotate and self.colour == Colour.BLUE:
            #If playing as Blue, we want to connect left -> right.
            #Rotate the board by 90 degrees.
            x = torch.rot90(x, k=-1, dims=[2, 3])

        return x

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        size = board.size
        x_unrot = self._convert_board_format(board=board, rotate=False)
        x = self._convert_board_format(board=board, rotate=True)

        with torch.no_grad():
            logits = self.model(x)[0]
            logits = logits.reshape(size, size) #Convert to move coords

        #If playing as Blue, need to re-rotate back into "normal" coords
        if self.colour == Colour.BLUE:
            logits = torch.rot90(logits, k=1, dims=[0,1])

        #Mask out illegal positions
        occupied = (x_unrot[0, 0, 1:-1, 1:-1] + x_unrot[0, 1, 1:-1, 1:-1]) > 0
        logits[occupied] = -float('inf')

        #Select argmax
        flat_index = torch.argmax(logits).item()
        move_x = flat_index // size
        move_y = flat_index % size

        return Move(int(move_x), int(move_y))
