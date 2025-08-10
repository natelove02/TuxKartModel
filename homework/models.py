from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

#Copilot autocomplete, homeworks, and lectures used throughout homework
#Copilot never directly edited or changed my code

class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()
        input_shape = n_track * 4  # left and right boundaries
        output_shape = n_waypoints * 2  # x and y for each waypoint
        
        self.mlp = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_shape)
        )
        self.n_track = n_track
        self.n_waypoints = n_waypoints

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        #used copilot to debug and set the correct dimensions through forward call
        lefttrack_flatten = track_left.view(track_left.size(0), -1)  
        righttrack_flatten = track_right.view(track_right.size(0), -1)

        x = torch.cat((lefttrack_flatten, righttrack_flatten), dim=1)  
        x = self.mlp(x)  
        output = x.view(x.size(0), self.n_waypoints, 2) 
        return output

class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        #copilot assisted with implementing both the extra cross attention
        self.self_att = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_att = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.mlp = torch.nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        #self.in_norm = torch.nn.LayerNorm(d_model)
        #self.out_norm = torch.nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, track_memory: torch.Tensor) -> torch.Tensor:
        x = x + self.cross_att(self.norm1(x), track_memory, track_memory)[0]
        
        x = self.norm2(x)
        x = x + self.self_att(x, x, x)[0]
    
        x = x + self.mlp(self.norm3(x))
        
        return x

class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.track_projection = nn.Linear(2, d_model)


        self.query_embed = nn.Embedding(n_waypoints, d_model)
        self.output_projection = nn.Linear(d_model, 2)
        self.network = torch.nn.ModuleList(
            [
            TransformerLayer(d_model, nhead=8) for _ in range(n_waypoints)
            ]
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        #copilot assisted with debugging and adding lines below, especially with sizing
        batch_size = track_left.size(0)
        track_points = torch.cat((track_left, track_right), dim=1)
        track_memory = self.track_projection(track_points)  # shape (b, n_track * 2, d_model)
        query_indices = torch.arange(self.n_waypoints, device=track_left.device)
        x = self.query_embed(query_indices)
        x = x.unsqueeze(0).expand(batch_size, -1, -1)
        
        for layer in self.network:
            x = layer(x, track_memory)
        
        
        waypoints = self.output_projection(x)
        return waypoints



class CNNPlanner(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1) // 2
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding = padding, stride=stride)
            self.norm = torch.nn.BatchNorm2d(out_channels)
            self.relu = torch.nn.ReLU()
            

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.relu(self.norm(self.conv(x)))

    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)
        
        layers = [32,64,128,256]
        
        cnn_layers = [
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        ]
        c1 = 32
        for c2 in layers:
            cnn_layers.append(self.Block(c1, c2, stride =2))
            c1 = c2
        cnn_layers.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        cnn_layers.append(torch.nn.Conv2d(c1, n_waypoints*2, kernel_size=1))
        self.network = torch.nn.Sequential(*cnn_layers)



    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        #copilit assisted in the dimensions and debugging of forward call
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        x = self.network(x)  # shape (b, n_waypoints * 2, 1, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x.view(x.size(0), self.n_waypoints, 2)  # shape (b, n_waypoints, 2)

MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
