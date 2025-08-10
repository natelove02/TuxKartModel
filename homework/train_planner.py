"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import MLPPlanner, TransformerPlanner, CNNPlanner, load_model, save_model
from .datasets.road_dataset import RoadDataset, load_data
from .metrics import PlannerMetric

#Copilot autocomplete, homeworks, and lectures used throughout homework
#Copilot never directly edited or changed my code

def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 50,
    lr: float = 1e-4,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    if model_name == "cnn_planner":
        train_data = load_data("drive_data/train", transform_pipeline="default", shuffle=True, batch_size=batch_size, num_workers=2)
        val_data = load_data("drive_data/val", transform_pipeline="default", shuffle=False)
    else:
        train_data = load_data("drive_data/train", transform_pipeline="state_only", shuffle=True, batch_size=batch_size, num_workers=2)
        val_data = load_data("drive_data/val", transform_pipeline="state_only", shuffle=False)
    # create loss function and optimizer
    loss_func = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)



    global_step = 0
    training_metrics = PlannerMetric()
    validation_metrics = PlannerMetric()

    # training loop
    #copilot assisted in adjusting this from HW3 to HW4 (track dimensions and image for part 2)
    for epoch in range(num_epoch):
        training_metrics.reset()
        validation_metrics.reset()

        model.train()

        for batch in train_data:
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)

            if model_name == "cnn_planner":
                image = batch["image"].to(device)
                pred = model(image)
            else:
                pred = model(track_left, track_right)

            loss = loss_func(pred[waypoints_mask], waypoints[waypoints_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_metrics.add(pred, waypoints, waypoints_mask)

            logger.add_scalar("train_loss", loss.item(), global_step)
            global_step += 1

        with torch.no_grad():
            model.eval()
            for batch in val_data:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                if model_name == "cnn_planner":
                    image = batch["image"].to(device)
                    pred = model(image)
                else:
                    pred = model(track_left, track_right)
                validation_metrics.add(pred, waypoints, waypoints_mask)

        training_results = training_metrics.compute()
        val_results = validation_metrics.compute()

        logger.add_scalar("train_longitudinal_error", training_results["longitudinal_error"], global_step)
        logger.add_scalar("train_lateral_error", training_results["lateral_error"], global_step)
        logger.add_scalar("val_longitudinal_error", val_results["longitudinal_error"], global_step)
        logger.add_scalar("val_lateral_error", val_results["lateral_error"], global_step)
        print(f"Epoch {epoch+1}: Train Lon: {training_results['longitudinal_error']:.4f}, "
        f"Lat: {training_results['lateral_error']:.4f} | "
        f"Val Lon: {val_results['longitudinal_error']:.4f}, "
        f"Lat: {val_results['lateral_error']:.4f}")
       
    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))

print("Time to train")
