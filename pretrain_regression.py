import argparse
import os
import uuid

import torch

from datetime import datetime
from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.metrics import r2_score

from tfmplayground.callbacks import ConsoleLoggerCallback
from tfmplayground.evaluation import get_openml_predictions, TOY_TASKS_REGRESSION
from tfmplayground.interface import NanoTabPFNRegressor
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors import PriorDumpDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed, make_global_bucket_edges

parser = argparse.ArgumentParser()

parser.add_argument("--priordump", type=str, default="workdir/dumps/50x3_1280k_regression.h5", help="path to the prior dump")
parser.add_argument("--heads", type=int, default=6, help="number of attention heads")
parser.add_argument("--embeddingsize", type=int, default=192, help="the size of the embeddings used for the cells")
parser.add_argument("--hiddensize", type=int, default=768, help="size of the hidden layer of the mlps")
parser.add_argument("--layers", type=int, default=6, help="number of transformer layers")
parser.add_argument("--batchsize", type=int, default=1, help="batch size used during training (before gradient accumulation)")
parser.add_argument("--accumulate", type=int, default=1, help="number of gradients to accumulate before updating the weights")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--steps", type=int, default=100, help="number of steps that constitute one epoch (important for lr scheduler)")
parser.add_argument("--epochs", type=int, default=10000, help="number of epochs to train for")
parser.add_argument("--loadcheckpoint", type=str, default=None, help="checkpoint from which to continue training")
parser.add_argument("--n_buckets", type=int, default=100, help="number of buckets for the data loader")
parser.add_argument("--experiments-dir", type=str, default='workdir/experiments/regression')
parser.add_argument("--name", type=str, default='test')

args = parser.parse_args()

ts = datetime.now().strftime("%y%m%d-%H%M%S")
uid = uuid.uuid4().hex[:8]
e_name = args.name.strip()
e_id = f"{ts}-{uid}-{e_name}"
e_root = os.path.join(args.experiments_dir, e_name)
e_dir = os.path.join(e_root, e_id)
os.makedirs(e_dir, exist_ok=True)
buckets_path = os.path.join(e_dir, f"{e_id}-buckets.pth")
print(f"experiment id: {e_id}")

set_randomness_seed(2402)

device = get_default_device()
ckpt = None
if args.loadcheckpoint:
    ckpt = torch.load(args.loadcheckpoint)

prior = PriorDumpDataLoader(filename=args.priordump, num_steps=args.steps, batch_size=args.batchsize, device=device, starting_index=args.steps*(ckpt['epoch'] if ckpt else 0))

model = NanoTabPFNModel(
    num_attention_heads=args.heads,
    embedding_size=args.embeddingsize,
    mlp_hidden_size=args.hiddensize,
    num_layers=args.layers,
    num_outputs=args.n_buckets,
)

bucket_edges = make_global_bucket_edges(
    filename=args.priordump,
    n_buckets=args.n_buckets,
    device=device,
)

torch.save(
    bucket_edges,
    buckets_path,
)

if ckpt:
    model.load_state_dict(ckpt['model'])

dist = FullSupportBarDistribution(bucket_edges)

class EvaluationLoggerCallback(ConsoleLoggerCallback):
    def __init__(self, tasks):
        self.tasks = tasks

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        regressor = NanoTabPFNRegressor(model, dist, device)
        predictions = get_openml_predictions(model=regressor, tasks=self.tasks)
        scores = []
        for dataset_name, (y_true, y_pred, _) in predictions.items():
            scores.append(r2_score(y_true, y_pred))
        avg_score = sum(scores) / len(scores)
        print(f'epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.2f} | avg r2 score {avg_score:.3f}',
              flush=True)


callbacks = [EvaluationLoggerCallback(TOY_TASKS_REGRESSION)]

trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=dist,
    epochs=args.epochs,
    accumulate_gradients=args.accumulate,
    lr=args.lr,
    device=device,
    callbacks=callbacks,
    ckpt=ckpt,
    run_name=e_name,
    experiment_id=e_id,
    experiment_dir=e_dir,
)
