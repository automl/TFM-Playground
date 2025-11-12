import hydra
from omegaconf import DictConfig

from tqdm import tqdm

from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
import torch.multiprocessing as mp

from tfmplayground.training.callbacks import ConsoleLoggerCallback, WandbLoggerCallback
from tfmplayground.evaluation import get_openml_predictions, TOY_TASKS_CLASSIFICATION, TABARENA_TASKS
from tfmplayground.interface import NanoTabPFNClassifier
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors.tabicl import TabICLPriorDataLoader
from tfmplayground.priors.dataloader import PriorDumpDataLoader
from tfmplayground.priors.dataset import PriorDumpDataset
from tfmplayground.utils import set_randomness_seed
from tfmplayground.training.trainer import BaseTrainer
from tfmplayground.training.util import tqdm_on_main

set_randomness_seed(2402)

class ToyEvaluationLoggerCallback(ConsoleLoggerCallback):
    def __init__(self, tasks):
        self.tasks = tasks

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        classifier = NanoTabPFNClassifier(model, "cuda")
        predictions = get_openml_predictions(model=classifier, tasks=self.tasks)
        scores = []
        for dataset_name, (y_true, y_pred, y_proba) in predictions.items():
            scores.append(accuracy_score(y_true, y_pred))
        avg_score = sum(scores) / len(scores)
        tqdm_on_main(f'epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.2f} | avg accuracy {avg_score:.3f}')

class ProductionEvaluationLoggerCallback(WandbLoggerCallback):
    def __init__(self, project: str, name: str = None, config: dict = None, log_dir: str = None):
        super().__init__(project, name, config, log_dir)

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        classifier = NanoTabPFNClassifier(model, "cuda")
        predictions = get_openml_predictions(model=classifier, classification=True, tasks=TABARENA_TASKS)
        scores = []
        for dataset_name, (y_true, y_pred, y_proba) in predictions.items():
            scores.append(roc_auc_score(y_true, y_proba, multi_class='ovr'))
        avg_score = sum(scores) / len(scores)
        self.wandb.log({
            'epoch': epoch,
            'epoch_time': epoch_time,
            'mean_loss': loss,
            'tabarena_avg_roc_auc': avg_score
        })
        print(f'epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.2f} | avg roc auc {avg_score:.3f}',
              flush=True)

@hydra.main(version_base=None, config_path="configs", config_name="train_classification")
def main(cfg: DictConfig):
    dataset = PriorDumpDataset(
        **cfg.dataset,
        num_steps=cfg.training.steps
    )
    model = NanoTabPFNModel(
        **cfg.model,
        num_outputs=dataset.max_num_classes,
    )
    # dataset = TabICLPriorDataLoader(
    #     **cfg.dataset
    # )
    callbacks = [ToyEvaluationLoggerCallback(TOY_TASKS_CLASSIFICATION)]
    trainer = BaseTrainer(
        model=model,
        train_dataset=dataset,
        criterion=nn.CrossEntropyLoss(),
        callbacks=callbacks,
        **cfg.training
    )
    model = trainer.train()

if __name__ == "__main__":
    main()