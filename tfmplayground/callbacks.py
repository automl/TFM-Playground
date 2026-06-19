from abc import ABC, abstractmethod


class Callback(ABC):
    @abstractmethod
    def on_epoch_end(self, epoch, epoch_time, loss, model, **kwargs):
        pass

    @abstractmethod
    def close(self):
        pass


class BaseLoggerCallback(Callback):
    pass


class ConsoleLoggerCallback(BaseLoggerCallback):
    def on_epoch_end(self, epoch, epoch_time, loss, model, **kwargs):
        print(f"Epoch {epoch:5d} | Time {epoch_time:5.2f}s | Mean Loss {loss:5.2f}", flush=True)

    def close(self):
        pass


class TensorboardLoggerCallback(BaseLoggerCallback):
    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir)

    def on_epoch_end(self, epoch, epoch_time, loss, model, **kwargs):
        self.writer.add_scalar("Loss/train", loss, epoch)
        self.writer.add_scalar("Time/epoch", epoch_time, epoch)

    def close(self):
        self.writer.close()


class WandbLoggerCallback(BaseLoggerCallback):
    def __init__(self, project, name=None, config=None, log_dir=None):
        try:
            import wandb

            self.wandb = wandb
            wandb.init(project=project, name=name, id=name, config=config, dir=log_dir, resume="allow")
        except ImportError as e:
            raise ImportError("wandb is not installed. Install it with: pip install wandb") from e

    def on_epoch_end(self, epoch, epoch_time, loss, model, **kwargs):
        log_dict = {"epoch": epoch, "loss": loss, "epoch_time": epoch_time}
        self.wandb.log(log_dict)

    def close(self):
        self.wandb.finish()
