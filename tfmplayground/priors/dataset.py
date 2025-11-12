from torch.utils.data import IterableDataset
import torch
import h5py

class PriorDumpDataset(IterableDataset):
    def __init__(self, filename, num_steps, batch_size, device, starting_index=0):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        with h5py.File(self.filename, "r") as f:
            self.num_datapoints_max = f['X'].shape[0]
            if "max_num_classes" in f:
                self.max_num_classes = f["max_num_classes"][0]
            else:
                self.max_num_classes = None
            self.problem_type = f['problem_type'][()].decode('utf-8')
        self.device = device
        self.pointer = starting_index

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                self.data = f
                end = self.pointer + self.batch_size

                num_features=self.data['num_features'][self.pointer:end].max()
                x = torch.from_numpy(self.data['X'][self.pointer:end,:,:num_features])
                y = torch.from_numpy(self.data['y'][self.pointer:end])
                single_eval_pos = self.data['single_eval_pos'][self.pointer:end]

                self.pointer += self.batch_size
                if self.pointer >= self.data['X'].shape[0]:
                    print("""Finished iteration over all stored datasets! """
                          """Will start reusing the same data with different splits now.""")
                    self.pointer = 0

                yield dict(
                    x=x.to(self.device),
                    y=y.to(self.device),
                    target_y=y.to(self.device),
                    single_eval_pos=single_eval_pos[0].item()
                )

    def __len__(self):
        return self.num_steps