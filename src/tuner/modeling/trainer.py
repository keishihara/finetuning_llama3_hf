from deepspeed.runtime.lr_schedules import WarmupDecayLR
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    Trainer,
)
from tuner.utils.distributed_utils import print_on_rank_0


class SFTTrainer(Trainer):

    def create_scheduler(self, num_training_steps: int, optimizer: Optimizer = None) -> LambdaLR:
        """
        Setup the scheduler. The optimizer of this trainer must have been set up either
        before this method is called or passed as an argument.

        Args:
            num_training_steps (`int`): The number of training steps to do.
            optimizer (`Optimizer`): The optimizer

        Returns:
            LambdaLR
        """

        # Is there a way to access deepspeed config from inside the Trainer?
        if self.lr_scheduler is None:
            print_on_rank_0(
                f'Setting `WarmupDecayLR` from deepspeed with following params:\n'
                f'- steps_per_epoch={num_training_steps // self.args.num_train_epochs}\n'
                f'- num_training_steps={num_training_steps}\n'
                f'- num_warmup_steps={self.args.get_warmup_steps(num_training_steps)}',
            )
            self.lr_scheduler = WarmupDecayLR(
                optimizer=self.optimizer if optimizer is None else optimizer, # Here, optimizer will be DummyOptim, which is not a subclass of torch.optim.Optimizer
                total_num_steps=num_training_steps,
                warmup_min_lr=0.0,
                warmup_max_lr=self.args.learning_rate,
                warmup_num_steps=self.args.get_warmup_steps(num_training_steps),
            )
            self._created_lr_scheduler = True

        return self.lr_scheduler
