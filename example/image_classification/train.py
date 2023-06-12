from rdl.engine.trainer import SimpleTrainer
from rdl.engine import hook, event
from rdl.utils.common import set_all_seed


class ImgClasssificationTrainer(SimpleTrainer):
    def run_step(self):
        pass

    def run_val(self):
        pass

    def run_test(self):
        pass


def build_trainer(cfg):
    # Set seed
    set_all_seed(0)

    trainer = ImgClasssificationTrainer()

    # ------
    # 1. Build dataloader
    # ------
    # TBD

    # ------
    # 2. Build model
    # ------
    model = None
    trainer.set_model(model)

    # ------
    # 3. Build criterion
    # ------
    criterion = None
    trainer.set_criterion(criterion)

    # ------
    # 4. Build Optimizer
    # ------
    optimizer = None
    trainer.set_optimizer(optimizer)

    # ------
    # 5. Build lr_scheduler
    # ------
    lr_scheduler = None
    trainer.set_lr_scheduler(lr_scheduler)

    # ------
    # 6. Build hooks
    # ------
    hooks = []
    trainer.register_hooks(hooks)

    return trainer
