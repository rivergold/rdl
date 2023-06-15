from argparse import ArgumentParser
from contextlib import nullcontext
import torch
import diffusers
from diffusers import (AutoencoderKL, UNet2DConditionModel, ControlNetModel,
                       DDPMScheduler)
from transformers import AutoTokenizer, CLIPTextModel
from omegaconf import OmegaConf
from accelerate import Accelerator
from rdl.engine import hook, event
from rdl.engine.trainer import MultiModelTrainer
from rdl.utils.common import set_all_seed
from rdl.optimization.lr_scheduler import build_lr_scheduler
#
from dataset import build_train_dataloader


class ControlnetTrainer(MultiModelTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, batch_sample):
        pass

    def run_step(self, batch_sample):
        with self.accelerator.accumulate(
                self.model['controlnet']
        ) if self.cfg.accelerate.enable else nullcontext() as gs:
            batch_rgb_img = batch_sample['batch_rgb_img'].cuda()
            batch_condition_rgb_img = batch_sample[
                'batch_condition_rgb_img'].cuda()
            batch_token_ids = batch_sample['batch_token_ids'].cuda()

            # Latent
            batch_latent = self.model['vae'].encode(
                batch_rgb_img).latent_dist.sample()
            batch_size = batch_latent.shape[0]
            batch_latent = batch_latent * self.model[
                'vae'].config.scaling_factor

            # Noise
            batch_noise = torch.randn_like(batch_latent)

            # Timestep
            batch_timestep = torch.randint(
                0,
                self.model['noise_scheduler'].config.num_train_timesteps,
                (batch_size, ),
                device=batch_latent.device)
            batch_timestep = batch_timestep.long()

            # Add noise
            batch_noisy_latent = self.model['noise_scheduler'].add_noise(
                batch_latent, batch_noise, batch_timestep)

            # Text embedding
            batch_text_embedding = self.model['text_encoder'](
                batch_token_ids)[0]

            batch_down_block_x, batch_mid_block_x = self.model['controlnet'](
                batch_noisy_latent,
                batch_timestep,
                encoder_hidden_states=batch_text_embedding,
                controlnet_cond=batch_condition_rgb_img,
                return_dict=False,
            )

            # Predict noise
            batch_pred_noise = self.model['unet'](
                batch_noisy_latent,
                batch_timestep,
                encoder_hidden_states=batch_text_embedding,
                down_block_additional_residuals=[
                    each_x for each_x in batch_down_block_x
                ],
                mid_block_additional_residual=batch_mid_block_x).sample

            target = batch_noise

            loss = self.criterion(batch_pred_noise, target)
            self.optimizer.zero_grad()
            if self.cfg.accelerate.enable:
                self.accelerator.backward(loss)
            else:
                loss.backward()
            self.optimizer.step()
            print(f"loss: {loss}")


def build_trainer(cfg):
    set_all_seed()

    # Trainer
    trainer = ControlnetTrainer(cfg)

    # Accelerator
    if cfg.accelerate.enable:
        accelerator = Accelerator(log_with='tensorboard',
                                  logging_dir=cfg.work_dir,
                                  **cfg.accelerate.kwargs)
        trainer.set_accelerator(accelerator)

    # Dataloader
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.Tokenizer.model_card_name,
        subfolder='tokenizer',
        use_fast=False,
        **cfg.model.Tokenizer.kwargs)
    train_dataloader = build_train_dataloader(cfg, tokenizer)
    trainer.set_train_dataloder(train_dataloader)

    # Model
    text_encoder = CLIPTextModel.from_pretrained(
        cfg.model.TextEncoder.model_card_name,
        subfolder='text_encoder',
        **cfg.model.TextEncoder.kwargs)
    vae = AutoencoderKL.from_pretrained(
        cfg.model.AutoencoderKL.model_card_name,
        subfolder='vae',
        **cfg.model.AutoencoderKL.kwargs)
    unet = UNet2DConditionModel.from_pretrained(cfg.model.Unet.model_card_name,
                                                subfolder='unet',
                                                **cfg.model.Unet.kwargs)
    controlnet = ControlNetModel.from_unet(unet)
    noise_scheduler = DDPMScheduler.from_pretrained(
        cfg.model.NoiseScheduler.model_card_name, subfolder='scheduler')

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    trainer.set_model('text_encoder', text_encoder, cuda_id=0)
    trainer.set_model('vae', vae, cuda_id=0)
    trainer.set_model('unet', unet, cuda_id=0)
    trainer.set_model('controlnet', controlnet, cuda_id=0)
    trainer.set_model('noise_scheduler', noise_scheduler, cuda_id=None)

    # Criterion
    criterion = torch.nn.MSELoss()
    trainer.set_criterion(criterion)

    # Optimizer
    params_to_optimize = controlnet.parameters()
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=cfg.train.lr,
                                  **cfg.optimizer.kwargs)
    trainer.set_optimizer(optimizer)

    # lr_scheduler
    lr_scheduler = build_lr_scheduler('MultiStepLR',
                                      optimizer,
                                      max_epoch=cfg.train.max_epoch,
                                      milestones=[120, 200, 260],
                                      gamma=0.1)
    trainer.set_lr_scheduler(lr_scheduler)

    # Hooks
    hooks = [
        # Load pretrained checkpoint
        hook.LoadPretrainedCheckpointHook(cfg.train.pretrained_checkpoint),
        # Resume checkpoint hook
        hook.ResumeCheckpointHook(cfg.train.resume_from),
        # Lrschedule hook
        hook.LRSchedulerHook(),
        # Log hook
        # hook.LogHook(base_out_dir=cfg.work_dir,
        #              exp_name=cfg.exp_name,
        #              step_period=cfg.vis.log_step_period),
        # Tensorboard hook
        hook.TensorBoardWriteHook(
            event.TensorboardWriter(base_out_dir=cfg.work_dir,
                                    exp_name=cfg.exp_name,
                                    enable_draw_img=False),
            step_period=cfg.vis.tensorboard_step_period,
        ),
        # Visualization hook
        hook.VisualizationHook(base_out_dir=cfg.work_dir,
                               exp_name=cfg.exp_name,
                               step_period=cfg.vis.visualzation_step_period,
                               color_mode='rgb'),
        # Checkpoint save hook
        # hook.SaveCheckpointHook(
        #     base_out_dir=cfg.work_dir,
        #     exp_name=cfg.exp_name,
        #     epoch_period=cfg.vis.save_checkpoint_epoch_period),
    ]
    trainer.register_hooks(hooks)

    return trainer


def main(cfg):
    # print(type(cfg.optimizer.kwargs.betas))
    trainer = build_trainer(cfg)
    trainer.train(start_step=0, max_epoch=cfg.train.max_epoch)


if __name__ == '__main__':
    opt_parser = ArgumentParser()
    opt_parser.add_argument('--cfg_path', type=str, required=True)
    opt_parser.add_argument('--work_dir', type=str, default=None)
    opt_parser.add_argument('--flag_dir', type=str, default=None)
    opt = opt_parser.parse_args()

    cfg = OmegaConf.load(opt.cfg_path)
    if opt.work_dir is not None:
        cfg.work_dir = opt.work_dir
    main(cfg)