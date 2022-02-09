import os
import sys
import json
import logging

from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    print("Load TPU mode")
except ImportError:
    pass


class WarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps=10000, last_epoch=-1):
        

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            return 0.99 ** step

        super(WarmupScheduler, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)


def getDevice(device):
    chk = False

    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    elif device == 'tpu':
        chk = True
        device = xm.xla_device()
        n_gpu = 0
    else:
        device = device
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return device, n_gpu, chk


class Trainer(object):
    def __init__(self, model, tokenizer, optimizer, scheduler, device=None,
                 train_batch_size=12, test_batch_size=32,
                 checkpoint_path=None, model_name=None,
                 **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        if test_batch_size is None:
            self.test_batch_size = train_batch_size

        self.model_name = model_name
        self.checkpoint_path = checkpoint_path

        self.device, self.n_gpu, self.tpu = getDevice(device)
        print(self.device)
        self.model.to(self.device)

        # logging.basicConfig(filename=f'{log_dir}/{self.model_name}-{datetime.now().date()}.log', level=logging.INFO)

    def build_dataloaders(self, train_dataset, eval_dataset=None, train_shuffle=True, eval_shuffle=False, train_test_split=0.1):
        if eval_dataset is None:
            dataset_len = len(train_dataset)
            eval_len = int(dataset_len * train_test_split)
            train_len = dataset_len - eval_len
            train_dataset, eval_dataset = random_split(
                train_dataset, (train_len, eval_len))

        train_loader = DataLoader(
            train_dataset, batch_size=self.train_batch_size, shuffle=train_shuffle)
        self.train_loader = train_loader
        eval_loader = DataLoader(
            eval_dataset, batch_size=self.test_batch_size, shuffle=eval_shuffle)
        self.eval_loader = eval_loader

        logging.info(f'''train_dataloader size: {len(train_loader.dataset)} | shuffle: {train_shuffle}
                         eval_dataloader size: {len(eval_loader.dataset)} | shuffle: {eval_shuffle}''')

    def train(self,
              epochs,
              log_steps,
              ckpt_steps,
              gradient_accumulation_steps=1):

        losses = {}
        global_steps = 0
        local_steps = 0
        step_loss = 0.0
        start_epoch = 0
        start_step = 0

        if self.checkpoint_path and self.model_name:
            if os.path.isfile(f'{self.checkpoint_path}/{self.model_name}.pth'):
                checkpoint = torch.load(
                    f'{self.checkpoint_path}/{self.model_name}.pth', map_location=self.device)
                start_epoch = checkpoint['epoch']
                losses = checkpoint['losses']
                global_steps = checkpoint['train_step']
                start_step = global_steps if start_epoch == 0 else global_steps * \
                    self.train_batch_size % len(self.train_loader)

                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)
            logging.info(f'{datetime.now()} | Utilizing {self.n_gpu} GPUs')

        logging.info(f'{datetime.now()} | Moved model to: {self.device}')
        logging.info(
            f'{datetime.now()} | train_batch_size: {self.train_batch_size} | eval_batch_size: {self.test_batch_size}')
        logging.info(
            f'{datetime.now()} | Epochs: {epochs} | log_steps: {log_steps} | ckpt_steps: {ckpt_steps}')
        logging.info(
            f'{datetime.now()} | gradient_accumulation_steps: {gradient_accumulation_steps}')

        self.model.train()

        # tqdm(range(epochs), desc='Epochs', position=0):
        for epoch in range(start_epoch, epochs):
            logging.info(f'{datetime.now()} | Epoch: {epoch}')
            pb = tqdm(enumerate(self.train_loader),
                      desc=f'Epoch-{epoch} Iterator',
                      total=len(self.train_loader),
                      bar_format='{l_bar}{bar:10}{r_bar}'
                      )
            for step, batch in pb:
                if step < start_step:
                    continue
                inputs, inputs_mask, labels = batch
                inputs, inputs_mask, labels = inputs.to(
                    self.device), inputs_mask.to(self.device), labels.to(self.device)

                output = self.model(
                    inputs, attention_mask=inputs_mask, labels=labels)
                loss = output.loss

                if gradient_accumulation_steps > 1:
                    loss /= gradient_accumulation_steps

                loss.backward()
                # heelo

                step_loss += loss.item()
                losses[global_steps] = loss.item()
                local_steps += 1
                global_steps += 1

                if global_steps % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)

                    if self.tpu:
                        # Note: Cloud TPU-specific code!
                        xm.optimizer_step(self.optimizer, barrier=True)
                    else:
                        self.optimizer.step()

                    self.model.zero_grad()
                    self.scheduler.step()

                if global_steps % log_steps == 0:
                    pb.set_postfix_str(
                        f'''{datetime.now()} | Train Loss: {step_loss / local_steps} | Steps: {global_steps}''')
                    with open(f'{self.log_dir}/{self.model_name}_train_results.json', 'w') as results_file:
                        json.dump(losses, results_file)
                        results_file.close()
                    step_loss = 0.0
                    local_steps = 0
                if self.checkpoint_path:
                    if global_steps % ckpt_steps == 0:
                        self.save(epoch, self.model, self.optimizer, self.scheduler,
                                  losses, global_steps)
                        logging.info(
                            f'{datetime.now()} | Saved checkpoint to: {self.checkpoint_path}')

            # Evaluate every epoch
            self.evaluate(self.eval_loader)

            self.model.train()
            start_step = 0

        self.save(epochs, self.model, self.optimizer, self.scheduler, losses, global_steps)

        return self.model

    def evaluate(self, dataloader):
        if self.n_gpu > 1 and not isinstance(self.model, nn.DataParallel):
            self.model = nn.DataParallel(self.model)

        self.model.eval()

        self.metric()
        eval_loss = 0.0
        eval_steps = 0

        logging.info(f'{datetime.now()} | Evaluating...')
        for step, batch in tqdm(enumerate(dataloader),
                                desc='Evaluating',
                                leave=True,
                                total=len(dataloader),
                                bar_format='{l_bar}{bar:10}{r_bar}'):
            inputs, inputs_mask, labels = batch
            inputs, inputs_mask, labels = inputs.to(
                self.device), inputs_mask.to(self.device), labels.to(self.device)
            with torch.no_grad():
                output = self.model(
                    inputs, attention_mask=inputs_mask, labels=labels)

            if output.get("logits") is not None:
                preds = output.logits
                self.metric(preds, labels)
            elif output.get("start_logits") is not None:
                preds = [output.start_logits, output.end_logits]
                self.metric(preds, labels)

            tmp_eval_loss = output.loss

            if self.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
            eval_steps += 1

            total_eval_loss = eval_loss/eval_steps

            logging.info(
                f'{datetime.now()} | Step: {step} | Eval Loss: {total_eval_loss}')
            with open(f'{self.log_dir}/{self.model_name}_eval_results.txt', 'a+') as results_file:
                results_file.write(
                    f'{datetime.now()} | Step: {step} | Eval Loss: {total_eval_loss}\n')
                results_file.close()

        print(eval_loss / len(dataloader))
        self.print_metric()

    def metric(self, preds=None, labels=None):
        pass

    def print_metric(self,):
        pass

    def save(self, epoch, model, optimizer, scheduler, losses, train_step):
        if self.checkpoint_path:
            torch.save({
                'epoch': epoch,  # 현재 학습 epoch
                'model_state_dict': model.state_dict(),  # 모델 저장
                'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 저장
                'scheduler_state_dict': scheduler.state_dict(),  
                'losses': losses,  # Loss 저장
                'train_step': train_step,  # 현재 진행한 학습
            }, f'{self.checkpoint_path}/{self.model_name}.pth')
