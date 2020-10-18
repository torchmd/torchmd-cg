import os
import numpy as np
import time

import torch
from torch.nn import MSELoss, L1Loss
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import schnetpack as spk
import schnetpack.atomistic as atm
import schnetpack.representation as rep
from schnetpack.nn.cutoff import CosineCutoff
from schnetpack.data.loader import _collate_aseatoms
from schnetpack.environment import SimpleEnvironmentProvider

from torchmd_cg.nnp.schnet_dataset import SchNetDataset
from torchmd_cg.nnp.utils import LoadFromFile, LogWriter
from torchmd_cg.nnp.utils import save_argparse
from torchmd_cg.nnp.utils import train_val_test_split, set_batch_size
from torchmd_cg.nnp.npdataset import NpysDataset, NpysDataset2
from torchmd_cg.nnp.model import make_schnet_model

import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--conf','-c', type=open, action=LoadFromFile)#keep first
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch-size',  default=32,type=int, help='batch size')
    parser.add_argument('--num-epochs', default=300,type=int, help='number of epochs')
    parser.add_argument('--order', default=None, help='Npy file with order on which to split idx_train,idx_val,idx_test')
    parser.add_argument('--coords', default='coords.npy', help='Data source')
    parser.add_argument('--forces', default='forces.npy', help='Data source')
    parser.add_argument('--embeddings', default='embeddings.npy', help='Data source')
    parser.add_argument('--weights', default=None, help='Data source')
    parser.add_argument('--splits', default=None, help='Npz with splits idx_train,idx_val,idx_test')
    parser.add_argument('--gpus', default=0, help='Number of GPUs. Use CUDA_VISIBLE_DEVICES=1,2 to decide gpu')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--log-dir', '-l', default='/tmp/cgnet', help='log file')
    parser.add_argument('--label', default='forces', help='Label')
    parser.add_argument('--derivative', default='forces', help='Label')
    parser.add_argument('--eval-interval',type=int,default=2,help='eval interval, one eval per n updates (default: 2)')
    parser.add_argument('--test-interval',type=int,default=10,help='eval interval, one eval per n updates (default: 10)')
    parser.add_argument('--save-interval',type=int,default=10,help='save interval, one save per n updates (default: 10)')
    parser.add_argument('--seed',type=int,default=1,help='random seed (default: 1)')
    parser.add_argument('--load-model',default=None,help='Restart training using a model checkpoint')  
    parser.add_argument('--progress',action='store_true', default=False,help='Progress bar during batching')  
    parser.add_argument('--val-ratio',type=float, default=0.05,help='Percentual of validation set')  
    parser.add_argument('--test-ratio',type=float, default=0.1,help='Percentual of test set')  
    parser.add_argument('--num-workers',type=int, default=0,help='Number of workers for data prefetch') 
    parser.add_argument('--num-filters',type=int, default=128,help='Number of filter in model') 
    parser.add_argument('--num-gaussians',type=int, default=50,help='Number of Gaussians in model') 
    parser.add_argument('--num-interactions',type=int, default=2,help='Number of interactions in model') 
    parser.add_argument('--max-z',type=int, default=100,help='Max atomic number in model') 
    parser.add_argument('--cutoff',type=float, default=9,help='Cutoff in model') 
    parser.add_argument('--lr-patience',type=int,default=10,help='Patience for lr-schedule. Patience per eval-interval of validation')
    parser.add_argument('--lr-min',type=float, default=1e-6,help='Minimum learning rate before early stop') 
    parser.add_argument('--lr-factor',type=float, default=0.8,help='Minimum learning rate before early stop') 
    parser.add_argument('--distributed-backend', default='ddp', help='Distributed backend: dp, ddp, ddp2')
    # fmt: on
    args = parser.parse_args()

    if args.test_ratio == 0:
        args.test_interval = 0

    if args.val_ratio == 0:
        args.eval_interval = 0

    save_argparse(args, os.path.join(args.log_dir, "input.yaml"), exclude=["conf"])

    return args


def make_splits(
    dataset_len, val_ratio, test_ratio, seed, filename=None, splits=None, order=None
):
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
        idx_test = splits["idx_test"]
    else:
        idx_train, idx_val, idx_test = train_val_test_split(
            dataset_len, val_ratio, test_ratio, seed, order
        )

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    return idx_train, idx_val, idx_test


class LNNP(pl.LightningModule):
    def __init__(self, hparams):
        super(LNNP, self).__init__()
        self.hparams = hparams
        if self.hparams.load_model:
            raise NotImplementedError  # TODO
        else:
            self.model = make_schnet_model(self.hparams)
        # save linear fit model with random parameters
        self.loss_fn = MSELoss()
        self.test_fn = L1Loss()

    def prepare_data(self):
        print("Preparing data...", flush=True)
        self.dataset = NpysDataset2(
            self.hparams.coords, self.hparams.forces, self.hparams.embeddings
        )
        self.dataset = SchNetDataset(
            self.dataset,
            environment_provider=SimpleEnvironmentProvider(),
            label=["forces"],
        )
        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(self.dataset),
            self.hparams.val_ratio,
            self.hparams.test_ratio,
            self.hparams.seed,
            os.path.join(self.hparams.log_dir, f"splits.npz"),
            self.hparams.splits,
        )
        self.train_dataset = torch.utils.data.Subset(self.dataset, self.idx_train)
        self.val_dataset = torch.utils.data.Subset(self.dataset, self.idx_val)
        self.test_dataset = torch.utils.data.Subset(self.dataset, self.idx_test)
        print(
            "train {}, val {}, test {}".format(
                len(self.train_dataset), len(self.val_dataset), len(self.test_dataset)
            )
        )

        if self.hparams.weights is not None:
            self.weights = torch.from_numpy(np.load(self.hparams.weights))
        else:
            self.weights = torch.ones(len(self.dataset))

    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            sampler=WeightedRandomSampler(
                self.weights[self.idx_train], len(self.train_dataset)
            ),
            batch_size=set_batch_size(self.hparams.batch_size, len(self.train_dataset)),
            shuffle=False,
            collate_fn=_collate_aseatoms,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        return train_loader

    def training_step(self, batch, batch_idx):
        prediction = self(batch)
        loss = self.loss_fn(prediction[self.hparams.label], batch[self.hparams.label])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def val_dataloader(self):
        val_loader = None
        if len(self.val_dataset) > 0:
            # val_loader = DataLoader(self.val_dataset, sampler=WeightedRandomSampler(self.weights[self.idx_val], len(self.val_dataset)),
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=set_batch_size(
                    self.hparams.batch_size, len(self.val_dataset)
                ),
                collate_fn=_collate_aseatoms,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
            )
        return val_loader

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        prediction = self(batch)
        torch.set_grad_enabled(False)
        loss = self.loss_fn(prediction[self.hparams.label], batch[self.hparams.label])
        self.log('val_loss', loss)

    def test_dataloader(self):
        test_loader = None
        if len(self.test_dataset) > 0:
            # test_loader = DataLoader(self.test_dataset, sampler=WeightedRandomSampler(self.weights[self.idx_test], len(self.test_dataset)),
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=set_batch_size(
                    self.hparams.batch_size, len(self.test_dataset)
                ),
                collate_fn=_collate_aseatoms,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
            )
        return test_loader

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        prediction = self(batch)
        torch.set_grad_enabled(False)
        loss = self.test_fn(prediction[self.hparams.label], batch[self.hparams.label])
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr, momentum=0.9)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            min_lr=self.hparams.lr_min
        )
        lr_scheduler = {'scheduler':scheduler,
                        'monitor':'val_loss',
                        'interval': 'epoch',
                        'frequency': 1,
                        } 
        return [optimizer], [lr_scheduler]


def main():
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = LNNP(args)
    checkpoint_callback = ModelCheckpoint(
        filepath=args.log_dir,
        monitor="val_loss",
        save_top_k=-1,
        period=args.eval_interval,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    tb_logger = pl.loggers.TensorBoardLogger(args.log_dir)
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.num_epochs,
        distributed_backend=args.distributed_backend,
        num_nodes=args.num_nodes,
        default_root_dir=args.log_dir,
        auto_lr_find=False,
        resume_from_checkpoint=args.load_model,
        checkpoint_callback=checkpoint_callback,
        callbacks=[lr_monitor],
        logger=tb_logger,
        reload_dataloaders_every_epoch=False
    )

    trainer.fit(model)

    # run test set after completing the fit
    trainer.test()

    # logs = LogWriter(args.log_dir,keys=('epoch','train_loss','val_loss','test_mae','lr'))


#        logs.write_row({'epoch':epoch,'train_loss':train_loss,'val_loss':val_loss,
#                        'test_mae':test_mae, 'lr':optimizer.param_groups[0]['lr']})
#        progress.set_postfix({'Loss': train_loss, 'lr':optimizer.param_groups[0]['lr']})

#        if optimizer.param_groups[0]['lr'] < args.lr_min:
#            print("Early stop reached")
#            break


if __name__ == "__main__":
    main()
