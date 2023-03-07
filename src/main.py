import os
import numpy as np
import warnings
import torch
import pytorch_lightning as pl
from pl_bolts.datamodules import MNISTDataModule, FashionMNISTDataModule, CIFAR10DataModule, ImagenetDataModule
from datamodules import SmallNORBDataModule, CIFAR100DataModule
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint  # Callback, 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models import Agglomerator
from utils import TwoCropTransform, count_parameters
from custom_transforms import CustomTransforms

import flags_Agglomerator
from absl import app
from absl import flags
FLAGS = flags.FLAGS

LOG_DIR = './lightning_logs/'


def seed_everything():
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    pl.seed_everything(FLAGS.seed)
    torch.cuda.empty_cache()

def get_DataModule(FLAGS):
    DataModuleWrapper = {
        "MNIST": MNISTDataModule,
        "FashionMNIST": FashionMNISTDataModule,
        "smallNORB": SmallNORBDataModule,
        "CIFAR10": CIFAR10DataModule,
        "CIFAR100": CIFAR100DataModule,
        "IMAGENET": ImagenetDataModule
    }

    if FLAGS.dataset not in DataModuleWrapper.keys():
        print("❌ Dataset not compatible")
        quit(0)

    dm = DataModuleWrapper[FLAGS.dataset](  "./datasets", 
                                            batch_size=FLAGS.batch_size, 
                                            shuffle=True, 
                                            pin_memory=True, 
                                            drop_last=True
                                        )

    ct = CustomTransforms(FLAGS)

    # Apply trainsforms
    if(FLAGS.supervise):
        dm.train_transforms = ct.train_transforms[FLAGS.dataset]
        dm.val_transforms = ct.test_transforms[FLAGS.dataset]
        dm.test_transforms = ct.test_transforms[FLAGS.dataset]
    else:
        dm.train_transforms = TwoCropTransform(ct.train_transforms[FLAGS.dataset])
        dm.val_transforms = TwoCropTransform(ct.test_transforms[FLAGS.dataset])
        dm.test_transforms = TwoCropTransform(ct.test_transforms[FLAGS.dataset])

    return dm


def get_ckpt_dir(ckpt_dir):
    return os.path.join(os.getcwd(), ckpt_dir) if ckpt_dir is not None else None


def get_model(FLAGS):

    model = Agglomerator(FLAGS)

    ckpt_dir = get_ckpt_dir(FLAGS.ckpt_dir)
    if ckpt_dir is not None:
        model = model.load_from_checkpoint(ckpt_dir, FLAGS=FLAGS, strict=False)

    print("Total trainable parameters: ", count_parameters(model))
    return model 

def get_trainer(FLAGS):
    ckpt_dir = get_ckpt_dir(FLAGS.ckpt_dir)

    if FLAGS.logger == 'tensorboard':
        logger = TensorBoardLogger(LOG_DIR, name = FLAGS.exp_name, default_hp_metric=False, version= getattr(FLAGS, 'logger_version', None))
    else:
        logger = WandbLogger(project="Agglomerator", name=FLAGS.exp_name);    logger.experiment.config.update(FLAGS)

    trainer = pl.Trainer(
        gpus=FLAGS.gpus, #-1, 
        strategy='dp',
        resume_from_checkpoint=ckpt_dir, 
        max_epochs=FLAGS.max_epochs, 
        limit_train_batches=FLAGS.limit_train, 
        limit_val_batches=FLAGS.limit_val, 
        limit_test_batches=FLAGS.limit_test, 
        callbacks = [CustomProgressBar(refresh_rate=1, enable_val=False), #FLAGS.enable_val_progress), 
                     LearningRateMonitor(logging_interval='step'),
                        # WeightDecayMonitor(logging_interval='step'),  
                    #   earlystop_callback,
                      ],
        logger=logger,                           # optional for 'test' and 'freeze' mode
        reload_dataloaders_every_n_epochs = 1    # optional for 'test' and 'freeze' mode
    )
    return trainer 



##################################

import sys
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm

class CustomProgressBar(TQDMProgressBar):
    def __init__(self, enable_val=False, *args, **kwargs):
        self.enable_validation_bar=enable_val
        super().__init__(*args, **kwargs)
        
    def init_validation_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        bar = Tqdm(
            desc=self.validation_description,
            position=(2 * self.process_position + has_main_bar),
            disable=not self.enable_validation_bar, #self.is_disabled,
            leave=not has_main_bar,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar


def main(argv):
    # print(argv, FLAGS)
    print(FLAGS.flag_values_dict())
    # import pdb; pdb.set_trace()

    seed_everything()

    model = get_model(FLAGS)
    dm = get_DataModule(FLAGS)
    trainer = get_trainer(FLAGS)

    if FLAGS.mode == "train":
        # model = model.load_from_checkpoint(ckpt_dir, FLAGS=FLAGS, strict=False) if FLAGS.resume_training else model
        trainer.fit(model, dm)

    elif FLAGS.mode == "test":
        # model = model.load_from_checkpoint(ckpt_dir, FLAGS=FLAGS, strict=False)
        model.configure_optimizers()
        
        dm.prepare_data()
        dm.setup()

        trainer.test(model, test_dataloaders=dm.test_dataloader())

    elif FLAGS.mode == "freeze":
        
        datasplits = [dm.train_dataloader, dm.val_dataloader, dm.test_dataloader]
        modes = ["Training", "Validation", "Test"]
        features_names = ['/features_train', '/features_val', '/features_test']
        labels_names = ['/labels_train', '/labels_val', '/labels_test']

        for i, (d, m, f, l) in enumerate(zip(datasplits, modes, features_names, labels_names)):
            # model = model.load_from_checkpoint(ckpt_dir, FLAGS=FLAGS, strict=False)
            model.configure_optimizers()

            dm.prepare_data()
            dm.setup()

            trainer.test(model, test_dataloaders=d())
            
            print(m + " features shape: ", np.array(model.features).shape)
            np.save('output/' + FLAGS.dataset + f, np.array(model.features))
            np.save('output/' + FLAGS.dataset + l, np.array(model.labels))

        model.batch_acc = 0

if __name__ == '__main__':
    app.run(main)