import argparse
import warnings
import yaml
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything

from utils.model_handling import ModelIO, get_tag
from utils.train_helper_funcs import set_all_seeds, set_all_paths, get_files, get_files_ood
from model.ELECTRAFI import ELECTRAFI

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    """Parse CLI arguments allowing selective overrides of the YAML config."""
    parser = argparse.ArgumentParser(
        description="Run ELECTRA training with optional config overrides.")

    # Boolean flag for pruning
    parser.set_defaults(prune=None)

    # Float overrides for learning rates
    parser.add_argument("--initial_lr", type=float, default=None,
                        help="Override the initial learning rate.")
    parser.add_argument("--final_lr", type=float, default=None,
                        help="Override the final learning rate.")
    parser.add_argument("--lr_gamma", type=float, default=None,
                        help="Override the lr gamma")

    load_model_group = parser.add_mutually_exclusive_group()
    load_model_group.add_argument("--load_model", dest="load_model", action="store_true",
                              help="Enable model loading (overrides config).")
    load_model_group.add_argument("--no_load_model", dest="load_model", action="store_false",
                              help="Disable model loading (overrides config).")
    parser.set_defaults(load_model=None)
    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Return a new config with CLI overrides applied when provided."""
    if args.prune is not None:
        config["prune"] = args.prune

    if args.initial_lr is not None:
        config["initial_lr"] = args.initial_lr

    if args.final_lr is not None:
        config["final_lr"] = args.final_lr

    if args.lr_gamma is not None:
        config["lr_gamma"] = args.lr_gamma
    if args.load_model is not None:
        config["load_model"] = args.load_model

    return config


def load_base_config() -> dict:
    """Load the base YAML configuration depending on GPU availability and flags inside the YAML."""
    torch.set_float32_matmul_precision('high')
    base_path = "/ELECTRAFI"
    config = yaml.safe_load(open(f"{base_path}/hpc_conf.yaml"))
    if config["data_split"] == "ECD":
        config["master_units"] = 2160
        config["gaus_per_electrons"] = 120
        config["cutoff"] = 30
        config["dens_path"] = "/ECD_Full"
        config["project_name"] = "ELECTRAFI-ECD"
        config["lr_gamma"] = 0.98
        config['max_time'] = '06:20:00:00'
        config["ood_eval"] = False
    elif config["data_split"] == "cubic":
        config["master_units"] = 2200
        config["gaus_per_electrons"] = 100
        config["cutoff"] = 20
        config["dens_path"] = "/CUBIC_DATA"
        config["project_name"] = "ELECTRAFI-CUBIC"
        config["lr_gamma"] = 0.9
        config['max_time'] = '06:20:00:00'
        config["ood_eval"] = False
    elif config["data_split"] == "mp_mixed":
        config["master_units"] = 2160
        config["gaus_per_electrons"] = 120
        config["cutoff"] = 30
        config["dens_path"] = "/MP_MIXED_DATA"
        config["project_name"] = "ELECTRAFI-MP_Mixed"
        config["lr_gamma"] = 0.9
        config['max_time'] = '06:20:00:00'
        config["ood_eval"] = False
    elif config["data_split"] == "mpfull2025":
        config["master_units"] = 2160
        config["gaus_per_electrons"] = 120
        config["cutoff"] = 30
        config["dens_path"] = "/MP_FULL_2025"
        config["project_name"] = "ELECTRAFI-MP_Full"
        config["lr_gamma"] = 0.9
        config["save_memory"] = True
        config['max_time'] = '39:00:00:00'
        config["ood_eval"] = True
    elif config["data_split"] == "qm9":
        config["master_units"] = 2400
        config["gaus_per_electrons"] = 300
        config["cutoff"] = 20
        config["dens_path"] = "/QM9_Full"
        config["project_name"] = "ELECTRAFI-QM9"
        config["lr_gamma"] = 0.9
        config["save_memory"] = True
        config['max_time'] = '06:20:00:00'
        config["ood_eval"] = False
        config['pbc'] = False
        config['wrap'] = False
    return config


def run():
    args = parse_args()
    config = load_base_config()
    config = apply_overrides(config, args)
    if config['backbone'] == 'escaip':
        escaip_cfg_path = config['escaip_cfg_path']
        with open(escaip_cfg_path, "r") as f:
            escaip_cfg = yaml.safe_load(f)
        config['escaip_config'] = escaip_cfg
        if torch.cuda.is_available():
            if config["data_split"] == "ECD":
                config['escaip_config']['model']["backbone"]["max_num_nodes_per_batch"] = 20
                config['escaip_config']['model']["backbone"]["use_compile"] = True
                config['escaip_config']['model']["backbone"]['max_neighbors'] = 200
            elif config["data_split"] == "cubic":
                config['escaip_config']['model']["backbone"]["max_num_nodes_per_batch"] = 64
                config['escaip_config']['model']["backbone"]["use_compile"] = True
                config['escaip_config']['model']["backbone"]['max_neighbors'] = 128
            elif config["data_split"] == "mp_mixed":
                config['escaip_config']['model']["backbone"]["max_num_nodes_per_batch"] = 154
                config['escaip_config']['model']["backbone"]["use_compile"] = True
                config['escaip_config']['model']["backbone"]['max_neighbors'] = 200
            elif config["data_split"] == "mpfull2025":
                config['escaip_config']['model']["backbone"]["max_num_nodes_per_batch"] = 154
                config['escaip_config']['model']["backbone"]["use_compile"] = True
                config['escaip_config']['model']["backbone"]['max_neighbors'] = 200
            elif config["data_split"] == "qm9":
                config['escaip_config']['model']["backbone"]["max_num_nodes_per_batch"] = 35
                config['escaip_config']['model']["backbone"]["use_compile"] = True
                config['escaip_config']['model']["backbone"]['max_neighbors'] = 35

    # Seed alignment
    set_all_seeds(config['seed'])

    # Lightning Trainer defaults: auto strategy is fine for single-GPU
    accel = 'auto' if torch.cuda.is_available() else 'cpu'
    devices = 1

    trainer = L.Trainer(
        accelerator=accel,
        devices=devices,
        logger=WandbLogger(
            config=config,
            project=config["project_name"],
            log_model=True,
            group=f"Split_{config['data_split']}"
        ) if config.get('wandb', False) else None,
        check_val_every_n_epoch=config['eval_every'],
        log_every_n_steps=1,
        max_epochs=config['max_epochs'],
        gradient_clip_val=(config['gradient_clip_value'] if config.get('clip_grad', False) else None),
        gradient_clip_algorithm='value',
        max_time=config['max_time'],
    )

    # WandB naming and paths
    if config.get('wandb', False):
        wb_name = trainer.logger.experiment.name
        tag = get_tag(wb_name)
    else:
        wb_name = None
        tag = get_tag("test")

    config = set_all_paths(config, wb_name)
    with open(".wandbignore", "w") as f:
        f.write(f"{config['model_dir']}/\n*.pth\n")

    train_files, test_files, validation_files = get_files(config)
    if config["ood_eval"]:
        ood_file_dict = get_files_ood(config)
    model_handler = ModelIO(directory=config['model_dir'], tag=tag) if config.get('save_model', False) else None

    electrafi = ELECTRAFI(
        train_files=train_files,
        test_files=test_files,
        validation_files=validation_files,
        model_handler=model_handler,
        config=config,
    )

    if config.get("load_model", False):
        print(f"Loading model from {config['load_model_path']}")
        electrafi.load_state_dict(torch.load(config['load_model_path']))

    # DataLoaders and seed workers
    train_loader = electrafi.train_dataloader()
    val_loader = electrafi.val_dataloader()
    test_loader = electrafi.test_dataloader()
    seed_everything(config['seed'], workers=True)

    # Run training and testing
    trainer.fit(model=electrafi, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model=electrafi, dataloaders=test_loader, ckpt_path=None)
    if config["ood_eval"]:
        for name in ood_file_dict.keys():
            ood_loader = electrafi.ood_dataloader(files=ood_file_dict[name]['files'], path=ood_file_dict[name]['path'], name=name)
            trainer.test(model=electrafi, dataloaders=ood_loader, ckpt_path=None)


if __name__ == "__main__":
    run()


