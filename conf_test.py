import torch
import hydra
from loguru import logger
from omegaconf import OmegaConf

@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(cfg.pretty())

    # Get device
    device = torch.device("cuda" if (torch.cuda.is_available() and not cfg.training.no_cuda) else "cpu")
    logger.info("DEVICE : {}".format(device))


if __name__ == "__main__":
    main()