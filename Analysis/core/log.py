# Training logs that stay on the terminal and in a file.
from torch.utils.tensorboard import SummaryWriter
import logging
import os
import sys
import shutil
import datetime


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        first = self.streams[0]
        return first.isatty() if hasattr(first, "isatty") else False


def config_logger(cfg, OUT_PATH="results/", time=True):
    """File + terminal log for the legacy trainer.

    Prints stay visible in the terminal and are copied to
    ``logs/<dataset>/<run>.txt``. The previous version replaced stdout, so a
    job looked hung unless you opened that file.
    """
    data_name = cfg.dataset
    os.makedirs(OUT_PATH, exist_ok=True)
    if cfg.logfile is None:
        if cfg.metis.n_patches > 0:
            model_name = cfg.model.gMHA_type
        else:
            model_name = 'MPGNN'
        config_string = f'{model_name}_{cfg.model.gnn_type}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        if cfg.depth > 0:
            config_string = config_string + '_depth' + str(cfg.depth)
    else:
        config_string = cfg.logfile

    writer_folder = os.path.join(OUT_PATH, data_name, config_string)
    if time:
        writer_folder = os.path.join(
            writer_folder, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if os.path.isdir(writer_folder):
        shutil.rmtree(writer_folder)
    writer = SummaryWriter(writer_folder)

    logger = logging.getLogger("scijepa")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    logger_file = os.path.join(OUT_PATH, data_name, 'summary.log')
    os.makedirs(os.path.dirname(logger_file), exist_ok=True)
    file_handler = logging.FileHandler(logger_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    stream_handler = logging.StreamHandler(sys.__stdout__)
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    os.makedirs(os.path.join('logs', data_name), exist_ok=True)
    log_path = f'logs/{data_name}/{config_string}.txt'
    if not isinstance(sys.stdout, _Tee):
        sys.stdout = _Tee(sys.stdout, open(log_path, 'w', buffering=1))

    print("-"*50)
    print(cfg)
    print("-"*50)
    print('Time:', datetime.datetime.now().strftime("%Y/%m/%d - %H:%M"))
    return writer, logger
