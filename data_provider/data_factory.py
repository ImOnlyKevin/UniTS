from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, GLUONTSDataset, CustomNpySegLoader
from data_provider.uea import collate_fn
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    # 'm4': Dataset_M4,  Removed due to the LICENSE file constraints of m4.py
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'ESA-Mission1': CustomNpySegLoader,
    'ESA-Mission2': CustomNpySegLoader,
    'STPSat4-TCS': CustomNpySegLoader,
    'STPSat4-HRR': CustomNpySegLoader,
    'STPSat4-MRR': CustomNpySegLoader,
    'STPSat4-PCE1': CustomNpySegLoader,
    'STPSat4-PCE2': CustomNpySegLoader,
    'STPSat4-ADCS': CustomNpySegLoader,
    'STPSat7-EPS':  CustomNpySegLoader,
    'STPSat7-ADCS': CustomNpySegLoader,
    'STPSat7-TO':   CustomNpySegLoader,
    'STPSat7-HRR':  CustomNpySegLoader,
    'STPSat7-MRR':  CustomNpySegLoader,
    'STPSat7-TC':   CustomNpySegLoader,
    'UEA': UEAloader,
    # datasets from gluonts package:
    "gluonts": GLUONTSDataset,
}


def random_subset(dataset, pct, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=generator)
    return Subset(dataset, idx[:int(len(dataset) * pct)].long().numpy())


def build_dataloader_kwargs(args):
    num_workers = max(0, int(getattr(args, "num_workers", 0)))
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def data_provider(args, config, flag, ddp=False):  # args,
    Data = data_dict[config['data']]
    timeenc = 0 if config['embed'] != 'timeF' else 1
    loader_kwargs = build_dataloader_kwargs(args)

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        if 'anomaly_detection' in config['task_name']:  # working on one gpu
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if 'gluonts' in config['data']:
        # process gluonts dataset:
        data_set = Data(
            dataset_name=config['dataset_name'],
            size=(config['seq_len'], config['label_len'], config['pred_len']),
            path=config['root_path'],
            # Don't set dataset_writer
            features=config["features"],
            flag=flag,
        )
        if args.subsample_pct is not None and flag == "train":
            data_set = random_subset(
                data_set, args.subsample_pct, args.fix_seed)

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            **loader_kwargs,
            drop_last=drop_last
        )

        return data_set, data_loader

    timeenc = 0 if config['embed'] != 'timeF' else 1

    if 'anomaly_detection' in config['task_name']:
        drop_last = False
        data_set = Data(
            root_path=config['root_path'],
            win_size=config['seq_len'],
            flag=flag,
        )
        if args.subsample_pct is not None and flag == "train":
            data_set = random_subset(
                data_set, args.subsample_pct, args.fix_seed)
        print("ddp mode is set to false for anomaly_detection", ddp, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False if ddp else shuffle_flag,
            sampler=DistributedSampler(data_set) if ddp else None,
            **loader_kwargs,
            drop_last=drop_last)
        return data_set, data_loader
    elif 'classification' in config['task_name']:
        drop_last = False
        data_set = Data(
            root_path=config['root_path'],
            flag=flag,
        )
        if args.subsample_pct is not None and flag == "train":
            data_set = random_subset(
                data_set, args.subsample_pct, args.fix_seed)
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False if ddp else shuffle_flag,
            drop_last=drop_last,
            sampler=DistributedSampler(data_set) if ddp else None,
            **loader_kwargs,
            collate_fn=lambda x: collate_fn(x, max_len=config['seq_len'])
        )
        return data_set, data_loader
    else:
        if config['data'] == 'm4':
            drop_last = False
        data_set = Data(
            root_path=config['root_path'],
            data_path=config['data_path'],
            flag=flag,
            size=[config['seq_len'], config['label_len'], config['pred_len']],
            features=config['features'],
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=config['seasonal_patterns'] if config['data'] == 'm4' else None
        )
        if args.subsample_pct is not None and flag == "train":
            data_set = random_subset(
                data_set, args.subsample_pct, args.fix_seed)
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False if ddp else shuffle_flag,
            sampler=DistributedSampler(data_set) if ddp else None,
            **loader_kwargs,
            drop_last=drop_last)
        return data_set, data_loader
