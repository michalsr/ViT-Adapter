import argparse
import copy
import os
import os.path as osp
import time
import warnings
from tqdm import tqdm 
from pycocotools import mask as mask_utils
import mmcv
import itertools 
import torch.nn.functional as F
import math 
import torchvision.transforms as T
import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash
from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger
import torch._tensor
from torch._tensor import Tensor
import logging
import pickle 
import json 
import numpy as np 
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class CenterPadding(torch.nn.Module):
    def __init__(self, multiple = 14):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output
    
def open_file(filename):
    """
    Based on https://github.com/salesforce/LAVIS/blob/main/lavis/common/utils.py
    Supported:
        .pkl, .pickle, .npy, .json
    """
    file_ext = os.path.splitext(filename)[1]
    if file_ext == '.txt':
        with open(filename,'r+') as fopen:
            data = fopen.readlines()
    elif file_ext in [".npy",".npz"]:
        data = np.load(filename,allow_pickle=True)
    elif file_ext == '.json':
        with open(filename,'r+') as fopen:
            data = json.load(fopen)
    else:
        # assume pickle
        with open(filename,"rb+") as fopen:
            data = pickle.load(fopen)
    return data

def save_file(filename,data,json_numpy=False):
    """
    Based on https://github.com/salesforce/LAVIS/blob/main/lavis/common/utils.py
    Supported:
        .pkl, .pickle, .npy, .json
    """
    parent_dir = os.path.dirname(filename)
    if parent_dir != '':
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
    # Path(parent_dir).chmod(0o0777)
    file_ext = os.path.splitext(filename)[1]
    if file_ext == ".npy":
        with open(filename, "wb+") as fopen:
            np.save(fopen, data)
    elif file_ext == ".json":

            with open(filename,'w+') as fopen:
                json.dump(data,fopen,indent=2)

    else:
        # assume file is pickle
         with open(filename, "wb+") as fopen:
            pickle.dump(data, fopen)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        "--layers",
        type=str,
        default="[23]",
        help="List of layers or number of last layers to take"
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--no_load_dino',action='store_true')
    parser.add_argument(
        "--train_dino_dir",
        type=str,
        default='/tmp/dinov2_train',
        help="Location of feature files",
    )
    parser.add_argument(
        "--val_dino_dir",
        type=str,
        default='/tmp/dinov2_val',
        help="Location of feature files",
    )
    parser.add_argument(
        "--sam_train_dir",
        type=str,
        default='/scratch/bcgp/michal5/sam_slic_ade20k/sam_regions/train',
        help="Location of feature files",
    )
    parser.add_argument(
        "--sam_val_dir",
        type=str,
        default='/scratch/bcgp/michal5/sam_slic_ade20k/sam_regions/val',
        help="Location of feature files",
    )
    parser.add_argument(
        "--train_region_features",
        type=str,
        default='/scratch/bcgp/michal5/dinov2_512_region_features/train',
        help="Location of feature files",
    )
    parser.add_argument(
        "--val_region_features",
        type=str,
        default='/scratch/bcgp/michal5/dinov2_512_region_features/val',
        help="Location of feature files",
    )
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args
def load_all_regions(directory):
    logger.info(f"Loading region masks from {directory}")
    image_id_to_mask = {}
    for f in tqdm(os.listdir(directory)):
        filename_extension = os.path.splitext(f)[1]
        regions = open_file(os.path.join(directory,f))
        
        image_id_to_mask[f.replace(filename_extension,'')] = regions
    return image_id_to_mask
def extract(model,args,image):
    # transform = T.Compose([
    #     T.ToTensor(),
    #     lambda x: x.unsqueeze(0),
    #     CenterPadding(multiple = 14),
    #     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    transform = T.Compose([
        # T.ToTensor(),
        # lambda x: x.unsqueeze(0),
        CenterPadding(multiple = 14)])
        #T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    with torch.inference_mode():
        layers = [23]
        # intermediate layers does not use a norm or go through the very last layer of output
        #img = transform(image).to(device='cuda',dtype=torch.bfloat16)
        image = image.unsqueeze(0)

        img = transform(image).to(device='cuda',dtype=torch.bfloat16)
        
        features_out = model.get_intermediate_layers(img, n=layers,reshape=True)    
        features = torch.cat(features_out, dim=1) # B, C, H, W 
    return features.detach().cpu().to(torch.float32).numpy()
def region_features(feature_dir,region_feature_dir,image_id_to_sam):

    # Get the intersection of the feature files and the sam regions
    all_feature_files = [f for f in os.listdir(feature_dir) if os.path.isfile(os.path.join(feature_dir, f))]
    feature_files_in_sam = [f for f in all_feature_files if os.path.splitext(f)[0] in image_id_to_sam]

    features_minus_sam = set(all_feature_files) - set(feature_files_in_sam)
    if len(features_minus_sam) > 0:
        logger.warning(f'Found {len(features_minus_sam)} feature files that are not in the set of SAM region files: {features_minus_sam}')

    prog_bar = tqdm(feature_files_in_sam)
    def extract_features(f,region_feature_dir,feature_dir,device='cuda',features_exist=True):
        prog_bar.set_description(f'Region features: {f}')

        features = open_file(os.path.join(feature_dir,f))

        if len(features.shape)>4:
            features = np.squeeze(features,axis=0)
        file_name = f
        ext = os.path.splitext(f)[1]
        all_region_features_in_image = []
        sam_regions = image_id_to_sam[file_name.replace(ext,'')]

        if len(sam_regions) > 0:
            # sam regions within an image all have the same total size
            new_h, new_w = mask_utils.decode(sam_regions[0]['segmentation']).shape
            patch_length = 14
            padded_h, padded_w = math.ceil(new_h / patch_length) * patch_length, math.ceil(new_w / patch_length) * patch_length # Get the padded height and width
            upsample_feature = torch.nn.functional.interpolate(torch.from_numpy(features).cuda(), size=[padded_h,padded_w],mode='bilinear') # First interpolate to the padded size
            upsample_feature = T.CenterCrop((new_h, new_w)) (upsample_feature).squeeze(dim = 0) # Apply center cropping to the original size
            f,h,w = upsample_feature.size()

            for region in sam_regions:
                start_region_time = time.time()
                sam_region_feature = {}
                if 'region_id' in region:

                    sam_region_feature['region_id'] = region['region_id']
                sam_mask = mask_utils.decode(region['segmentation'])
                if 'area' in region:
                    sam_region_feature['area'] = region['area']
                r_1, r_2 = np.where(sam_mask == 1)
                features_in_sam = upsample_feature[:,r_1,r_2].view(f,-1).mean(1).cpu().numpy()
                
                sam_region_feature['region_feature'] = features_in_sam
                all_region_features_in_image.append(sam_region_feature)
                
    
        save_file(os.path.join(region_feature_dir, file_name.replace(ext,'.pkl')), all_region_features_in_image)
    for i,f in enumerate(prog_bar):
        try:
            extract_features(f,region_feature_dir,feature_dir)

        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f'Caught CUDA out of memory error for {f}; falling back to CPU')
            torch.cuda.empty_cache()
            continue 
def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    cfg.auto_resume = args.auto_resume

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # gpu_ids is used to calculate iter when resuming checkpoint
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # cfg.device = 'cuda'  # fix 'ConfigDict' object has no attribute 'device'
    # # create work_dir
    # #cfg.work_dir = '/scratch/bcgp/michal5/ViT-Adapter/output/full_train'
    # cfg.model.pretrained = None

    # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # # dump config
    # cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # # init the logger before other steps
    # timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # # init the meta dict to record some important information such as
    # # environment info and seed, which will be logged
    # meta = dict()
    # # log env info
    # env_info_dict = collect_env()
    # env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    # dash_line = '-' * 60 + '\n'
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' +
    #             dash_line)
    # meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')


    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed

    try:
        torch._tensor._rebuild_from_type_v2
    except AttributeError:
        def _rebuild_from_type_v2(func, new_type, args, state):
            ret = func(*args)
            if type(ret) is not new_type:
                ret = ret.as_subclass(new_type)
            # Tensor does define __setstate__ even though it doesn't define
            # __getstate__. So only use __setstate__ if it is NOT the one defined
            # on Tensor
            if (
                getattr(ret.__class__, "__setstate__", Tensor.__setstate__)
                is not Tensor.__setstate__
            ):
                ret.__setstate__(state)
            else:
                ret = torch._utils._set_obj_state(ret, state)
            return ret
        torch._tensor. _rebuild_from_type_v2 = _rebuild_from_type_v2
    def _set_obj_state(obj, state):
        if isinstance(state, tuple):
            if not len(state) == 2:
                raise RuntimeError(f"Invalid serialized state: {state}")
            dict_state = state[0]
            slots_state = state[1]
        else:
            dict_state = state
            slots_state = None

        # Starting with Python 3.11, the __dict__ attribute is lazily created
        # and is serialized as None when not needed.
        if dict_state:
            for k, v in dict_state.items():
                setattr(obj, k, v)

        if slots_state:
            for k, v in slots_state.items():
                setattr(obj, k, v)
        return obj

    torch._utils._set_obj_state = _set_obj_state
    torch.hub.set_dir('/scratch/bcgp/michal5')
    model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vitl14")
    model.eval()
    model = model.to(device='cuda', dtype=torch.bfloat16)  
    train_dataset = build_dataset(cfg.data.train)
    for entry in tqdm(train_dataset):
        image_filename = entry['img_metas'].data['ori_filename']
        image = entry['img'].data 
        features = extract(model,args,image)
        save_file(os.path.join(args.train_dino_dir,image_filename.replace('.jpg','.pkl')),features)
    train_image_id_to_mask = load_all_regions(args.sam_train_dir)
    region_features(args.train_dino_dir,args.train_region_features,train_image_id_to_mask)
    val_dataset = copy.deepcopy(cfg.data.val)
    val_dataset.pipeline = cfg.data.train.pipeline
    val_dataset = build_dataset(val_dataset)
    for entry in tqdm(val_dataset):
        image_filename = entry['img_metas'].data['ori_filename']
        image = entry['img'].data 
        features = extract(model,args,image)
        save_file(os.path.join(args.val_dino_dir,image_filename.replace('.jpg','.pkl')),features)
    val_image_id_to_mask = load_all_regions(args.sam_val_dir)
    region_features(args.val_dino_dir,args.val_region_features,val_image_id_to_mask)








if __name__ == '__main__':
    main()
