from data import build_dataloader
from processor import do_train_w_teachers
import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO" 
from model.MHSA import Mlp
from train import set_seed, set_up_params, set_up_dist, setup_logging, setup_model, modify_params, add_additional_attributes, vid_set
from processor.train_fn import * 
from processor import * 
from loss.custom_loss import * 
import torch

def add_external_training_fns(base_kwargs, kwargs_external, kwargs_internal):
    kwargs = dict(base_kwargs)  

    if kwargs_external is not None:
        kwargs.update(kwargs_external)

    tm_ext = kwargs_external.get("training_mode") if kwargs_external else None
    tm_int = kwargs_internal.get("training_mode") if kwargs_internal else None
    if tm_ext is not None:
        kwargs["teacher_training_mode"] = tm_ext
    if tm_int is not None:
        kwargs["training_mode"] = tm_int

    if 'TRAIN_step_FN' in kwargs:
        del kwargs['TRAIN_step_FN']  

    if kwargs_external and 'TRAIN_step_FN' in kwargs_external:
        kwargs['TRAIN_ext_step_FN'] = kwargs_external['TRAIN_step_FN']
    if kwargs_internal and 'TRAIN_step_FN' in kwargs_internal:
        kwargs['TRAIN_step_FN'] = kwargs_internal['TRAIN_step_FN']

    return kwargs


if __name__ == '__main__':

    args, cfg = set_up_params()
    cfg = modify_params(cfg, args)
    local_rank, dist_local_rank, output_dir = set_up_dist(cfg, args)
    logger = setup_logging(cfg, args, output_dir, dist_local_rank)

    ################  TEACHER MODE  ################
    kwargs = {}

    student_dataset = cfg.DATA.DATASET
    student_dataset_root = cfg.DATA.ROOT
    student_model = cfg.MODEL.NAME
    student_resume = args.resume
    student_test_weight = cfg.TEST.WEIGHT
    student_dataset_fix = cfg.DATA.DATASET_FIX
    batch_size = cfg.DATA.BATCH_SIZE
    logger.info("[MAIN] after set_up_params/top")

    cfg.defrost()
    cfg.DATA.DATASET = cfg.TRAIN.TEACH1
    cfg.DATA.ROOT = cfg.TRAIN.DIR_TEACH1    
    logger.info("[MAIN] after freeze")

    cfg.TRAIN.TRAIN_VIDEO = cfg.DATA.DATASET in vid_set
    if cfg.TRAIN.TEACH1_LOAD_AS_IMG:
        cfg.TRAIN.TRAIN_VIDEO = False 
    
    cfg.DATA.DATASET_FIX = cfg.TRAIN.TEACH_DATASET_FIX
    if cfg.TRAIN.TEACH1_MODEL:
        cfg.MODEL.NAME = cfg.TRAIN.TEACH1_MODEL
        cfg.TEST.WEIGHT = cfg.TRAIN.TEACH1_MODEL_WT
        args.resume = True 

    if cfg.TRAIN.TEACH1:
        teacher_trainloader, _, _, teacher_dataset, _, _ = build_dataloader(cfg, local_rank=args.local_rank, teacher_mode=True )
        cfg.TRAIN.TEACH1_NUMCLASSES = teacher_dataset.num_train_pids
    else:
        teacher_trainloader, teacher_dataset = None , None 
        cfg.TRAIN.TEACH1_NUMCLASSES = None
    logger.info("[MAIN] after dataloader")

    if cfg.TRAIN.TEACH1_MODEL:
        teacher_model, _, _, _, _, _ = setup_model(cfg, args, logger, teacher_dataset, )
        teacher_model.eval()
        teacher_model.to(args.local_rank)
        teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.local_rank],find_unused_parameters=True)
        kwargs['teacher_model'] = teacher_model
        kwargs['mse'] = MSE()
        args.resume = student_resume
        cfg.TEST.WEIGHT = student_test_weight
    logger.info("hello logger")

    kwargs_external = {"training_mode" : None }
    if not args.eval:
        logger.info("[MAIN] after setup_model(student)")
        logger.info("[MAIN] before add_additional_attributes(student)")
        _, kwargs_external = add_additional_attributes(cfg, args)
        logger.info("[MAIN] after add_additional_attributes(student)")
        logger.info("[MAIN] calling do_train_w_teachers")

    if cfg.TRAIN.TEACH1_LOAD_AS_IMG:
        kwargs_external['training_mode'] = "image"

    # restore student cfg
    cfg.MODEL.NAME = student_model
    cfg.DATA.BATCH_SIZE = batch_size
    cfg.DATA.DATASET = student_dataset
    cfg.DATA.ROOT = student_dataset_root
    args.cal_eval = False 
    if cfg.TRAIN.HYBRID:
        cfg.DATA.DATASET_FIX = None
        cfg.TRAIN.COLOR_PROFILE = None
    else:
        cfg.DATA.DATASET_FIX = student_dataset_fix
    cfg.DATA.SAMPLING_PERCENTAGE = None 
    cfg.DATA.DATASET_SAMPLING_PERCENTAGE = None
    cfg.TRAIN.TRAIN_VIDEO = cfg.DATA.DATASET in vid_set
    cfg.TRAIN.COLOR_ADV = None 
    cfg.freeze()

    ################  STUDENT MODE  ################
    if "prcc" in cfg.DATA.DATASET:
        trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler,val_loader,val_loader_same= build_dataloader(cfg, local_rank=args.local_rank)  # prcc_test
    else:
        trainloader, queryloader, galleryloader, dataset, train_sampler,val_loader = build_dataloader(cfg, local_rank=args.local_rank)
    logger.info("[MAIN] after build_dataloader(student)")

    if cfg.TRAIN.TEACH1_NUMCLASSES is None:
        cfg.defrost(); cfg.TRAIN.TEACH1_NUMCLASSES = dataset.num_train_pids; cfg.freeze()
    
    model, loss_func, center_criterion, optimizer, optimizer_center, scheduler = \
        setup_model(cfg, args, logger, dataset)

    # unwrap if DDP
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        base_model = model.module
    else:
        base_model = model

    student_dim = 768
    teacher_dim = 1024

    projector = Mlp(
        in_features=student_dim,
        hidden_features=teacher_dim,
        out_features=teacher_dim,
        drop=0.0,
    )

    base_group = optimizer.param_groups[0]

    optimizer.add_param_group({
        "params": projector.parameters(),
        "lr": optimizer.param_groups[0]["lr"],
        "weight_decay": base_group.get("weight_decay", 0.0)
    })

    wd = cfg.SOLVER.WEIGHT_DECAY

    logger.info("WD group0: %s", optimizer.param_groups[0].get("weight_decay", None))
    logger.info("WD projector: %s", optimizer.param_groups[-1].get("weight_decay", None))

    _, kwargs_internal = add_additional_attributes(cfg, args)
    kwargs = add_external_training_fns(kwargs, kwargs_external, kwargs_internal)
    kwargs.pop('threshold_drop', None)

    if 'prcc' in cfg.DATA.DATASET:
        do_train_w_teachers_distillation(cfg, model, center_criterion, trainloader, optimizer,
                            optimizer_center, scheduler, loss_func, args.local_rank, dataset,
                            teacher_trainloader=teacher_trainloader, threshold_drop=100, teacher_dataset=teacher_dataset,
                            val_loader=val_loader, val_loader_same=val_loader_same,
                            eval=args.eval, projector=projector, **kwargs)
    else:
        do_train_w_teachers_distillation(cfg, model, center_criterion, trainloader, optimizer,
                            optimizer_center, scheduler, loss_func, args.local_rank, dataset,
                            teacher_trainloader=teacher_trainloader, threshold_drop=100, teacher_dataset=teacher_dataset,
                            val_loader=val_loader, eval=args.eval, projector=projector, **kwargs)
