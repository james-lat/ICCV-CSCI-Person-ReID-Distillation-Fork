import logging
import os
import time
import torch.distributed as dist
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval,R1_mAP_eval_LTCC, R1_mAP_eval_LaST
from torch.cuda import amp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
from processor.eval_mevid import test_mevid
from processor.eval_ccvid import test as vid_test
from processor.train_fn import *
from collections import defaultdict
import pickle
import sys 
from model.MHSA import Mlp
from torchvision.utils import save_image 
import torch.nn as nn
# from model.MHSA import Mlp
#Timm pr Function
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp
from torch.nn.parallel import DistributedDataParallel as DDP

def default_img_loader(cfg, data, ):
    text = None
    if cfg.MODEL.ADD_META:
        samples, targets, camids, _,clothes, meta = data
        meta = [m.float() for m in meta]
        meta = torch.stack(meta, dim=0)
        meta = meta.cuda(non_blocking=True)
        if cfg.MODEL.MASK_META:
            meta[:, 5:21] = 0
            meta[:, 35:57] = 0
            meta[:, 84] = 0
            meta[:, 90] = 0
            meta[:, 92:96] = 0
            meta[:, 97] = 0
            meta[:, 100] = 0
            meta[:, 102:105] = 0
    else:
        samples, targets, camids,_, clothes,meta,text = data
        meta = None

    samples = samples.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)
    clothes = clothes.cuda(non_blocking=True)
    
    return samples, targets, clothes, meta, camids, text

def evaluate_fn(
    cfg, model, val_loader, logger,
    evaluator_diff, evaluator_same, val_loader_same,
    device, epoch, rank_writer, mAP_writer, dataset,
    eval_mode=None, evaluator=None, dump=None,
    evaluator_general=None, queryloader=None, galleryloader=None,
    projector=None, aux_dump=None
):
    model.eval()
    cmc_overall, mAP_overall = None, None
    rank1, mAP, cmc_overall, mAP_overall = None, None, None, None

    do_dump = bool(eval_mode) or bool(dump)
    dump_w_index = bool(eval_mode)

    if 'prcc' in cfg.DATA.DATASET:
        evaluator_diff.reset()
        evaluator_same.reset()
        logger.info("Clothes changing setting")

        if dump_w_index:
            if aux_dump:
                rank1, mAP = test_w_index_w_aux(
                    cfg, model, evaluator_diff, val_loader, logger, device,
                    epoch, rank_writer, mAP_writer,
                    dump=do_dump,
                    dataset_name=cfg.DATA.DATASET,
                    prefix=" CC : "
                )
                logger.info("Standard setting")
                test_w_index_w_aux(
                    cfg, model, evaluator_same, val_loader_same, logger, device,
                    epoch, rank_writer, mAP_writer,
                    test=True,
                    dump=False,
                    dataset_name=cfg.DATA.DATASET,
                )
            else:
                rank1, mAP = test_w_index(
                    cfg, model, evaluator_diff, val_loader, logger, device,
                    epoch, rank_writer, mAP_writer,
                    dump=do_dump,
                    dataset_name=cfg.DATA.DATASET,
                    prefix=" CC : "
                )
                logger.info("Standard setting")
                test_w_index(
                    cfg, model, evaluator_same, val_loader_same, logger, device,
                    epoch, rank_writer, mAP_writer,
                    test=True,
                    dump=False,
                    dataset_name=cfg.DATA.DATASET,
                )
        else:
            rank1, mAP = test(
                cfg, model, evaluator_diff, val_loader, logger, device,
                epoch, rank_writer, mAP_writer,
                prefix=" CC : "
            )
            logger.info("Standard setting")
            test(
                cfg, model, evaluator_same, val_loader_same, logger, device,
                epoch, rank_writer, mAP_writer,
                test=True,
            )

    elif 'ltcc' in cfg.DATA.DATASET:
        evaluator_diff.reset()
        evaluator_general.reset()

        if dump_w_index:
            logger.info("Clothes changing setting")

            if aux_dump:
                evaluator_diff.reset()
                rank1_768, mAP_768 = test_w_index_w_aux(
                    cfg, model, evaluator_diff, val_loader, logger, device,
                    epoch, rank_writer, mAP_writer,
                    cc=True,
                    prefix=" CC[768]: ",
                    projector=None,              
                    dump=do_dump,
                    dataset_name=cfg.DATA.DATASET,
                )

                evaluator_diff.reset()
                rank1_1024, mAP_1024 = test_w_index_w_aux(
                    cfg, model, evaluator_diff, val_loader, logger, device,
                    epoch, rank_writer, mAP_writer,
                    cc=True,
                    prefix=" CC[1024]: ",
                    projector=projector,         
                    dump=do_dump,
                    dataset_name=cfg.DATA.DATASET,
                )

                rank1, mAP = rank1_1024, mAP_1024

                logger.info("Standard setting")
                test_w_index_w_aux(
                    cfg, model, evaluator_general, val_loader, logger, device,
                    epoch, rank_writer, mAP_writer,
                    test=True,
                    projector=None,              
                    prefix=" General: ",
                    dump=False,
                    dataset_name=cfg.DATA.DATASET,
                )

            else:
                evaluator_diff.reset()
                rank1_768, mAP_768 = test_w_index(
                    cfg, model, evaluator_diff, val_loader, logger, device,
                    epoch, rank_writer, mAP_writer,
                    cc=True,
                    prefix=" CC[768]: ",
                    projector=None,              
                    dump=do_dump,
                    dataset_name=cfg.DATA.DATASET,
                )

                evaluator_diff.reset()
                rank1_1024, mAP_1024 = test_w_index(
                    cfg, model, evaluator_diff, val_loader, logger, device,
                    epoch, rank_writer, mAP_writer,
                    cc=True,
                    prefix=" CC[1024]: ",
                    projector=projector,         
                    dump=do_dump,
                    dataset_name=cfg.DATA.DATASET,
                )

                # use projector metrics as main
                rank1, mAP = rank1_1024, mAP_1024

                logger.info("Standard setting")
                test_w_index(
                    cfg, model, evaluator_general, val_loader, logger, device,
                    epoch, rank_writer, mAP_writer,
                    test=True,
                    projector=None,
                    prefix=" General: ",
                    dump=False,
                    dataset_name=cfg.DATA.DATASET,
                )

        else:
            logger.info("Clothes changing setting")

            # CC[768]: student backbone
            evaluator_diff.reset()
            rank1_768, mAP_768 = test(
                cfg, model, evaluator_diff, val_loader, logger, device,
                epoch, rank_writer, mAP_writer,
                cc=True,
                prefix=" CC[768]: ",
                projector=None,                  
            )

            # CC[1024]: projector feature
            evaluator_diff.reset()
            rank1_1024, mAP_1024 = test(
                cfg, model, evaluator_diff, val_loader, logger, device,
                epoch, rank_writer, mAP_writer,
                cc=True,
                prefix=" CC[1024]: ",
                projector=projector,             
            )

            rank1, mAP = rank1_1024, mAP_1024

            logger.info("Standard setting")
            test(
                cfg, model, evaluator_general, val_loader, logger, device,
                epoch, rank_writer, mAP_writer,
                test=True,
                projector=None,                  
                prefix=" General: ",
            )

    elif 'mevid' in cfg.DATA.DATASET:
        rank1, mAP, cmc_overall, mAP_overall = test_mevid(
            cfg, model, val_loader[0], val_loader[1], dataset, device,
            dump_w_index=eval_mode,
            dump=do_dump,
        )

    elif 'ccvid' in cfg.DATA.DATASET:
        rank1, mAP = vid_test(
            cfg, model, val_loader[0], val_loader[1], dataset, device,
            dump_w_index=eval_mode,
            dump=do_dump,
        )

    if 'mevid' in cfg.DATA.DATASET:
        if dist.get_rank() == 0:
            st = "\n"
            for k1, k2 in zip(cmc_overall, mAP_overall):
                st += f" {k1} : {cmc_overall[k1]:.1%} & {k2} : {mAP_overall[k2]:.1%} \n"
            logger.info(f"==> {st}")
        dist.barrier()

    return rank1, mAP, cmc_overall, mAP_overall

def set_train_methods(cfg, model, local_rank, projector=None):
    eval_period = cfg.SOLVER.EVAL_PERIOD

    # Proper CUDA device object
    device = torch.device(f"cuda:{local_rank}")
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("EVA-attribure.train")
    _LOCAL_PROCESS_GROUP = None

    model.to(device)
    if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
        print('Using {} GPUs for training'.format(torch.cuda.device_count()))
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )

    if projector is not None:
        projector.to(device)

    if cfg.TENSORBOARD:
        train_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'train'))
        rank_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'rank'))
        mAP_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'mAP'))
    else:
        train_writer = None
        rank_writer = None
        mAP_writer = None

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    scaler = amp.GradScaler()

    if projector is not None:
        return (
            eval_period, device, epochs, logger,
            train_writer, rank_writer, mAP_writer,
            loss_meter, acc_meter, scaler,
            model, projector,
        )

    return (
        eval_period, device, epochs, logger,
        train_writer, rank_writer, mAP_writer,
        loss_meter, acc_meter, scaler,
        model
    )


def train_step(cfg, model, train_loader, optimizer, optimizer_center, loss_fn, scaler, loss_meter, acc_meter, scheduler, epoch, DEFAULT_LOADER=default_img_loader, train_writer=None , training_mode="image", **kwargs):
    log_period = cfg.SOLVER.LOG_PERIOD
    logger = logging.getLogger("EVA-attribure.train")

    model.train()

    for idx, data in enumerate(train_loader):

        samples, targets, clothes, meta, camids, text = DEFAULT_LOADER(cfg, data, )
        # save_image (normalize(samples), "temp.png")
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        with amp.autocast(enabled=True):
            if cfg.MODEL.ADD_META:
                if cfg.MODEL.CLOTH_ONLY:
                    score, feat = model(samples, clothes)
                else:
                    score, feat = model(samples, clothes, meta)
            else:
                score, feat = model(samples)
        loss = loss_fn(score, feat, targets, camids, training_mode=training_mode)
        if cfg.TENSORBOARD:
            train_writer.add_scalar('loss', loss.item(), epoch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
            for param in center_criterion.parameters():
                param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
            scaler.step(optimizer_center)
            scaler.update()
        if isinstance(score, list):
            acc = (score[0].max(1)[1] == targets).float().mean()
        else:
            acc = (score.max(1)[1] == targets).float().mean()

        loss_meter.update(loss.item(), samples.shape[0])
        acc_meter.update(acc, 1)

        torch.cuda.synchronize()
        if (idx + 1) % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(epoch, (idx + 1), len(train_loader),
                                loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
    return idx 
    
def eval_step(cfg, model, val_loader, evaluator_diff, evaluator_same, val_loader_same, device, epoch, rank_writer, mAP_writer, dataset, 
    best_rank1_dict, best_map_dict, best_rank1, best_map, evaluator=None, evaluator_general=None , queryloader=None, galleryloader=None, projector=None, threshold_drop=10):
    logger = logging.getLogger("EVA-attribure.train")
    rank1, mAP, cmc_overall, mAP_overall =  evaluate_fn(cfg, model, val_loader, logger, evaluator_diff, evaluator_same, val_loader_same, device, epoch, rank_writer, mAP_writer, dataset, evaluator=evaluator, evaluator_general=evaluator_general, queryloader=queryloader, galleryloader=galleryloader, projector=projector)    
    best_epoch = 0 

    if 'mevid' in cfg.DATA.DATASET :
        for key in cmc_overall:best_rank1_dict[key]= max(best_rank1_dict[key], cmc_overall[key])
        for key in mAP_overall:best_map_dict[key] = max(mAP_overall[key], best_map_dict[key])
        
        rank1 = torch.tensor(rank1).cuda()
        dist.barrier()
        dist.broadcast(rank1, src=0)
        rank1 = rank1.cpu().item()
        
        
        
    # print(dist.get_rank(), rank1, best_rank1)
    if (best_rank1 < 1 and rank1 + threshold_drop / 100  < best_rank1) or (rank1 + threshold_drop < best_rank1):
        logger.info(f" \n\n\n *** TERMINATING... Top1 : {rank1:.1%} << Best Rank-1 {best_rank1:.1%} - {threshold_drop} *** \n\n\n")
        logger.info("==> Best Rank-1 {:.1%}, Best Map {:.1%} achieved at epoch {}".format(best_rank1, best_map, best_epoch))
        dist.barrier()
        sys.exit()

    is_best = rank1 > best_rank1
    best_map = max(best_map, mAP)
    
    if is_best:
        best_rank1 = rank1
        best_epoch = epoch
        if dist.get_rank() == 0:
            logger.info("==> Best Rank-1 {:.1%}, Best Map {:.1%} achieved at epoch {}".format(best_rank1, best_map, best_epoch))
            if 'mevid' in cfg.DATA.DATASET :
                st = "\n"
                for k1, k2 in zip(best_rank1_dict, best_map_dict): st += f" {k1} : {best_rank1_dict[k1]:.1%} & {k2} : {best_map_dict[k2]:.1%} \n"
                logger.info(f"==> {st}")
        if cfg.MODEL.DIST_TRAIN:
            if dist.get_rank() == 0:
                logger.info(f"Saving the model Now ....., {rank1:.4f}, {mAP:.4f}")
                torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))

    if cfg.MODEL.DIST_TRAIN and dist.get_rank() == 0 and 'briar' in cfg.DATA.DATASET :
        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + f'_{epoch}.pth'))

    return best_map, best_rank1, best_epoch, best_rank1_dict, best_map_dict 

def evaluator_gen(cfg, dataset):
    evaluator_diff, evaluator_general, evaluator_same, evaluator = None, None, None, None 
    if 'ltcc' in cfg.DATA.DATASET :
        evaluator_diff = R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, length = len(dataset.query) + len(dataset.gallery))  # ltcc
        evaluator_general = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, length = len(dataset.query) + len(dataset.gallery))
        evaluator_diff.reset()
        evaluator_general.reset()
    elif 'prcc' in cfg.DATA.DATASET :
        evaluator_diff = R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, length = dataset.num_query_imgs_diff + len(dataset.gallery))  # prcc
        evaluator_same = R1_mAP_eval(dataset.num_query_imgs_same, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, length = dataset.num_query_imgs_same + len(dataset.gallery))
        evaluator_diff.reset()
        evaluator_same.reset()
    elif ('mevid' in cfg.DATA.DATASET) or ('casia' in cfg.DATA.DATASET) or ('ccvid' in cfg.DATA.DATASET) or ('briar' in cfg.DATA.DATASET):
        evaluator_diff, evaluator_general, evaluator_same = None, None, None
    elif "last" in cfg.DATA.DATASET:
        evaluator = R1_mAP_eval_LaST(num_query=dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, q_length = len(dataset.query) , g_length = len(dataset.gallery))
        evaluator.reset()
    else:
        evaluator = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, length = len(dataset.query) + len(dataset.gallery))
        evaluator.reset()
    return evaluator_diff, evaluator_general, evaluator_same, evaluator



################ Train ################
#### Default Normal Video / Image 
def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             local_rank,
             dataset,
             val_loader = None,
             val_loader_same = None,
             eval=None, save5=None, TRAIN_step_FN=train_step, training_mode="image", queryloader=None, galleryloader=None, threshold_drop=10, projector=None, **kwargs):
    
    eval_period, device, epochs, logger, train_writer, rank_writer, \
        mAP_writer, loss_meter, acc_meter, scaler, model = set_train_methods(cfg, model, local_rank)
    
    evaluator_diff, evaluator_general, evaluator_same, evaluator = evaluator_gen(cfg, dataset)

    best_rank1 = -np.inf
    best_map = -np.inf
    best_epoch = 0
    mAP = 0 
    best_map_dict = defaultdict(int)
    best_rank1_dict = defaultdict(int)
    start_train_time = time.time()
    
    if eval:
        if cfg.TRAIN_DUMP:
            if TRAIN_step_FN == train_w_color_labels:
                train_w_color_labels_dump(cfg, model, train_loader, optimizer, optimizer_center, loss_fn, scaler, loss_meter, acc_meter, scheduler,  -1, train_writer=train_writer, training_mode=training_mode, **kwargs )
            elif TRAIN_step_FN == train_step:
                train_step_labels_dump(cfg, model, train_loader, optimizer, optimizer_center, loss_fn, scaler, loss_meter, acc_meter, scheduler,  -1, train_writer=train_writer, training_mode=training_mode, **kwargs )
            else:
                assert False, "not verified yet"
        elif cfg.AUX_DUMP:
            rank1, mAP, _, _ =  evaluate_fn(cfg, model, val_loader, logger, evaluator_diff, evaluator_same, val_loader_same, device, -1, rank_writer, mAP_writer, dataset, eval_mode=cfg.TEST.MODE, evaluator=evaluator, dump=True, evaluator_general=evaluator_general, queryloader=queryloader, galleryloader=galleryloader, aux_dump=True)    
            logger.info("==> EVAL: Rank-1 {:.1%}, Map {:.1%} achieved".format(rank1, mAP))
        else:
            rank1, mAP, _, _ =  evaluate_fn(cfg, model, val_loader, logger, evaluator_diff, evaluator_same, val_loader_same, device, -1, rank_writer, mAP_writer, dataset, eval_mode=cfg.TEST.MODE, evaluator=evaluator, dump=True, evaluator_general=evaluator_general, queryloader=queryloader, galleryloader=galleryloader,)    
            logger.info("==> EVAL: Rank-1 {:.1%}, Map {:.1%} achieved".format(rank1, mAP))
        return 
    idx =0            
    logger.info('start training')
    logger.info("Train Start !!")
    for epoch in range(cfg.TRAIN.START_EPOCH, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()

        scheduler.step(epoch)
        idx = TRAIN_step_FN(cfg, model, train_loader, optimizer, optimizer_center, loss_fn, scaler, loss_meter, acc_meter, scheduler,  epoch, train_writer=train_writer, training_mode=training_mode, **kwargs )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (idx + 1)

        if epoch % eval_period == 0:
            best_map, best_rank1, potential_best_epoch, best_rank1_dict, best_map_dict = eval_step(cfg, model, val_loader, evaluator_diff, evaluator_same, val_loader_same, device, epoch, rank_writer, mAP_writer, dataset,
            best_rank1_dict, best_map_dict, best_rank1, best_map, evaluator=evaluator, evaluator_general=evaluator_general, queryloader=queryloader, galleryloader=galleryloader, projector=None, threshold_drop=threshold_drop)
            best_epoch = max(best_epoch, potential_best_epoch)
        
        if save5 and epoch % 5 == 0 and dist.get_rank() == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + f'_{epoch}.pth'))

            
    total_time = time.time() - start_train_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    logger.info("==> Best Rank-1 {:.1%}, Best Map {:.1%} achieved at epoch {}".format(best_rank1, best_map, best_epoch))
    if 'mevid' in cfg.DATA.DATASET :
        st = "\n"
        for k1, k2 in zip(best_rank1_dict, best_map_dict): st += f" {k1} : {best_rank1_dict[k1]:.1%} & {k2} : {best_map_dict[k2]:.1%} \n"
        logger.info(f"==> {st}")

    if dist.get_rank() == 0:
        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_last.pth'))
  
def train_step_distillation(
    cfg, model, train_loader, optimizer, optimizer_center, loss_fn, scaler,
    loss_meter, acc_meter, scheduler, epoch,
    DEFAULT_LOADER=default_img_loader, projector=None,
    train_writer=None, training_mode="image", **kwargs
):
    log_period = cfg.SOLVER.LOG_PERIOD
    logger = logging.getLogger("EVA-attribure.train")

    teacher = kwargs.get("teacher_model", None)
    mse = kwargs.get("mse", None)

    KD_TARGET  = float(getattr(cfg.TRAIN, "KD_WEIGHT", 0.5))
    KD_WARMUP_EPOCHS = int(getattr(cfg.TRAIN, "KD_WARMUP_EPOCHS", 0))

    deviced = False
    model.train()
    
    if teacher is not None:
        teacher_ref = teacher.module if hasattr(teacher, "module") else teacher
        teacher_ref.eval()
    else:
        teacher_ref = None

    last_idx = -1
    for cycle in range(2):   # 0 = base  1 = KD
        for idx, data in enumerate(train_loader):
            last_idx = idx
            use_kd = (cycle == 1) 
            samples, targets, clothes, meta, camids, text = DEFAULT_LOADER(cfg, data)

            optimizer.zero_grad()
            optimizer_center.zero_grad()

            s_out = model(samples, clothes)
            s_score, s_feat, s_color_output, s_color_feats, s_dist = s_out

            feat_768 = s_feat

            if projector is not None:
                student_1024 = projector(feat_768)
            else:
                student_1024 = feat_768  

            if epoch == 1 and idx == 0:
                logger.info(f"[DEBUG] feat_768.shape: {tuple(feat_768.shape)}")
                logger.info(f"[DEBUG] student_1024.shape: {tuple(student_1024.shape)}")

            with amp.autocast(enabled=True):
                main_loss = loss_fn(
                    s_score, feat_768, targets, camids,
                    training_mode=training_mode,
                )
                main_loss += loss_fn(
                    s_score, student_1024, targets, camids,
                    training_mode=training_mode,
                )
               
            if cfg.TENSORBOARD and train_writer is not None:
                tag = "loss_main_base" if not use_kd else "loss_main_kd_cycle"
                train_writer.add_scalar(tag, float(main_loss.item()), epoch)

            if isinstance(s_score, list):
                acc = (s_score[0].max(1)[1] == targets).float().mean()
            elif s_score is not None:
                acc = (s_score.max(1)[1] == targets).float().mean()
            else:
                logger.info("Score is None, setting acc=0")
                acc = torch.tensor(0.0, device=feat_768.device)

            kd_loss = 0.0
            if use_kd:
                with torch.no_grad():
                    with amp.autocast(enabled=True):   
                        if getattr(cfg.TRAIN, "TEACH1_LOAD_AS_IMG", False):
                            tout = teacher_ref(samples)
                        else:
                            if cfg.MODEL.ADD_META:
                                if cfg.MODEL.CLOTH_ONLY:
                                    tout = teacher_ref(samples, clothes)
                                else:
                                    tout = teacher_ref(samples, clothes, meta)
                            else:
                                tout = teacher_ref(samples, clothes)

                    if isinstance(tout, (list, tuple)):
                        teacher_1024 = tout[0].float()
                    else:
                        teacher_1024 = tout.float()

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                    s_raw = student_1024.float()
                    t_raw = teacher_1024.float()

                    s_hat = F.normalize(s_raw, p=2, dim=-1, eps=1e-6)
                    t_hat = F.normalize(t_raw, p=2, dim=-1, eps=1e-6)

                    kd_raw = mse(s_hat, t_hat)
                    cur_kd_w = 0.5
                    kd_loss = cur_kd_w * kd_raw


                if cfg.TENSORBOARD and train_writer is not None:
                    train_writer.add_scalar("loss_kd", float(kd_loss.item()), epoch)

            # Loss per cycle
            if isinstance(kd_loss, torch.Tensor):
                total_loss = main_loss + kd_loss
            else:
                total_loss = main_loss

            if idx == 500:
                ml = float(main_loss.item())
                kl = float(kd_loss.item() if isinstance(kd_loss, torch.Tensor) else 0.0)
                logger.info(
                    f"[DEBUG KD] epoch={epoch} cycle={cycle} main_loss={ml:.4f} kd_loss={kl:.4f} KD_TARGET={KD_TARGET}"
                )

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)

            if "center" in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)

            scaler.update()

            loss_meter.update(main_loss.item(), samples.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (idx + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}][cycle {}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                        epoch,
                        cycle,
                        (idx + 1),
                        len(train_loader),
                        loss_meter.avg,
                        acc_meter.avg,
                        scheduler._get_lr(epoch)[0],
                    )
                )

    return last_idx


def do_train_w_teachers_distillation(cfg,
             model,
             center_criterion,
             train_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             local_rank,
             dataset,
             val_loader = None,
             val_loader_same = None,
             eval=None, save5=None, TRAIN_step_FN=train_step_distillation, TRAIN_ext_step_FN=train_step_distillation,
             teacher_trainloader=None, teacher_dataset=None, training_mode="image", teacher_training_mode="video", queryloader=None, galleryloader=None, threshold_drop=10, projector=None, **kwargs):

    eval_period, device, epochs, logger, train_writer, rank_writer, \
        mAP_writer, loss_meter, acc_meter, scaler, model, projector = set_train_methods(cfg, model, local_rank, projector)
    
    evaluator_diff, evaluator_general, evaluator_same, evaluator = evaluator_gen(cfg, dataset)
    print("BEFORE")
    teacher_model = kwargs.get("teacher_model", None)
    logger.info("BEFORE BEFORE BEFORE:")
    if teacher_model is not None and val_loader is not None:
        logger.info("==> Teacher pre-eval (before student training)")
        # freeze & eval
        for p in teacher_model.parameters():
            p.requires_grad = False
        teacher_model.eval()
        teacher_ref = teacher_model.module if hasattr(teacher_model, "module") else teacher_model
        rank1_t, mAP_t, _, _ = evaluate_fn(
            cfg, teacher_ref,          
            val_loader,
            logger,
            evaluator_diff,
            evaluator_same,
            val_loader_same,
            device,
            -1,                        
            rank_writer,
            mAP_writer,
            dataset,
            eval_mode=cfg.TEST.MODE,
            evaluator=evaluator,
            dump=True,
            evaluator_general=evaluator_general,
            queryloader=queryloader,
            galleryloader=galleryloader,
        )

        logger.info(
            "==> TEACHER baseline: Rank-1 {:.1%}, mAP {:.1%}".format(rank1_t, mAP_t)
        )

    else:
        logger.info('ELSE')

    loss_meter_teacher = AverageMeter()
    acc_meter_teacher = AverageMeter()
    best_rank1 = -np.inf
    best_map = -np.inf
    best_epoch = 0
    mAP = 0 
    best_map_dict = defaultdict(int)
    best_rank1_dict = defaultdict(int)
    start_train_time = time.time()
    
    if eval:
        rank1, mAP, _, _ =  evaluate_fn(cfg, model, val_loader, logger, evaluator_diff, evaluator_same, val_loader_same, device, -1, rank_writer, mAP_writer, dataset, eval_mode=cfg.TEST.MODE, evaluator=evaluator, dump=True, evaluator_general=evaluator_general, queryloader=queryloader, galleryloader=galleryloader)    
        logger.info("==> EVAL: Rank-1 {:.1%}, Map {:.1%} achieved".format(rank1, mAP))
        return 
    idx =0            
    DEFAULT_LOADER=default_img_loader
    logger.info('start training')
    logger.info("Train Start !!")
    
    for epoch in range(cfg.TRAIN.START_EPOCH, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        loss_meter_teacher.reset()

        acc_meter.reset()
        acc_meter_teacher.reset()

        scheduler.step(epoch)

        logger.info("==> Student Training .... ")
        idx = TRAIN_step_FN(cfg, model, train_loader, optimizer, optimizer_center, loss_fn, scaler, loss_meter, acc_meter, scheduler,  epoch, train_writer=train_writer, training_mode=training_mode, projector=projector, **kwargs )
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (idx + 1)

        if epoch % eval_period == 0:
            best_map, best_rank1, potential_best_epoch, best_rank1_dict, best_map_dict = eval_step(cfg, model, val_loader, evaluator_diff, evaluator_same, val_loader_same, device, epoch, rank_writer, mAP_writer, dataset,
            best_rank1_dict, best_map_dict, best_rank1, best_map, evaluator=evaluator, evaluator_general=evaluator_general, queryloader=queryloader, galleryloader=galleryloader, projector=projector, threshold_drop=threshold_drop)
            best_epoch = max(best_epoch, potential_best_epoch)
        
        if save5 and epoch % 5 == 0 and dist.get_rank() == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + f'_{epoch}.pth'))

    total_time = time.time() - start_train_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    logger.info("==> Best Rank-1 {:.1%}, Best Map {:.1%} achieved at epoch {}".format(best_rank1, best_map, best_epoch))
    if 'mevid' in cfg.DATA.DATASET :
        st = "\n"
        for k1, k2 in zip(best_rank1_dict, best_map_dict): st += f" {k1} : {best_rank1_dict[k1]:.1%} & {k2} : {best_map_dict[k2]:.1%} \n"
        logger.info(f"==> {st}")
                
    if dist.get_rank() == 0:
        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_last.pth'))


def do_train_w_teachers(cfg,
             model,
             center_criterion,
             train_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             local_rank,
             dataset,
             val_loader = None,
             val_loader_same = None,
             eval=None, save5=None, TRAIN_step_FN=train_step, TRAIN_ext_step_FN=train_step,
             teacher_trainloader=None, teacher_dataset=None, training_mode="image",
             teacher_training_mode="video", queryloader=None, galleryloader=None, threshold_drop=10, **kwargs):

    eval_period, device, epochs, logger, train_writer, rank_writer, \
        mAP_writer, loss_meter, acc_meter, scaler, model = set_train_methods(cfg, model, local_rank)
    
    evaluator_diff, evaluator_general, evaluator_same, evaluator = evaluator_gen(cfg, dataset)

    loss_meter_teacher = AverageMeter()
    acc_meter_teacher = AverageMeter()
    best_rank1 = -np.inf
    best_map = -np.inf
    best_epoch = 0
    mAP = 0 
    best_map_dict = defaultdict(int)
    best_rank1_dict = defaultdict(int)
    start_train_time = time.time()
    
    if eval:
        rank1, mAP, _, _ =  evaluate_fn(cfg, model, val_loader, logger, evaluator_diff, evaluator_same, val_loader_same, device, -1, rank_writer, mAP_writer, dataset, eval_mode=cfg.TEST.MODE, evaluator=evaluator, dump=True, evaluator_general=evaluator_general, queryloader=queryloader, galleryloader=galleryloader)    
        logger.info("==> EVAL: Rank-1 {:.1%}, Map {:.1%} achieved".format(rank1, mAP))
        return 
    idx =0            
    DEFAULT_LOADER=default_img_loader
    logger.info('start training')
    logger.info("Train Start !!")
    
    for epoch in range(cfg.TRAIN.START_EPOCH, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        loss_meter_teacher.reset()

        acc_meter.reset()
        acc_meter_teacher.reset()

        scheduler.step(epoch)
        if not cfg.TRAIN.DEBUG:
            logger.info("==> Teacher Training .... ")
            model.module.student_mode = True
            model.student_mode = True 
            TRAIN_ext_step_FN(cfg, model, teacher_trainloader, optimizer,  optimizer_center,  loss_fn,  scaler, loss_meter_teacher, acc_meter_teacher, scheduler,   epoch, train_writer=train_writer, training_mode=teacher_training_mode, **kwargs )    
                            # cfg, model, train_loader,        optimizer,  optimizer_center,   loss_fn, scaler, loss_meter,         acc_meter,         scheduler,   epoch, train_writer=train_writer, training_mode=training_mode,         **kwargs )
            model.student_mode = False 
            model.module.student_mode = False

        logger.info("==> Student Training .... ")
        idx = train(cfg, model, train_loader, optimizer, optimizer_center, loss_fn, scaler, loss_meter, acc_meter, scheduler,  epoch, train_writer=train_writer, training_mode=training_mode, **kwargs )
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (idx + 1)

        # name = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + f'_{epoch}.pth')
        # torch.save(model.state_dict(), name)

        if epoch % eval_period == 0:
            best_map, best_rank1, potential_best_epoch, best_rank1_dict, best_map_dict = eval_step(cfg, model, val_loader, evaluator_diff, evaluator_same, val_loader_same, device, epoch, rank_writer, mAP_writer, dataset,
            best_rank1_dict, best_map_dict, best_rank1, best_map, evaluator=evaluator, evaluator_general=evaluator_general, queryloader=queryloader, galleryloader=galleryloader, projector=None, threshold_drop=threshold_drop)
            best_epoch = max(best_epoch, potential_best_epoch)
        
        if save5 and epoch % 5 == 0 and dist.get_rank() == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + f'_{epoch}.pth'))

    total_time = time.time() - start_train_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    logger.info("==> Best Rank-1 {:.1%}, Best Map {:.1%} achieved at epoch {}".format(best_rank1, best_map, best_epoch))
    if 'mevid' in cfg.DATA.DATASET :
        st = "\n"
        for k1, k2 in zip(best_rank1_dict, best_map_dict): st += f" {k1} : {best_rank1_dict[k1]:.1%} & {k2} : {best_map_dict[k2]:.1%} \n"
        logger.info(f"==> {st}")
                
    if dist.get_rank() == 0:
        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_last.pth'))
  


################ Eval ################
def do_inference(cfg,
                 model,
                 dataset,
                 val_loader = None,
                 val_loader_same=None):
    logger = logging.getLogger("EVA-attribure.test")
    logger.info("Enter inferencing")

    logger.info("transreid inferencing")
    device = "cuda"
    if 'ltcc' in cfg.DATA.DATASET:
        evaluator_diff = R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)  # ltcc
        evaluator_general = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    elif 'prcc' in cfg.DATA.DATASET:
        evaluator_diff = R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)  # prcc
        evaluator_same = R1_mAP_eval(dataset.num_query_imgs_same, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    if 'ltcc' in cfg.DATA.DATASET:
        evaluator_diff.reset()
        evaluator_general.reset()

    elif 'prcc' in cfg.DATA.DATASET:
        evaluator_diff.reset()
        evaluator_same.reset()
    else:
        evaluator.reset()
    model.to(device)
    model.eval()
    if 'prcc' in cfg.DATA.DATASET:
        logger.info("Clothes changing setting")
        test(cfg, model, evaluator_diff, val_loader, logger, device, test=True)
        logger.info("Standard setting")
        test(cfg, model, evaluator_same, val_loader_same, logger, device, test=True)
    elif 'ltcc' in cfg.DATA.DATASET :
        logger.info("Clothes changing setting")
        test(cfg, model, evaluator_diff, val_loader, logger, device, test=True,cc=True)
        logger.info("Standard setting")
        test(cfg, model, evaluator_general, val_loader, logger, device, test=True)
    else:
        test(cfg, model, evaluator, val_loader, logger, device, test=True)

def test(cfg, model, evaluator, val_loader, logger, device, epoch=None,
         rank_writer=None, mAP_writer=None, test=False, cc=False,
         prefix=None, dump=None, projector=None):

    logger.info("[EVAL] Entered test()")

    # put projector on the right device and in eval mode

    for n_iter, (imgs, pids, camids, clothes_id, clothes_ids, meta) in enumerate(val_loader):
        with torch.no_grad():
            imgs = imgs.to(device)
            meta = meta.to(device)
            clothes_ids = clothes_ids.to(device)
            meta = meta.to(torch.float32)

            if cfg.MODEL.CLOTH_ONLY:
                raw_out = model(imgs, clothes_ids)
            else:
                if cfg.MODEL.MASK_META:
                    meta[:, 5:21] = 0
                    meta[:, 35:57] = 0
                    meta[:, 84] = 0
                    meta[:, 90] = 0
                    meta[:, 92:96] = 0
                    meta[:, 97] = 0
                    meta[:, 100] = 0
                    meta[:, 102:105] = 0
                if cfg.TEST.TYPE == 'image_only':
                    meta = torch.zeros_like(meta)
                raw_out = model(imgs, clothes_ids, meta)

            # get backbone feature (feat_768)
            if isinstance(raw_out, (list, tuple)):
                tensor_feats = [x for x in raw_out if torch.is_tensor(x)]
                assert tensor_feats, f"test(): raw_out has no tensor element: {type(raw_out)}"
                feat_768 = tensor_feats[0]
            else:
                if not torch.is_tensor(raw_out):
                    raise RuntimeError(f"Unexpected eval output type: {type(raw_out)}")
                feat_768 = raw_out

            # choose which feature we actually evaluate with
            if projector is not None:
                feat = projector(feat_768)
            else:
                feat = feat_768

            # normal evaluator update
            if cc:
                evaluator.update((feat, pids, camids, clothes_id))
            else:
                evaluator.update((feat, pids, camids))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()

    if not prefix:
        prefix = " CC: " if cc else " SC: "
    string = "{} CMC curve, ".format(prefix)
    for r in [1, 5, 10]:
        string += "Rank-{:<3}:{:.1%}  ".format(r, cmc[r - 1])
    logger.info(string)
    logger.info("{} mAP Acc. :{:.1%}".format(prefix, mAP))

    if test:
        torch.cuda.empty_cache()
        return

    logger.info("Validation Results - Epoch: {}".format(epoch))
    rank1 = cmc[0]
    if rank_writer:
        rank_writer.add_scalar('rank1', rank1, epoch)
        mAP_writer.add_scalar('mAP', mAP, epoch)
    torch.cuda.empty_cache()
    return rank1, mAP



def test_w_index(cfg, model, evaluator, val_loader, logger, device, epoch=None, rank_writer=None, mAP_writer=None,test=False,cc=False, prefix=None, dump=None, dataset_name=None):
    indices = []
    for n_iter, (imgs, pids, camids, clothes_id, clothes_ids, meta, index) in enumerate(val_loader):
        with torch.no_grad():
            imgs = imgs.to(device)
            meta = meta.to(device)
            clothes_ids = clothes_ids.to(device)
            meta = meta.to(torch.float32)
            if cfg.MODEL.CLOTH_ONLY:
                feat = model(imgs, clothes_ids)
            else:
                if cfg.MODEL.MASK_META:
                    meta[:, 5:21] = 0
                    meta[:, 35:57] = 0
                    meta[:, 84] = 0
                    meta[:, 90] = 0
                    meta[:, 92:96] = 0
                    meta[:, 97] = 0
                    meta[:, 100] = 0
                    meta[:, 102:105] = 0
                if cfg.TEST.TYPE == 'image_only':
                    meta = torch.zeros_like(meta)
                feat = model(imgs, clothes_ids, meta)
            if cc:
                evaluator.update((feat, pids, camids, clothes_id))
            else:
                evaluator.update((feat, pids, camids))
            indices.append(index)
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    
    if not prefix:
        prefix = " CC: " if cc else  " SC: "
    string = "{} CMC curve, ".format(prefix)
    for r in [1, 5, 10]:
        string+= "Rank-{:<3}:{:.1%}  ".format(r, cmc[r - 1])
    logger.info(string)
    logger.info("{} mAP Acc. :{:.1%}".format(prefix,  mAP ))
    if test :
        torch.cuda.empty_cache()
        return
    logger.info("Validation Results - Epoch: {}".format(epoch))
    rank1 = cmc[0]
    if rank_writer:
        rank_writer.add_scalar('rank1', rank1, epoch)
        mAP_writer.add_scalar('mAP', mAP, epoch)
    torch.cuda.empty_cache()

    if dump:
        if dist.get_rank() == 0:
            indices = torch.cat(indices)
            if cfg.TAG:
                evaluator.dump_vals(indices, cfg.TAG)
            else:
                evaluator.dump_vals(indices, dataset_name)
        dist.barrier()

    return rank1, mAP

def test_w_index_w_aux(cfg, model, evaluator, val_loader, logger, device, epoch=None, rank_writer=None, mAP_writer=None,test=False,cc=False, prefix=None, dump=None, dataset_name=None):
    print(" *** DUMPING WITH 2 AUX OUTPUT *** ")
    indices = []
    model.dump_aux = True 
    
    extrafeats = []
    extrafeats_outputs = [] 
    for n_iter, (imgs, pids, camids, clothes_id, clothes_ids, meta, index) in enumerate(val_loader):
        with torch.no_grad():
            imgs = imgs.to(device)
            meta = meta.to(device)
            clothes_ids = clothes_ids.to(device)
            meta = meta.to(torch.float32)
            feat, extra_token_feats, extra_token_output = model(imgs, clothes_ids)
            extrafeats_outputs.append(extra_token_output)
            extrafeats.append(extra_token_feats)
            if cfg.TEST.CONCAT_COLORS:
                feat = torch.cat([feat, extra_token_feats],-1)
            if cc:
                evaluator.update((feat, pids, camids, clothes_id))
            else:
                evaluator.update((feat, pids, camids))
            indices.append(index)
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    
    if not prefix:
        prefix = " CC: " if cc else  " SC: "
    string = "{} CMC curve, ".format(prefix)
    for r in [1, 5, 10]:
        string+= "Rank-{:<3}:{:.1%}  ".format(r, cmc[r - 1])
    logger.info(string)
    logger.info("{} mAP Acc. :{:.1%}".format(prefix,  mAP ))
    if test :
        torch.cuda.empty_cache()
        return
    logger.info("Validation Results - Epoch: {}".format(epoch))
    rank1 = cmc[0]
    if rank_writer:
        rank_writer.add_scalar('rank1', rank1, epoch)
        mAP_writer.add_scalar('mAP', mAP, epoch)
    torch.cuda.empty_cache()

    if dump:
        if dist.get_rank() == 0:
            from utils.metrics import concat_all_gather
            extrafeats = torch.cat(extrafeats)
            extrafeats_outputs = torch.cat(extrafeats_outputs)

            extrafeats, extrafeats_outputs = concat_all_gather([extrafeats, extrafeats_outputs], evaluator.length)

            extrafeats_qf = extrafeats[:evaluator.num_query]
            extrafeats_outputs_qf = extrafeats_outputs[:evaluator.num_query]

            extrafeats_gf = extrafeats[evaluator.num_query:]
            extrafeats_outputs_gf = extrafeats_outputs[evaluator.num_query:]
            
            indices = torch.cat(indices)
            if cfg.TAG:
                evaluator.dump_vals(indices, cfg.TAG, aux_dump=dict(extrafeats_qf=extrafeats_qf, extrafeats_outputs_qf=extrafeats_outputs_qf, extrafeats_gf=extrafeats_gf, extrafeats_outputs_gf=extrafeats_outputs_gf))
            else:
                evaluator.dump_vals(indices, dataset_name, aux_dump=dict(extrafeats_qf=extrafeats_qf, extrafeats_outputs_qf=extrafeats_outputs_qf, extrafeats_gf=extrafeats_gf, extrafeats_outputs_gf=extrafeats_outputs_gf))
        dist.barrier()
    model.dump_aux = False 
    return rank1, mAP



