#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for "Computing a human-like RT metric from stable recurrent vision models".
Starting point was https://github.com/c-rbp/pathfinder_experiments/blob/main/mainclean.py.

"""
import json
import os
import tempfile
import time
from collections import defaultdict
from functools import partial
from statistics import mean

import torch
import torch.nn.parallel
import torch.optim

from sklearn.metrics import classification_report
from torch.nn import CrossEntropyLoss
from utils.loss import EDLLoss, get_edl_diagnostics
from utils.misc import AverageMeter, save_checkpoint, save_npz
from utils.opts import parser
from datasets import setup_dataset
from models import setup_model
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True

# Setting up
# ======================================================================================================================

global_step = 1
args = parser.parse_args()

# Wandb
if args.wandb:
    if args.wandb_project is None or args.wandb_entity is None:
        parser.error("--wandb requires --wandb_project and --wandb_entity.")
    import wandb  # https://wandb.ai/

    wname = args.name
    wandb_config = vars(args)

    wandb.init(project=args.wandb_project,
               entity=args.wandb_entity,
               name=wname,
               dir=tempfile.gettempdir(),
               config=wandb_config)
    wandb.run.log_code('.')

# GPUs
if len(args.gpu_ids) != 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu_id) for gpu_id in args.gpu_ids])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Saving
results_folder = 'results/{0}/'.format(args.name)
os.makedirs(results_folder, exist_ok=False)
with open(os.path.join(results_folder, 'opts.json'), 'w') as f:
    opts = vars(args)
    opts['num_gpus'] = torch.cuda.device_count()
    if args.wandb:
        opts["wandb_id"] = wandb.run.id
    json.dump(opts, f)

# Logging
exp_logging = args.log

# Data
# =======================================================================================================================
data_root = args.data_root
del(args.data_root)

# Note: training batches will be shuffled by the dataloader
train_set = setup_dataset(args.dataset_str_train, data_root, subset=args.subset_train, shuffle=False, **vars(args))
val_set = setup_dataset(args.dataset_str_val, data_root, subset=1.0, shuffle=True, **vars(args))

train_loader = DataLoader(train_set,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=4,  # We used num_workers = 4 and 4 gpus
                          pin_memory=args.pin_memory,
                          drop_last=True)
val_loader = DataLoader(val_set,
                        batch_size=args.batch_size,
                        shuffle=False,  # already shuffled once, don't shuffle again every epoch
                        num_workers=4,  # We used num_workers = 4 and 4 gpus
                        pin_memory=args.pin_memory,
                        drop_last=True)

# Model
# =======================================================================================================================
model = setup_model(**vars(args))

if args.parallel is True:
    model = torch.nn.DataParallel(model).to(device)
    print("Loading parallel finished on GPU count:", torch.cuda.device_count())
else:
    print(device)
    model = model.to(device)
    print("Loading finished")

if args.wandb:
    wandb.watch(model, None, "gradients", args.print_freq)


# Training settings
# =======================================================================================================================

jacobian_penalty = args.penalty

# Criterion
if args.loss_fn == 'cross_entropy':
    criterion = CrossEntropyLoss(reduction='none').to(device)

elif args.loss_fn == 'EDL':
    edl = EDLLoss(num_classes=args.n_classes).to(device)
else:
    raise NotImplementedError('Loss not implemented')

# Optimizer
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
else:
    raise NotImplementedError('optimizer not implemented')
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.7)


# Validation function
# ======================================================================================================================

def validate(val_loader, model, criterion, epoch, logiters=None):
    print('global_step at start validation {}'.format(global_step))
    batch_timev = AverageMeter()  # how long to evaluate the batch
    lossesv = AverageMeter()  # loss
    accv = AverageMeter()  # accuracy
    sensitivityv = AverageMeter()  # true positive rate
    specificityv = AverageMeter()  # true negative rate
    f1scorev = AverageMeter()  # f1 score
    n_posv = AverageMeter()  # number of positive (1) samples in batch
    ev_succv = AverageMeter()  # EDL evidence for accurate predictions
    ev_failv = AverageMeter()  # EDL evidence for inaccurate predictions
    u_succv = AverageMeter()  # EDL uncertainty for accurate predictions
    u_failv = AverageMeter()  # EDL uncertainty for inaccurate predictions

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            target = batch["label"].long()
            target = target.to(device)
            imgs = batch["image"].cuda()

            output_dict = model.forward(imgs, epoch, i, target, criterion)
            output = output_dict['output']

            loss = output_dict['loss']
            loss = loss.mean()

            batch_size = target.size(0)
            target = target[:].cpu().detach().numpy().astype('uint8')
            _, pred = output.data.topk(1, 1, True, True)
            pred = pred.squeeze().cpu().detach().numpy()
            report = classification_report(target, pred, output_dict=True, zero_division=0)

            # Update average meters
            lossesv.update(loss.data.item(), 1)
            accv.update(report["accuracy"], batch_size)
            sensitivityv.update(report["1"]["recall"], report["1"]["support"])
            specificityv.update(report["0"]["recall"], report["0"]["support"])
            f1scorev.update(report["1"]["f1-score"], batch_size)
            n_posv.update(report["1"]["support"])

            # Update EDL average meters
            if args.loss_fn == 'EDL':
                ev_succ_np, ev_fail_np, u_succ_np, u_fail_np = get_edl_diagnostics(pred, target,
                                                                                   output_dict[
                                                                                       'evidence'].cpu().detach().numpy(),
                                                                                   output_dict[
                                                                                       'uncertainty'].cpu().detach().numpy())
                ev_succv.update(ev_succ_np.mean(), ev_succ_np.shape[0])
                ev_failv.update(ev_fail_np.mean(), ev_fail_np.shape[0])
                u_succv.update(u_succ_np.mean(), u_succ_np.shape[0])
                u_failv.update(u_fail_np.mean(), u_fail_np.shape[0])

            # Update time and reset
            batch_timev.update(time.time() - end)
            end = time.time()

            # Logging
            if (i % args.print_freq == 0 or (i == len(val_loader) - 1)) and logiters is None:

                print_string = 'Test: [{0}/{1}]\t Time: {batch_time.avg:.3f}\t Loss: {loss.val:.8f} ({loss.avg: .8f})\t' \
                               'acc: {acc.val:.8f} ({acc.avg:.5f}) sens: {sens.val:.5f} ({sens.avg:.5f}) spec: {spec.val:.5f}' \
                               '({spec.avg:.5f}) f1: {f1s.val:.5f} ({f1s.avg:.5f}) ' \
                    .format(i + 1, len(val_loader), batch_time=batch_timev, loss=lossesv, acc=accv,
                            sens=sensitivityv, spec=specificityv, f1s=f1scorev)
                print(print_string)

                # Write to log file
                with open(results_folder + args.name + '.txt', 'a+') as log_file:
                    log_file.write(print_string + '\n')

            elif logiters is not None:
                if i > logiters:
                    break

            del (output_dict)

        # --- Done all batches ---

        # Collect metrics
        return_dict = {'val_acc': accv.avg,
                       'val_loss': lossesv.avg,
                       'val_n_pos': n_posv.avg,
                       'val_f1_score': f1scorev.avg,
                       'val_specificity': specificityv.avg,
                       'val_sensitivity': sensitivityv.avg}
        if args.loss_fn == 'EDL':
            return_dict.update({'val_ev_succ': ev_succv.avg, 'val_ev_fail': ev_failv.avg, 'val_u_succ': u_succv.avg,
                                'val_u_fail': u_failv.avg})

        # WandB logging
        if args.wandb:
            # Make images
            stimuli = []
            captions = []
            for stimulus_idx in range(imgs.shape[0]):
                example = val_loader.dataset.tensor_to_image(imgs[stimulus_idx])
                captions.append('gt {}, pred {}'.format(target[stimulus_idx].item(), pred[stimulus_idx].item()))
                stimuli.append(example)
            wandb.log({"examples": [wandb.Image(stimuli[s], caption=captions[s]) for s in range(len(stimuli))]},
                      step=global_step - 1)
            wandb.log(return_dict, step=global_step - 1)

        model.train()
        return return_dict


# Training loop
# ======================================================================================================================

val_log_dict = defaultdict(list)
train_log_dict = {'loss': [], 'acc': [], 'sensitivity': [], 'specificity': [], 'f1score': [], 'jvpen': [], 'num_pos': []}
if args.loss_fn == "EDL":
    train_log_dict.update({'ev_success': [], 'ev_fail': [], 'u_success': [], 'u_fail': []})

for epoch in range(args.start_epoch, args.epochs):
    batch_time = AverageMeter()  # how long to evaluate the batch
    data_time = AverageMeter()  # time to load data
    losses = AverageMeter()  # loss
    acc = AverageMeter()  # accuracy
    sensitivity = AverageMeter()  # true positive rate
    specificity = AverageMeter()  # true negative rate
    f1score = AverageMeter()  # f1 score
    n_pos = AverageMeter()  # number of positive (1) samples in batch
    ev_succ = AverageMeter()  # EDL evidence for accurate predictions
    ev_fail = AverageMeter()  # EDL evidence for inaccurate predictions
    u_succ = AverageMeter()  # EDL uncertainty for accurate predictions
    u_fail = AverageMeter()  # EDL uncertainty for inaccurate predictions

    model.train()

    time_since_last = time.time()
    end = time.perf_counter()

    for i, batch in enumerate(train_loader):

        data_time.update(time.perf_counter() - end)

        if args.loss_fn == 'EDL':
            # the annealing_coef (rho in the paper) will eventually be float(global_step)/annealing_step
            # annealing_coef is importance of ensuring a uniform belief mass for wrong classes increases over training
            # iterations. More details in original EDL paper by Sensoy et al. (annealing_coef is called lambda there)
            criterion = partial(edl, global_step=global_step,
                                annealing_step=args.annealing_step * len(train_loader))

        imgs = batch["image"].to(device)
        target = batch["label"].long()
        target = target.to(device)
        output_dict = model.forward(imgs, epoch, i, target, criterion)
        output = output_dict['output']

        # Backward pass
        loss = output_dict['loss']
        loss = loss.mean()

        jv_penalty = output_dict['jv_penalty']
        jv_penalty = jv_penalty.mean()
        train_log_dict['jvpen'].append(jv_penalty.item())

        if jacobian_penalty:
            loss = loss + args.penalty_gamma * jv_penalty

        loss.backward()

        # Get performance metrics
        batch_size = target.size(0)
        target = target[:].cpu().detach().numpy().astype('uint8')
        _, pred = output.data.topk(1, 1, True, True)
        pred = pred.squeeze().cpu().detach().numpy()
        report = classification_report(target, pred, output_dict=True, zero_division=0)

        # Update average meters
        losses.update(loss.data.item(), 1)
        acc.update(report["accuracy"], batch_size)
        sensitivity.update(report["1"]["recall"], report["1"]["support"])
        specificity.update(report["0"]["recall"], report["0"]["support"])
        f1score.update(report["1"]["f1-score"], batch_size)  # [LG] is this the correct n?
        n_pos.update(report["1"]["support"])

        # Update EDL average meters
        if args.loss_fn == 'EDL':
            ev_succ_np, ev_fail_np, u_succ_np, u_fail_np = get_edl_diagnostics(pred, target,
                                                                   output_dict['evidence'].cpu().detach().numpy(),
                                                                   output_dict['uncertainty'].cpu().detach().numpy())
            ev_succ.update(ev_succ_np.mean(), ev_succ_np.shape[0])
            ev_fail.update(ev_fail_np.mean(), ev_fail_np.shape[0])
            u_succ.update(u_succ_np.mean(), u_succ_np.shape[0])
            u_fail.update(u_fail_np.mean(), u_fail_np.shape[0])

        # Make optimization step
        optimizer.step()
        optimizer.zero_grad()

        # Update time and reset
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        # Logging
        if exp_logging and i % 200 == 0:

            val_res = validate(val_loader, model, criterion, epoch=epoch, logiters=3)
            print('val accuracy', val_res['val_acc'])
            print(val_res)
            for k, v in val_res.items():
                print(k,v)
                val_log_dict[k].extend([v])

        if global_step % (args.print_freq) == 0:
            time_now = time.time()
            print_string = 'Epoch: [{0}][{1}/{2}] t: {3} lr: {lr:g} Time: {batch_time.val:.3f} (itavg:{timeiteravg:.3f}) ' \
                           '({batch_time.avg:.3f})  Data: {data_time.val:.3f} ({data_time.avg:.3f}) ' \
                           'Loss: {loss.val:.8f} ({lossprint:.8f}) ({loss.avg:.8f})  acc: {acc.val:.5f} ' \
                           '({acc.avg:.5f}) sens: {sens.val:.5f} ({sens.avg:.5f}) spec: {spec.val:.5f} ' \
                           '({spec.avg:.5f})  f1: {f1s.val:.5f} ({f1s.avg:.5f}) ' \
                           'jvpen: {jpena:.12f} {timeprint:.3f}' \
                .format(epoch, i + 1, len(train_loader), args.timesteps, batch_time=batch_time,
                        data_time=data_time, loss=losses,
                        lossprint=mean(losses.history[-args.print_freq:]), lr=optimizer.param_groups[0]['lr'],
                        acc=acc, timeiteravg=mean(batch_time.history[-args.print_freq:]),
                        timeprint=time_now - time_since_last, sens=sensitivity, spec=specificity,
                        f1s=f1score, jpena=jv_penalty.item())
            print(print_string)
            with open(results_folder + args.name + '.txt', 'a+') as log_file:
                log_file.write(print_string + '\n')

            # WandB logging
            if args.wandb:
                wandb_dict = {
                    "train_loss": loss.data.item(),
                    "train_jv_penalty": jv_penalty.item(),
                    "train_acc": acc.val,
                    "train_sensitivity": sensitivity.val,
                    "train_specificity": specificity.val,
                    "train_f1score": f1score.val,
                    "train_num_pos": n_pos.val,
                    "epoch": epoch,
                    "batch": i,
                    "timesteps": args.timesteps
                }
                if args.loss_fn == 'EDL':
                    wandb_dict.update({
                        'train_ev_succ': ev_succ.val,
                        'train_ev_fail': ev_fail.val,
                        'train_u_succ': u_succ.val,
                        'train_u_fail': u_fail.val
                    })
                wandb_dict.update(
                    {k: v for k, v in optimizer.state_dict()['param_groups'][0].items() if k is not 'params'})
                wandb.log(wandb_dict, step=global_step)

            del output_dict
            time_since_last = time_now

        global_step += 1

    # --- Done all batches ---

    # Update lr
    if args.adjust_lr:
        lr_scheduler.step()

    # Logging
    train_log_dict['loss'].extend(losses.history)
    train_log_dict['acc'].extend(acc.history)
    train_log_dict['sensitivity'].extend(sensitivity.history)
    train_log_dict['specificity'].extend(specificity.history)
    train_log_dict['f1score'].extend(f1score.history)
    train_log_dict['num_pos'].extend(n_pos.history)
    if args.loss_fn == 'EDL':
        train_log_dict['ev_success'].extend(ev_succ.history)
        train_log_dict['ev_fail'].extend(ev_fail.history)
        train_log_dict['u_success'].extend(u_succ.history)
        train_log_dict['u_fail'].extend(u_fail.history)

    save_npz(epoch, train_log_dict, results_folder, 'train')
    save_npz(epoch, val_log_dict, results_folder, 'val')

    if (epoch + 1) % 1 == 0 or epoch == args.epochs - 1:
        if hasattr(model, 'timesteps'):
            model.timesteps = args.timesteps
        val_res = validate(val_loader, model, criterion, epoch=epoch)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'acc': val_res['val_acc']}, 'acc', results_folder)
