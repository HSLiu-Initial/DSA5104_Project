# -*- coding:utf-8 -*-

import logging
import numpy as np
import os
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from tqdm import tqdm

from codes.utils.metrics import evaluate_auc, evaluate_logloss


def train_validate_din(args, model_save_dir, model, criterion, optimizer, decay, train_loader, train_writer,
                        test_loader,
                        eval_writer, scaler=None):
    model = model.cuda()
    criterion = criterion.cuda()
    logging.info("Start training.")
    best_auc = 0.0
    best_val_auc_gap = 1
    kill_cnt = 0
    start_epoch = 0
    if args.resume is True:
        ckpt = torch.load(os.path.join(model_save_dir, 'state.pth'))
        para_dic = torch.load(os.path.join(model_save_dir, 'model.pth'))
        if args.use_ddp:
            model.module.load_state_dict(para_dic)
        else:
            model.load_state_dict(para_dic)
        start_epoch = ckpt['epoch']
        optimizer.load_state_dict(ckpt['optimizer'])
        decay.load_state_dict(ckpt['scheduler'])

    for epoch in range(start_epoch, args.epochs):
        # Training and validation
        if args.use_ddp is True:
            train_loader.sampler.set_epoch(epoch)
        model.train()
        train_loss = []
        train_auc = []
        with tqdm(total=len(train_loader), dynamic_ncols=True) as t:
            for step, batch in enumerate(train_loader):
                target_item = batch['target_item'].cuda()
                hist_item = batch['hist_item'].cuda()
                hist_valid_lens = batch['hist_valid_lens'].cuda()
                sparse_feature = batch['sparse_feature'].cuda()
                y = batch['y'].cuda().unsqueeze(1)
                if args.use_mix_precision:
                    with autocast(enabled=args.use_mix_precision):  #
                        logits = model(target_item, hist_item, hist_valid_lens, sparse_feature)
                        # compute loss
                        logit_loss = criterion(logits, y.float())
                else:
                    logits = model(target_item, hist_item, hist_valid_lens, sparse_feature)
                    # compute loss
                    logit_loss = criterion(logits, y.float())
                tr_loss = logit_loss
                train_loss.append(tr_loss.item())
                tr_auc = evaluate_auc(torch.sigmoid(logits).detach().cpu().numpy(),
                                      y.detach().cpu().numpy())
                train_auc.append(tr_auc)
                # train_auc.append(tr_auc)

                # backward
                optimizer.zero_grad()

                if args.use_mix_precision:
                    scaler.scale(tr_loss).backward()  #
                    scaler.step(optimizer)  #
                    scaler.update()  #
                else:
                    tr_loss.backward()
                    optimizer.step()

                train_iter = epoch * len(train_loader) + step
                if args.decay_cycle == 'iter':
                    decay.step(train_iter / len(train_loader))

                t.update()
                if (args.use_ddp is True and args.rank == 0) or args.use_ddp is False:
                    t.set_description(desc=f'Epoch: {epoch}/{args.epochs}')
                    t.set_postfix({
                        'logit_loss': f'{logit_loss.item():.4f}',
                        'train_loss': f'{tr_loss.item():.4f}',
                        'train_auc': f'{tr_auc.item():.4f}'
                    })

                if step % 10 == 0 and ((args.use_ddp is True and args.rank == 0) or args.use_ddp is False):
                    train_writer.add_scalar("logit_loss", logit_loss.item(), train_iter)
                    train_writer.add_scalar("train_loss", tr_loss.item(), train_iter)
                    train_writer.add_scalar("train_auc", tr_auc.item(), train_iter)

                    for name, param in model.named_parameters():  #
                        train_writer.add_histogram(name + '_grad', param.grad, epoch)
                        train_writer.add_histogram(name + '_data', param, epoch)
                if step % 100 == 0 and ((args.use_ddp is True and args.rank == 0) or args.use_ddp is False):
                    logging.info(
                        "Epoch {}, step {}/{}, train loss: {:.4f}\n".format(epoch, step, len(train_loader),
                                                                            tr_loss.item()))

        train_loss = np.mean(train_loss)
        train_auc = np.mean(train_auc)
        # print("Epoch {}, train loss: {}, train auc: {}\n".format(epoch, train_loss, train_auc))
        if (args.use_ddp is True and args.rank == 0) or args.use_ddp is False:
            logging.info("Epoch {}, train loss: {}, train auc: {}\n".format(epoch, train_loss, train_auc))
        if args.decay_cycle == 'epoch':
            decay.step()
        model.eval()
        with torch.no_grad():
            validate_loss = []
            validate_auc = []
            validate_logit_loss = []
            with tqdm(total=len(test_loader), dynamic_ncols=True) as t:
                for step, batch in enumerate(test_loader):
                    target_item = batch['target_item'].cuda()
                    hist_item = batch['hist_item'].cuda()
                    hist_valid_lens = batch['hist_valid_lens'].cuda()
                    sparse_feature = batch['sparse_feature'].cuda()
                    y = batch['y'].cuda().unsqueeze(1)
                    if args.use_mix_precision:
                        with autocast(enabled=args.use_mix_precision):  #
                            logits = model(target_item, hist_item, hist_valid_lens, sparse_feature)
                            # compute loss
                            val_logit_loss = criterion(logits, y.float())
                    else:
                        logits = model(target_item, hist_item, hist_valid_lens, sparse_feature)
                        # compute loss
                        val_logit_loss = criterion(logits, y.float())
                    val_loss = val_logit_loss
                    val_auc = evaluate_auc(torch.sigmoid(logits).detach().cpu().numpy(),
                                           y.detach().cpu().numpy())
                    validate_logit_loss.append(val_logit_loss.item())
                    validate_loss.append(val_loss.item())
                    validate_auc.append(val_auc)

                    t.update()
                    t.set_description(desc=f'Epoch: {epoch}/{args.epochs}')
                    t.set_postfix({
                        'logit_loss': f'{val_logit_loss.item():.4f}',
                        'valid loss': f'{val_loss.item():.4f}',
                        'valid auc': f'{val_auc.item():.4f}'
                    })
            validate_logit_loss = np.mean(validate_logit_loss) if not args.use_ddp else torch.tensor(
                np.mean(validate_logit_loss) / args.world_size, device=val_auc.device)
            validate_loss = np.mean(validate_loss) if not args.use_ddp else torch.tensor(
                np.mean(validate_loss) / args.world_size, device=val_auc.device)
            validate_auc = np.mean(validate_auc) if not args.use_ddp else torch.tensor(
                np.mean(validate_auc) / args.world_size, device=val_auc.device)
            val_auc_gap = abs(validate_auc - args.baseline_auc)
            if best_val_auc_gap > val_auc_gap:
                best_val_auc_gap = val_auc_gap
            if args.use_ddp:
                dist.reduce(validate_logit_loss, 0, op=dist.ReduceOp.SUM)  #
                dist.reduce(validate_loss, 0, op=dist.ReduceOp.SUM)
                dist.reduce(validate_auc, 0, op=dist.ReduceOp.SUM)
        if (args.use_ddp is True and args.rank == 0) or args.use_ddp is False:
            eval_writer.add_scalar("eval_logit_loss", validate_logit_loss, train_iter)
            eval_writer.add_scalar("eval_loss", validate_loss, train_iter)
            eval_writer.add_scalar("eval_auc", validate_auc, train_iter)

            # validate
            if validate_auc > best_auc:
                best_auc = validate_auc
                best_epoch = epoch
                if args.use_ddp:
                    state = {'epoch': epoch,
                             'model': model.module.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'scheduler': decay.state_dict()}

                    torch.save(
                        state, os.path.join(model_save_dir, 'state.pth')
                    )
                    torch.save(
                        model.module.state_dict(), os.path.join(model_save_dir, 'model.pth')
                    )
                    logging.info("saving model...")
                else:
                    state = {'epoch': epoch,
                             'model': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'scheduler': decay.state_dict()}

                    torch.save(
                        state, os.path.join(model_save_dir, 'state.pth')
                    )
                    torch.save(
                        model.state_dict(), os.path.join(model_save_dir, 'model.pth')
                    )
                    logging.info("saving model...\n")
                kill_cnt = 0
            else:
                kill_cnt += 1
                if kill_cnt > args.early_stop:
                    if (args.use_ddp is True and args.rank == 0) or args.use_ddp is False:
                        logging.info("early stop.")
                        logging.info("best epoch:{}".format(best_epoch))
                        break
            if (args.use_ddp is True and args.rank == 0) or args.use_ddp is False:
                logging.info((
                    "In epoch {}, Train Loss: {:.4f}, Train AUC: {:.4f}, Valid Loss: {:.5}, Valid AUC: {:.5}\n".format(
                        epoch, train_loss, train_auc, validate_loss, validate_auc
                    )
                ))


def test_din(model, model_save_dir, test_loader, criterion, args):
    model.eval()
    with torch.no_grad():
        if args.use_ddp is True:
            model.module.load_state_dict(torch.load(os.path.join(model_save_dir, 'model.pth'))['model'])
        else:
            model.load_state_dict(torch.load(os.path.join(model_save_dir, 'model.pth')))

        test_loss = []
        test_auc = []
        test_logloss = []
        with tqdm(total=len(test_loader), dynamic_ncols=True) as t:
            for step, batch in enumerate(test_loader):
                target_item = batch['target_item'].cuda()
                hist_item = batch['hist_item'].cuda()
                hist_valid_lens = batch['hist_valid_lens'].cuda()
                sparse_feature = batch['sparse_feature'].cuda()
                y = batch['y'].cuda().unsqueeze(1)
                if args.use_mix_precision:
                    with autocast(enabled=args.use_mix_precision):  #
                        logits = model(target_item, hist_item, hist_valid_lens, sparse_feature)
                        # compute loss
                        logit_loss = criterion(logits, y.float())
                else:
                    logits = model(target_item, hist_item, hist_valid_lens, sparse_feature)
                    # compute loss
                    logit_loss = criterion(logits, y.float())
                loss = logit_loss
                auc = evaluate_auc(torch.sigmoid(logits).detach().cpu().numpy(), y.detach().cpu().numpy())
                log_loss = evaluate_logloss(torch.sigmoid(logits).detach().cpu().numpy(),
                                            y.detach().cpu().numpy())

                test_loss.append(loss.item())
                test_auc.append(auc)
                test_logloss.append(log_loss)

        test_loss = np.mean(test_loss) if not args.use_ddp else torch.tensor(
            np.mean(test_loss), device=loss.device)
        test_auc = np.mean(test_auc) if not args.use_ddp else torch.tensor(
            np.mean(test_auc) / args.world_size, device=loss.device)
        test_logloss = np.mean(test_logloss) if not args.use_ddp else torch.tensor(
            np.mean(test_logloss) / args.world_size, device=loss.device)

        if args.use_ddp:
            dist.reduce(test_loss, 0, op=dist.ReduceOp.SUM)  #
            dist.reduce(test_auc, 0, op=dist.ReduceOp.SUM)  #
            dist.reduce(test_logloss, 0, op=dist.ReduceOp.SUM)  #
        if (args.use_ddp is True and args.rank == 0) or args.use_ddp is False:
            logging.info(
                "Test Loss: {:.5}, AUC: {:.5}, Logloss: {:.5}".format(
                    test_loss, test_auc, test_logloss
                )
            )
        if args.use_ddp:
            dist.destroy_process_group()
