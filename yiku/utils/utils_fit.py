import os
import csv
import torch
from pathlib import Path
from nets.deeplabv3_training import (CE_Loss, Dice_loss, Focal_Loss,
                                     weights_init)
#from tqdm import tqdm
from collections import OrderedDict
from utils.utils import get_lr
from utils.utils_metrics import f_score

try:
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,track,
        TimeElapsedColumn,
        TimeRemainingColumn)
    from rich import print
except ImportError:
    import warnings

    warnings.filterwarnings('ignore', message="Setuptools is replacing distutils.", category=UserWarning)
    from pip._vendor.rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        SpinnerColumn,track,
        TaskProgressColumn,
        TimeElapsedColumn,
        TimeRemainingColumn
    )
    from pip._vendor.rich import print


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, \
                  fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss = 0
    total_f_score = 0

    val_loss = 0
    val_f_score = 0

    if local_rank == 0:
        # pbar = tqdm(total=epoch_step, desc=f'epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3, leave=False,
        #             ncols=150)
        rich_pbar = Progress(SpinnerColumn(),
                             "🐱","{task.description}",
                             BarColumn(),
                             TaskProgressColumn(),
                             TimeElapsedColumn(),
                             TimeRemainingColumn(), '📈','[bold orange1]total_loss:',
                             '[bold]{task.fields[total_loss]:.3f}',
                             "  ", '[bold dark_magenta]f_score:',
                             '[bold]{task.fields[f_score]:.3f}', " ",
                             "[bold dodger_blue2]lr:",
                             "[bold]{task.fields[lr]:.6f}")
        task1 = rich_pbar.add_task(f'[orange1]training epoch {epoch + 1}/{Epoch}', total=len(gen), total_loss=float('nan'), f_score=float('nan'), lr=float('nan'))
        rich_pbar.start()

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()
        if not fp16:
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(imgs)
            # ----------------------#
            #   计算损失
            # ----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(outputs, labels)

            # ----------------------#
            #   反向传播
            # ----------------------#
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                # ----------------------#
                #   前向传播
                # ----------------------#
                outputs = model_train(imgs)
                # ----------------------#
                #   计算损失
                # ----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

                with torch.no_grad():
                    # -------------------------------#
                    #   计算f_score
                    # -------------------------------#
                    _f_score = f_score(outputs, labels)

            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            rich_pbar.update(task1, total_loss=total_loss / (iteration + 1),
                             f_score=total_f_score / (iteration + 1),
                             lr=get_lr(optimizer), advance=1)
            # pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
            #                     'f_score': total_f_score / (iteration + 1),
            #                     'lr': get_lr(optimizer)})
            # pbar.update(1)

    model_train.eval()
    if local_rank == 0:
        rich_pbar.stop()
        # pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
        del rich_pbar
        rich_pbar = Progress(SpinnerColumn(),
                             "🌕","{task.description}",
                             BarColumn(),
                             TaskProgressColumn(),
                             TimeElapsedColumn(),
                             TimeRemainingColumn(), '📈','[bold pink1]val_loss:',
                             '[bold]{task.fields[val_loss]:.3f}',
                             "  ", '[bold green4]f_score:',
                             '[bold]{task.fields[f_score]:.3f}', " ",
                             "[bold dodger_blue1]lr:",
                             "[bold]{task.fields[lr]:.6f}")

        task1 = rich_pbar.add_task(f'[pink1]val epoch {epoch + 1}/{Epoch}', total=len(gen_val), val_loss=float('nan'),
                                   f_score=float('nan'), lr=float('nan'))
        rich_pbar.start()


    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(imgs)
            # ----------------------#
            #   计算损失
            # ----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice
            # -------------------------------#
            #   计算f_score
            # -------------------------------#
            _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()

            if local_rank == 0:
                rich_pbar.update(task1, val_loss=val_loss / (iteration + 1),
                                 f_score=val_f_score / (iteration + 1),
                                 lr=get_lr(optimizer), advance=1)
                # pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                #                     'f_score': val_f_score / (iteration + 1),
                #                     'lr': get_lr(optimizer)})
                # pbar.update(1)

    if local_rank == 0:
        rich_pbar.stop()
        # pbar.close()
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('📜 [bold]Epoch:' + str(epoch + 1) + '/' + str(Epoch)+
              ': Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        with open(Path(save_dir)/ "logs.csv", 'a+') as f:
            csv_write = csv.writer(f)
            data_row = [epoch + 1, "%.3f" % (total_loss / epoch_step), "%.3f" % (val_loss / epoch_step_val)]
            csv_write.writerow(data_row)
        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        meta = OrderedDict()
        meta["val_his_loss"] = loss_history.val_loss
        meta["his_loss"] = loss_history.losses
        meta["curr_epoch"] = epoch
        meta["epoch_step_val"] = epoch_step_val
        meta["curr_val_loss"] = val_loss
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), Path(save_dir)/('ep%03d-loss%.3f-val_loss%.3f.pth' % (
                epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('⚾ [green1] Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), Path(save_dir)/ "best_epoch_weights.pth")
            torch.save(meta, Path(save_dir)/"best.meta")
        torch.save(model.state_dict(), Path(save_dir)/ "last_epoch_weights.pth")
        torch.save(meta, Path(save_dir)/"last.meta")
