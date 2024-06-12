import os
import csv
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from yiku.nets.loss import (CE_Loss, Dice_loss, Focal_Loss, Bootstrapped_CELoss)
from yiku.nets.training_utils import weights_init, set_optimizer_lr
# from tqdm import tqdm
from collections import OrderedDict
from yiku.utils.utils import get_lr
from yiku.utils.utils_metrics import f_score

try:
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn, track,
        TimeElapsedColumn,
        TimeRemainingColumn)
    from rich import print
    from rich.console import Console
    from rich.table import Table
except ImportError:
    import warnings

    warnings.filterwarnings('ignore', message="Setuptools is replacing distutils.", category=UserWarning)
    from pip._vendor.rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        SpinnerColumn, track,
        TaskProgressColumn,
        TimeElapsedColumn,
        TimeRemainingColumn
    )
    from pip._vendor.rich import print
    from pip._vendor.rich.table import Table
    from pip._vendor.rich.console import Console


class SegmentationMetric(object):
    def __init__(self, numClass, cuda=True):
        self.numClass = numClass
        self._cuda = cuda
        if self._cuda:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.confusionMatrix = torch.zeros((self.numClass,) * 2, device=self.device)  # 混淆矩阵（空）

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self) -> torch.Tensor:
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = classAcc[classAcc < float('inf')].mean()  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        if self._cuda:
            return meanAcc.cpu()
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = torch.sum(self.confusionMatrix, axis=1) + torch.sum(self.confusionMatrix, axis=0) - torch.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self):
        IoU = self.IntersectionOverUnion()
        mIoU = IoU[IoU < float('inf')].mean()  # 求各类别IoU的平均
        if self._cuda:
            return mIoU.cpu()
        else:
            return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel, ignore_labels):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        for IgLabel in ignore_labels:
            mask &= (imgLabel != IgLabel)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = torch.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.view(self.numClass, self.numClass)
        confusionMatrix = confusionMatrix.to(self.device)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU，频权交并比:为MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = torch.sum(self.confusion_matrix, axis=1) / torch.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                torch.sum(self.confusion_matrix, axis=1) + torch.sum(self.confusion_matrix, axis=0) -
                torch.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        if self._cuda:
            return FWIoU.cpu()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel, ignore_labels=[255]):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel, ignore_labels)
        # 得到混淆矩阵
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass), device=self.device)


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, \
                  fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss = 0
    total_f_score = 0

    val_loss = 0
    val_f_score = 0

    if local_rank == 0:
        rich_pbar = Progress(SpinnerColumn(),
                             "🐱", "{task.description}",
                             BarColumn(),
                             TaskProgressColumn(),
                             TimeElapsedColumn(),
                             "[bold]GPU mem:", '[bold]{task.fields[gmem]:.3g}',
                             TimeRemainingColumn(), '📈', '[bold orange1]total_loss:',
                             '[bold]{task.fields[total_loss]:.3f}',
                             "  ", '[bold dark_magenta]f_score:',
                             '[bold]{task.fields[f_score]:.3f}', " ",
                             "[bold dodger_blue2]lr:",
                             "[bold]{task.fields[lr]:.6f}")
        task1 = rich_pbar.add_task(f'[orange1]training epoch {epoch + 1}/{Epoch}', total=len(gen),
                                   total_loss=float('nan'), f_score=float('nan'), lr=float('nan'), gmem=0)
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
        optimizer.zero_grad(set_to_none=True)
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
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0

        if local_rank == 0:
            rich_pbar.update(task1, total_loss=total_loss / (iteration + 1),
                             f_score=total_f_score / (iteration + 1),
                             lr=get_lr(optimizer), advance=1, gmem=mem)

    model_train.eval()
    if local_rank == 0:
        rich_pbar.stop()
        del rich_pbar
        rich_pbar = Progress(SpinnerColumn(),
                             "🌕", "{task.description}",
                             BarColumn(),
                             TaskProgressColumn(),
                             TimeElapsedColumn(),
                             TimeRemainingColumn(), '📈', '[bold pink1]val_loss:',
                             '[bold]{task.fields[val_loss]:.3f}',
                             "  ", '[bold green4]f_score:',
                             '[bold]{task.fields[f_score]:.3f}', " ",
                             "[bold dodger_blue1]mIoU:",
                             "[bold]{task.fields[miou]:.6f}"
                             )

        task1 = rich_pbar.add_task(f'[pink1]val epoch {epoch + 1}/{Epoch}', total=len(gen_val), val_loss=float('nan'),
                                   f_score=float('nan'), miou=float('nan'))
        rich_pbar.start()

    metrics = SegmentationMetric(numClass=num_classes)
    mious = []
    mpas = []
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
            if local_rank == 0:
                pred = F.softmax(outputs.permute(0, 2, 3, 1), dim=-1)
                pred = torch.argmax(pred, -1)
                metrics.addBatch(pred, pngs)
                mIoU = metrics.meanIntersectionOverUnion()
                mpa = metrics.meanPixelAccuracy()
                mious.append(mIoU)
                mpas.append(mpa)
            # metrics.reset()
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
                                 f_score=val_f_score / (iteration + 1), miou=mIoU, advance=1)

    if local_rank == 0:
        rich_pbar.stop()
        miou = mious[-1]
        mpa = mpas[-1]
        flag = len(loss_history.val_loss) <= 1 or miou > max(loss_history.miou)
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val, miou)

        table = Table(title="📜 EpochEvalResult世代评测结果")
        table.add_column("参数Arg", justify="right", style="bold pink1", no_wrap=True)
        table.add_column("值Value", style="orange1")
        table.add_row("epoch", str(epoch + 1) + '/' + str(Epoch))
        table.add_row("Total Loss", "%.3f " % (total_loss / epoch_step))
        table.add_row("Val Loss", "%.3f " % (val_loss / epoch_step_val))
        table.add_row("mIoU", "%.3f " % (miou * 100) + "%")
        table.add_row("mPA", "%.3f " % (mpa * 100) + "%")
        console = Console()
        console.print(table)
        with open(Path(save_dir) / "logs.csv", 'a+') as f:
            csv_write = csv.writer(f)
            data_row = [epoch + 1, "%.3f" % (total_loss / epoch_step), "%.3f" % (val_loss / epoch_step_val)]
            data_row.append("%.3f" % (miou * 100))
            data_row.append("%.3f" % (mpa * 100))
            csv_write.writerow(data_row)
        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        meta = OrderedDict()
        meta["val_his_loss"] = loss_history.val_loss
        meta["his_loss"] = loss_history.losses
        meta["miou"] = loss_history.miou
        meta["curr_epoch"] = epoch
        meta["epoch_step_val"] = epoch_step_val
        meta["curr_val_loss"] = val_loss
        if flag:
            print('⚾ [green1] Save best model to %s' % str(Path(save_dir) / "best.pth"))
            with open(Path(save_dir) / "best.pth", mode="wb") as f:
                torch.save(model.state_dict(), f)
            with open(Path(save_dir) / "best.meta", mode="wb") as f:
                torch.save(meta, f)
        with open(Path(save_dir) / "last.pth", mode="wb") as f:
            torch.save(model.state_dict(), f)
        with open(Path(save_dir) / "last.meta", mode="wb") as f:
            torch.save(meta, f)
