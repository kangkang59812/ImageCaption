import os
import numpy as np
import h5py
import json
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
from matplotlib import cm
import os.path as osp
import torch


def save_checkpoint_fbasemodel(prefix, epoch, epochs_since_improvement, decoder, decoder_optimizer,
                               bleu4, is_best):

    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'decoder': decoder.state_dict(),
             'decoder_optimizer': decoder_optimizer.state_dict()}
    filename = prefix + '_checkpoint' + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


def save_checkpoint_basewithmiml4(path, prefix, epoch, encoder, decoder, optimizer, scheduler,
                                  bleu4, is_best):

    if not os.path.exists(path):
        os.makedirs(path)

    state = {'epoch': epoch,
             'bleu-4': bleu4,
             'encoder': encoder.state_dict(),
             'decoder': decoder.state_dict(),
             'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict()}
    filename = os.path.join('/home/lkk/code/MIML/', path,
                            prefix + '_checkpoint_' + str(epoch) + '.pth.tar')
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        filename = os.path.join('/home/lkk/code/MIML/', path,
                                prefix + '_BEST_checkpoint_' + str(epoch) + '.pth.tar')
        torch.save(state, filename)


def save_checkpoint_basewithmiml(prefix, epoch, epochs_since_improvement, miml, encoder, decoder, encoder_optimizer, decoder_optimizer,
                                 bleu4, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'miml': miml.state_dict(),
             'encoder': encoder.state_dict(),
             'decoder': decoder.state_dict(),
             'encoder_optimizer': encoder_optimizer.state_dict(),
             'decoder_optimizer': decoder_optimizer.state_dict()}
    filename = prefix + '_checkpoint_' + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


def save_checkpoint_miml(prefix, epoch, epochs_since_improvement, miml, decoder, decoder_optimizer, miml_optimizer,
                         bleu4, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'miml': miml.state_dict(),
             'miml_optimizer': miml_optimizer.state_dict(),
             'decoder': decoder.state_dict(),
             'decoder_optimizer': decoder_optimizer.state_dict()}
    filename = prefix + '_checkpoint_' + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


def save_checkpoint_basemodel(prefix, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,  decoder_optimizer,
                              bleu4, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder.state_dict(),
             'decoder': decoder.state_dict(),
             'encoder_optimizer': encoder_optimizer.state_dict(),
             'decoder_optimizer': decoder_optimizer.state_dict()}
    filename = prefix + '_checkpoint_' + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def plot_instance_attention(im, instance_points, instance_labels, save_path=None):
    """
    Arguments:
        im (ndarray): shape = (3, im_width, im_height)
            the image array
        instance_points: List of (x, y) pairs
            the instance's center points
        instance_labels: List of str
            the label name of each instance
    """
    fig, (ax, ax2) = plt.subplots(1, 2)
    ax.imshow(im)
    ax2.imshow(im)
    for i, (x_center, y_center) in enumerate(instance_points):
        label = instance_labels[i]
        center = plt.Circle((x_center, y_center), 10, color="r", alpha=0.5)
        ax.add_artist(center)
        ax.text(x_center, y_center, str(label), fontsize=18,
                bbox=dict(facecolor="blue", alpha=0.7), color="white")
    if save_path:
        if not osp.exists(osp.dirname(save_path)):
            os.makedirs(osp.dirname(save_path))
        fig.savefig(save_path)
    else:
        plt.show()


def plot_instance_probs_heatmap(instance_probs, save_path=None):
    """
    Arguments:
        instance_probs (ndarray): shape = (n_instances, n_labels)
            the probability distribution of each instance
    """
    n_instances, n_labels = instance_probs.shape
    fig, ax = plt.subplots()
    ax.set_title("Instance-Label Scoring Layer Visualized")

    cax = ax.imshow(instance_probs, vmin=0, vmax=1, cmap=cm.hot,
                    aspect=float(n_labels) / n_instances)
    cbar_ticks = list(np.linspace(0, 1, 11))
    cbar = fig.colorbar(cax, ticks=cbar_ticks)
    cbar.ax.set_yticklabels(map(str, cbar_ticks))

    if save_path:
        if not osp.exists(osp.dirname(save_path)):
            os.makedirs(osp.dirname(save_path))
        fig.savefig(save_path)
    else:
        plt.show()
