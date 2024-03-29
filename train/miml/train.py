import sys
sys.path.append('/home/lkk/code/ImageCaption')
from tensorboardX import SummaryWriter
from collections import OrderedDict
import os
import json
from nltk.translate.bleu_score import corpus_bleu
from utils.utils import AverageMeter, accuracy, adjust_learning_rate, clip_gradient, save_checkpoint_miml
from utils.data import CaptionDataset
from src.miml.model import MIML, Decoder
from torch.nn.utils.rnn import pack_padded_sequence
from torch import nn
import torchvision.transforms as transforms
import torch.utils.data
import torch.optim
import torch.backends.cudnn as cudnn
import time


# Data parameters
# folder with data files saved by create_input_files.py
data_folder = '/home/lkk/datasets/coco2014/'
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
prefix = 'miml1024'
# Model parameters
emb_dim = 1024  # dimension of word embeddings
attrs_dim = 1024  # dimension of attention linear layers
decoder_dim = 1024  # dimension of decoder RNN
attrs_size = 1024
dropout = 0.5
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
# set to true only if inputs to model are fixed size; otherwise lot of computational overhead
cudnn.benchmark = True

# Training parameters
start_epoch = 0
# number of epochs to train for (if early stopping is not triggered)
epochs = 30
# keeps track of number of epochs since there's been an improvement in validation BLEU
epochs_since_improvement = 0
batch_size = 128
workers = 0  # for data-loading; right now, only 1 works with h5py
miml_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 5e-3  # learning rate for decoder
pin_memory = False
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 1  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
# checkpoint = './checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'  # path to checkpoint, None if none
checkpoint = None
checkpoint_miml = '/home/lkk/code/ImageCaption/MIML.pth.tar'


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    miml = MIML(freeze=False, fine_tune=True)

    pretrained_net_dict = torch.load(
        checkpoint_miml, map_location=lambda storage, loc: storage)['model']
    new_state_dict = OrderedDict()
    for k, v in pretrained_net_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
        # load params
    miml.load_state_dict(new_state_dict)
    miml_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, miml.parameters()),
                                      lr=miml_lr, weight_decay=1e-4)

    decoder = Decoder(attrs_dim=attrs_dim, embed_dim=emb_dim,
                      decoder_dim=decoder_dim, attrs_size=attrs_size, vocab_size=len(
                          word_map), device=device,
                      dropout=dropout)

    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=decoder_lr, weight_decay=1e-4)

    if checkpoint:
        checkpoint = torch.load(
            checkpoint, map_location=lambda storage, loc: storage)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        miml.load_state_dict(checkpoint['miml'])
        decoder.load_state_dict(checkpoint['decoder'])
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])

    miml = miml.to(device)
    decoder = decoder.to(device)
    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN',
                       transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL',
                       transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory)
    writer = SummaryWriter(log_dir='./log_miml')
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 5 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 4 == 0:
            adjust_learning_rate(decoder_optimizer, 0.9)

        # One epoch's training
        train(train_loader=train_loader,
              miml=miml,
              decoder=decoder,
              criterion=criterion,
              miml_optimizer=miml_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch,
              writer=writer)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                miml=miml,
                                decoder=decoder,
                                criterion=criterion,
                                epoch=epoch,
                                writer=writer)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" %
                  (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint_miml(prefix, data_name, epoch, epochs_since_improvement, miml, decoder, miml_optimizer,
                             decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, miml, decoder, criterion, miml_optimizer, decoder_optimizer, epoch, writer):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    miml.train()
    total_step = len(train_loader)
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        attrs = miml(imgs)
        scores, caps_sorted, decode_lengths, sort_ind = decoder(
            attrs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        # torch在计算时会自动除去pad，这样不带pad计算不影响精度
        scores, _ = pack_padded_sequence(
            scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(
            targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores, targets)

        # Back prop.
        decoder_optimizer.zero_grad()
        miml_optimizer.zero_grad()

        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if miml_optimizer is not None:
                clip_gradient(miml_optimizer, grad_clip)
        # Update weights
        decoder_optimizer.step()
        miml_optimizer.step()
        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            writer.add_scalars(
                'train', {'loss': loss.item(), 'mAp': top5accs.val}, epoch*total_step+i)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, miml, decoder, criterion, epoch, writer):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    miml.eval()
    decoder.eval()  # eval mode (no dropout or batchnorm)
    total_step = len(val_loader)
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    with torch.no_grad():

        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.

            attrs = miml(imgs)
            scores, caps_sorted, decode_lengths, sort_ind = decoder(
                attrs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores, _ = pack_padded_sequence(
                scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(
                targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                writer.add_scalars(
                    'val', {'loss': loss.item(), 'mAp': top5accs.val}, epoch*total_step+i)
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            # because images were sorted in the decoder
            allcaps = allcaps[sort_ind]
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        weights = (1.0 / 1.0,)
        bleu1 = corpus_bleu(references, hypotheses, weights)

        weights = (1.0 / 2.0, 1.0 / 2.0,)
        bleu2 = corpus_bleu(references, hypotheses, weights)

        weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,)
        bleu3 = corpus_bleu(references, hypotheses, weights)
        bleu4 = corpus_bleu(references, hypotheses)
        writer.add_scalars(
            'Bleu', {'Bleu1': bleu1, 'Bleu2': bleu2, 'Bleu3': bleu3, 'Bleu4': bleu4}, epoch)
        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()
