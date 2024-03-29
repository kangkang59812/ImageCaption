import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from src.base_with_miml.model5 import Encoder, Decoder
from utils.data import CaptionDataset
from utils.utils import AverageMeter, accuracy, adjust_learning_rate, clip_gradient, save_checkpoint_basewithmiml4
from nltk.translate.bleu_score import corpus_bleu
import os
from collections import OrderedDict
from tensorboardX import SummaryWriter
import json
# Data parameters
# folder with data files saved by create_input_files.py
data_folder = '/home/lkk/datasets/flickr8k/'
# base name shared by data files
data_name = 'flickr_5_cap_per_img_5_min_word_freq'
prefix = 'base_with_miml5'
# Model parameters
emb_dim = 1024  # dimension of word embeddings
attention_dim = 1024
attrs_dim = 1024  # dimension of attention linear layers
decoder_dim = 1024  # dimension of decoder RNN
# attrs_size = 1024
dropout = 0.5
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
# set to true only if inputs to model are fixed size; otherwise lot of computational overhead
cudnn.benchmark = True

# Training parameters
start_epoch = 0
# number of epochs to train for (if early stopping is not triggered)
epochs = 20
# keeps track of number of epochs since there's been an improvement in validation BLEU
epochs_since_improvement = 0
batch_size = 16
workers = 0  # for data-loading; right now, only 1 works with h5py

encoder_lr1_1 = 2e-5  # MIML的resnet block if fine-tuning
blocks = 1
encoder_lr1_2 = 2e-4  # MIML的 concept_layer if fine-tuning

encoder_lr2 = 1e-4  # 提取特征的resnet block if fine-tuning

decoder_lr = 2e-4  # learning rate for decoder
grad_clip = 2.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 scor right now
print_freq = 1  # print training/validation stats every __ batches
fine_tune_encoder = True  # fine-tune encoder?
log_dir = './log_basewithmiml5_9_8'
# './BEST_checkpoint_allcoco_5_cap_per_img_5_min_word_freq.pth.tar'
# '/home/lkk/code/ImageCaption/model4_1_9_2/base_with_miml4_1_checkpoint_0.pth.tar'
# '/home/lkk/code/ImageCaption/model4_1_9_2/base_with_miml4_1_checkpoint_9.pth.tar'
checkpoint = None
tag_flag = True
miml_checkpoint = '/home/lkk/code/ImageCaption/checkpoint_ResNet_epoch_22.pth.tar'
save_path = '/home/lkk/code/ImageCaption/model5_9_8'


def main():
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map, miml_checkpoint, blocks, tag_flag
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    encoder = Encoder(fine_tune1=False,
                      blocks=blocks, fine_tune2=fine_tune_encoder)
    # 加载miml
    miml_checkpoint = torch.load(miml_checkpoint)
    encoder.miml_intermidate.load_state_dict(
        miml_checkpoint['intermidate'])
    encoder.miml_last.load_state_dict(miml_checkpoint['last'])
    encoder.miml_sub_concept_layer.load_state_dict(
        miml_checkpoint['sub_concept_layer'])

    del miml_checkpoint
    torch.cuda.empty_cache()

    # encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
    #                                      lr=encoder_lr) if fine_tune_encoder else None

    decoder = Decoder(attrs_dim=attrs_dim, attention_dim=attention_dim,
                      embed_dim=emb_dim,
                      decoder_dim=decoder_dim,
                      vocab_size=len(word_map),
                      device=device,
                      dropout=dropout)
    # decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
    #
    #{'params': filter(lambda p: p.requires_grad, encoder.miml_intermidate.parameters()), 'lr': encoder_lr1_1},
    #    {'params': filter(lambda p: p.requires_grad, encoder.miml_last.parameters(
    #    )), 'lr': encoder_lr1_1}
    optimizer = torch.optim.Adam(
        [{'params': encoder.features_model.parameters(), 'lr': encoder_lr2},
         {'params': decoder.parameters(), 'lr': decoder_lr}]
    )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=8)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.6)
    if checkpoint:
        checkpoint = torch.load(
            checkpoint, map_location=lambda storage, loc: storage)
        best_bleu4 = checkpoint['bleu-4']
        start_epoch = checkpoint['epoch']+1
        #epochs_since_improvement = checkpoint['epochs_since_improvement']
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        adjust_learning_rate(optimizer, 0.5)
        # scheduler.load_state_dict(checkpoint['scheduler'])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.6, last_epoch=9)
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
    else:

        encoder = encoder.to(device)
        decoder = decoder.to(device)

    criterion = nn.CrossEntropyLoss()
    # criterion2 = nn.BCELoss()
    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', tag_flag=tag_flag,
                       transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', tag_flag=tag_flag,
                       transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    writer = SummaryWriter(log_dir=log_dir)
    for epoch in range(start_epoch, epochs):

        scheduler.step()
        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              writer=writer)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion,
                                epoch=epoch,
                                writer=writer)

        # Check if there was an improvement

        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        # if not is_best:
        #     epochs_since_improvement += 1
        #     print("\nEpochs since last improvement: %d\n" %
        #           (epochs_since_improvement,))
        # else:
        #     epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint_basewithmiml4(save_path, prefix, epoch, encoder, decoder,
                                      optimizer, scheduler, recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion, optimizer, epoch, writer):
    """
    Performs one epoch's training.
    """
    encoder.train()
    decoder.train()  # train mode (dropout and batchnorm is used)

    total_step = len(train_loader)
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens, tags_target) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        #tags_target = tags_target.to(device)
        attributes, imgs_features = encoder(imgs)
        # loss2 = criterion2(attributes, tags_target)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
            attributes, imgs_features, caps, caplens)

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
        loss = criterion(scores, targets)  # + loss2
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        # Back prop.
        optimizer.zero_grad()
        # loss2.backward(retain_graph=True)
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

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


def validate(val_loader, encoder, decoder, criterion, epoch, writer):
    """
    Performs one epoch's validation.
    """

    encoder.eval()
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
        for i, (imgs, caps, caplens, allcaps, tags_target) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            tags_target = tags_target.to(device)
            # Forward prop.

            attributes, imgs_features = encoder(imgs)

            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
                attributes, imgs_features, caps, caplens)

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
            loss = criterion(scores, targets)  # + loss2
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
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


if __name__ == "__main__":
    main()
