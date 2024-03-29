import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from utils.data import CaptionDataset
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
from src.base_with_miml.model import Encoder, MIML, Decoder
import os
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
import json
# Parameters
# folder with data files saved by create_input_files.py
data_folder = '/home/lkk/datasets/coco2014/'
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
# model checkpoint
checkpoint = '/home/lkk/code/ImageCaption/checkpoint_basewithmiml_.pth.tar'
# word map, ensure it's the same the data was encoded with and the model was trained with
word_map_file = '/home/lkk/datasets/coco2014/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'
# sets device for model and PyTorch tensors
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# set to true only if inputs to model are fixed size; otherwise lot of computational overhead
cudnn.benchmark = True

emb_dim = 512  # dimension of word embeddings
attrs_dim = 1024  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
attrs_size = 1024
dropout = 0.5
attention_dim = 512
# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Load model
checkpoint = torch.load(checkpoint, map_location=device)
miml = MIML().to(device)
miml.load_state_dict(checkpoint['miml'])

miml.eval()

encoder = Encoder().to(device)
encoder.load_state_dict(checkpoint['encoder'])

encoder.eval()
decoder = Decoder(attrs_dim=attrs_dim, attention_dim=attention_dim,
                  embed_dim=emb_dim,
                  decoder_dim=decoder_dim,
                  attrs_size=attrs_size,
                  vocab_size=len(word_map),
                  device=device,
                  dropout=dropout).to(device)
decoder.load_state_dict(checkpoint['decoder'])
decoder.eval()


# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST',
                       transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = dict()
    hypotheses = dict()

    # For each image
    for j, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        attrs = miml(image).expand(3, attrs_dim)
        encoder_out = encoder(image)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)
        encoder_out = encoder_out.view(1, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)
        x0 = decoder.init_x0(attrs)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor(
            [[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h1, c1, h2, c2 = decoder.init_hidden_state(attrs, encoder_out)
        h1, c1 = decoder.decode_step1(x0, (h1, c1))
        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(
                k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h2)
            gate = decoder.sigmoid(decoder.f_beta(h2))
            awe = gate * awe

            h1, c1 = decoder.decode_step1(embeddings, (h1, c1))
            h2, c2 = decoder.decode_step2(
                torch.cat([embeddings, awe], dim=1), (h2, c2))

            pre1 = F.normalize(decoder.fc1(h1), p=2, dim=1)
            pre2 = F.normalize(decoder.fc2(h2), p=2, dim=1)
            scores = decoder.pre(pre1+pre2)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(
                    k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                # (s) 所有分数中最大的k个
                top_k_scores, top_k_words = scores.view(
                    -1).topk(k, 0, True, True)

            # Convert unrolled indices to actual indices of scores
            # 上面展开了,prev_word_inds得到哪些句子是概率最大的
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat(
                [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(
                set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h1 = h1[prev_word_inds[incomplete_inds]]
            c1 = c1[prev_word_inds[incomplete_inds]]
            h2 = h2[prev_word_inds[incomplete_inds]]
            c2 = c2[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]

            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        img_caps = [' '.join(c) for c in img_captions]
        # print(img_caps)
        references[str(j)] = img_caps

        # Hypotheses
        hypothesis = ([rev_word_map[w] for w in seq if w not in {
            word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        hypothesis = [' '.join(hypothesis)]
        # print(hypothesis)
        hypotheses[str(j)] = hypothesis

        assert len(references) == len(hypotheses)

    # Calculate BLEU-1~BLEU4 scores
    m1 = Bleu()
    m2 = Meteor()
    m3 = Cider()
    m4 = Rouge()
    m5 = Spice()
    (score1, scores1) = m1.compute_score(references, hypotheses)
    (score2, scores2) = m2.compute_score(references, hypotheses)
    (score3, scores3) = m3.compute_score(references, hypotheses)
    (score4, scores4) = m4.compute_score(references, hypotheses)
    (score5, scores5) = m5.compute_score(references, hypotheses)

    return score1, score2, score3, score4, score5


if __name__ == '__main__':
    beam_size = 3

    score1, score2, score3, score4, score5 = evaluate(beam_size)
    print("\nMetric score's @ beam size of {} is:\n \
          Bleu : {} \n \
          Meteor : {} \n \
          Cider : {} \n \
          Rouge : {} \n \
          Spice : {} ".format(
        beam_size, score1, score2, score3, score4, score5))
