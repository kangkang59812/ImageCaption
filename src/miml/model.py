import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict


class MIML(nn.Module):

    def __init__(self, L=1024, K=20, freeze=True, fine_tune=True):
        """
        Arguments:
            L (int):
                number of labels
            K (int):
                number of sub categories
        """
        super(MIML, self).__init__()
        self.L = L
        self.K = K
        # self.batch_size = batch_size
        # pretrained ImageNet VGG
        base_model = torchvision.models.vgg16(pretrained=True)
        base_model = list(base_model.features)[:-1]
        self.base_model = nn.Sequential(*base_model)

        self.sub_concept_layer = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(512, 512, 1)),
            ('dropout1', nn.Dropout(0.5)),  # (-1,512,14,14)
            ('conv2', nn.Conv2d(512, K*L, 1)),
            # input need reshape to (-1,L,K,H*W)
            ('maxpool1', nn.MaxPool2d((K, 1))),
            # reshape input to (-1,L,H*W), # permute(0,2,1)
            ('softmax1', nn.Softmax(dim=2)),
            # permute(0,2,1) # reshape to (-1,L,1,H*W)
            ('maxpool2', nn.MaxPool2d((1, 196)))
        ]))
        self.fine_tune(fine_tune)
        if freeze:
            self.freeze_all()
        # self.conv1 = nn.Conv2d(512, 512, 1))

        # self.dropout1=nn.Dropout(0.5)

        # self.conv2=nn.Conv2d(512, K*L, 1)
        # # input need reshape to (-1,L,K,H*W)
        # self.maxpool1=nn.MaxPool2d((K, 1))
        # # reshape input to (-1,L,H*W)
        # # permute(0,2,1)
        # self.softmax1=nn.Softmax(dim = 2)
        # # permute(0,2,1)
        # # reshape to (-1,L,1,H*W)
        # self.maxpool2=nn.MaxPool2d((1, 196))
        # # squeeze()

    def forward(self, x):
        # IN:(8,3,224,224)-->OUT:(8,512,14,14)
        base_out = self.base_model(x)
        # C,H,W = 512,14,14
        _, C, H, W = base_out.shape
        # OUT:(8,512,14,14)

        conv1_out = self.sub_concept_layer.dropout1(
            self.sub_concept_layer.conv1(base_out))

        # shape -> (n_bags, (L * K), n_instances, 1)
        conv2_out = self.sub_concept_layer.conv2(conv1_out)
        # shape -> (n_bags, L, K, n_instances)
        conv2_out = conv2_out.reshape(-1, self.L, self.K, H*W)
        # shape -> (n_bags, L, 1, n_instances),remove dim: 1
        maxpool1_out = self.sub_concept_layer.maxpool1(conv2_out).squeeze(2)

        # softmax
        permute1 = maxpool1_out.permute(0, 2, 1)
        softmax1 = self.sub_concept_layer.softmax1(permute1)
        permute2 = softmax1.permute(0, 2, 1)
        # reshape = permute2.unsqueeze(2)
        # predictions_instancelevel
        reshape = permute2.reshape(-1, self.L, 1, H*W)
        # shape -> (n_bags, L, 1, 1)
        maxpool2_out = self.sub_concept_layer.maxpool2(reshape)
        out = maxpool2_out.squeeze()

        return out

    def fine_tune(self, fine_tune=True):
        # only fine_tune the last three convs
        layer = -6
        for p in self.base_model.parameters():
            p.requires_grad = False
        for c in list(self.base_model.children())[-6:]:
            for p in c.parameters():
                p.requires_grad = True
        for p in self.sub_concept_layer.parameters():
            p.requires_grad = True

    def freeze_all(self):
        for p in self.base_model.parameters():
            p.requires_grad = False

        for p in self.sub_concept_layer:
            p.requires_grad = False


class Decoder(nn.Module):
    """
    MIML's Decoder.
    """

    def __init__(self, attrs_dim, embed_dim, decoder_dim, attrs_size, vocab_size, device, dropout=0.5):
        '''
        :param attrs_dim: size of MIML's output: 1024
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param attrs_size: size of attr's vocabulary
        :param dropout: dropout
        '''
        super(Decoder, self).__init__()

        self.attrs_dim = attrs_dim

        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.attrs_size = attrs_size
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.device = device

        self.init_x0 = nn.Linear(attrs_size, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer

        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(
            embed_dim, decoder_dim, bias=True)  # decoding LSTMCell
        # linear layer to find initial hidden state of LSTMCell
        self.init_h = nn.Linear(attrs_dim, decoder_dim)
        # linear layer to find initial cell state of LSTMCell
        self.init_c = nn.Linear(attrs_dim, decoder_dim)
        # linear layer to find scores over vocabulary
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.decode_step.weight_hh.data.uniform_(-0.1, 0.1)
        self.decode_step.weight_ih.data.uniform_(-0.1, 0.1)
        self.decode_step.bias_hh.data.uniform_(-0.1, 0.1)
        self.decode_step.bias_ih.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, attrs):
        """
        Creates the initial hidden and cell states for the decoder's LSTM
        """
        h = self.init_h(attrs)
        c = self.init_c(attrs)
        return h, c

    def forward(self, attrs, encoded_captions, caption_lengths):
        """
        Forward propagation.

       :param attrs: attributes, a tensor of dimension (batch_size, attrs_dim)
       :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
       :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
       :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
       """
        batch_size = attrs.shape[0]
        x0 = self.init_x0(attrs)
        vocab_size = self.vocab_size

        caption_lengths, sort_ind = caption_lengths.squeeze(
            1).sort(dim=0, descending=True)
        attrs = attrs[sort_ind]

        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions)

        h, c = self.init_hidden_state(attrs)

        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(
            decode_lengths), vocab_size).to(self.device)

        h, c = self.decode_step(x0, (h, c))  # (batch_size_t, decoder_dim)
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            h, c = self.decode_step(
                embeddings[:batch_size_t, t, :], (h[:batch_size_t], c[:batch_size_t]))

            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)

            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lengths, sort_ind

# class Faster_MIML(nn.Module):

#     def __init__(self, L=1024, K=20, freeze=True):
#         """
#         Arguments:
#             L (int):
#                 number of labels
#             K (int):
#                 number of sub categories
#         """
#         super(MIML, self).__init__()
#         self.L = L
#         self.K = K

#         self.sub_concept_layer = nn.Sequential(OrderedDict([
#             ('conv1', nn.Conv2d(36, 36, 1)),
#             ('dropout1', nn.Dropout(0)),  # (-1,512,14,14)
#             ('conv2', nn.Conv2d(36, K*L, 1)),
#             # input need reshape to (-1,L,K,H*W)
#             ('maxpool1', nn.MaxPool2d((K, 1))),
#             # reshape input to (-1,L,H*W), # permute(0,2,1)
#             ('softmax1', nn.Softmax(dim=2)),
#             # permute(0,2,1) # reshape to (-1,L,1,H*W)
#             ('maxpool2', nn.MaxPool2d((1, 196)))
#         ]))

#         if freeze:
#             self.freeze_all()
#         # self.conv1 = nn.Conv2d(512, 512, 1))

#         # self.dropout1=nn.Dropout(0.5)

#         # self.conv2=nn.Conv2d(512, K*L, 1)
#         # # input need reshape to (-1,L,K,H*W)
#         # self.maxpool1=nn.MaxPool2d((K, 1))
#         # # reshape input to (-1,L,H*W)
#         # # permute(0,2,1)
#         # self.softmax1=nn.Softmax(dim = 2)
#         # # permute(0,2,1)
#         # # reshape to (-1,L,1,H*W)
#         # self.maxpool2=nn.MaxPool2d((1, 196))
#         # # squeeze()

#     def forward(self, x):
#         # IN:(8,3,224,224)-->OUT:(8,512,14,14)
#         base_out = self.base_model(x)
#         # C,H,W = 512,14,14
#         _, C, H, W = base_out.shape
#         # OUT:(8,512,14,14)

#         conv1_out = self.sub_concept_layer.dropout1(
#             self.sub_concept_layer.conv1(base_out))

#         # shape -> (n_bags, (L * K), n_instances, 1)
#         conv2_out = self.sub_concept_layer.conv2(conv1_out)
#         # shape -> (n_bags, L, K, n_instances)
#         conv2_out = conv2_out.reshape(-1, self.L, self.K, H*W)
#         # shape -> (n_bags, L, 1, n_instances),remove dim: 1
#         maxpool1_out = self.sub_concept_layer.maxpool1(conv2_out).squeeze(2)

#         # softmax
#         permute1 = maxpool1_out.permute(0, 2, 1)
#         softmax1 = self.sub_concept_layer.softmax1(permute1)
#         permute2 = softmax1.permute(0, 2, 1)
#         # reshape = permute2.unsqueeze(2)
#         # predictions_instancelevel
#         reshape = permute2.reshape(-1, self.L, 1, H*W)
#         # shape -> (n_bags, L, 1, 1)
#         maxpool2_out = self.sub_concept_layer.maxpool2(reshape)
#         out = maxpool2_out.squeeze()

#         return out

#     def freeze_all(self):

#         for p in self.sub_concept_layer:
#             p.requires_grad = False


if __name__ == "__main__":
    model = MIML()
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0, 1])
    checkpoint = torch.load(
        '/home/lkk/code/caption_v1/checkpoint/MIML.pth.tar')
    model.load_state_dict(checkpoint['model'])
    print('')
    # out = model(torch.randn(8, 3, 224, 224))
    # print(out.shape)
    # summary(model.cuda(), (3, 224, 224), 8)
