import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
from src.custom_LSTM.custom_lstms import aLSTMCell


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
        self.fine_tune(fine_tune)

        self.sub_concept_layer = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(512, 512, 1)),
            ('dropout1', nn.Dropout(0)),  # (-1,512,14,14)
            ('conv2', nn.Conv2d(512, K*L, 1)),
            # input need reshape to (-1,L,K,H*W)
            ('maxpool1', nn.MaxPool2d((K, 1))),
            # reshape input to (-1,L,H*W), # permute(0,2,1)
            ('softmax1', nn.Softmax(dim=2)),
            # permute(0,2,1) # reshape to (-1,L,1,H*W)
            ('maxpool2', nn.MaxPool2d((1, 196)))
        ]))

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

    def freeze_all(self):
        for p in self.base_model.parameters():
            p.requires_grad = False

        for p in self.sub_concept_layer:
            p.requires_grad = False


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, basemodel='res101', encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        self.basemodel = basemodel
        if self.basemodel == 'vgg16':
            # pretrained ImageNet ResNet-101
            model = torchvision.models.vgg16(pretrained=True)
            encoded_image_size = 28
            # Remove linear and pool layers (since we're not doing classification)
            modules = list(model.features)[:-1]
            self.model = nn.Sequential(*modules)
            # self.model输出大小是[-1, 512, 32, 32]
            # Resize image to fixed size to allow input images of variable size
            self.adaptive_pool = nn.AdaptiveAvgPool2d(
                (encoded_image_size, encoded_image_size))

            self.fine_tune()
        else:
            model = torchvision.models.resnet101(
                pretrained=True)  # pretrained ImageNet ResNet-101

            # Remove linear and pool layers (since we're not doing classification)
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
            # self.model输出大小是[-1, 2048, 16, 16]
            # Resize image to fixed size to allow input images of variable size
            self.adaptive_pool = nn.AdaptiveAvgPool2d(
                (encoded_image_size, encoded_image_size))

            self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.model(
            images)  # (batch_size, 2048, image_size/32, image_size/32)
        # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = self.adaptive_pool(out)
        # (batch_size, encoded_image_size, encoded_image_size, 2048)
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        if self.basemodel == 'vgg16':
            layer = -6
        else:
            layer = -7

        for p in self.model.parameters():
            p.requires_grad = False
            # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.model.children())[layer:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        # linear layer to transform encoded image
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # linear layer to transform decoder's output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # linear layer to calculate values to be softmax-ed
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(
            encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        # (batch_size, num_pixels)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (
            encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class Decoder(nn.Module):
    """
    MIML's Decoder.
    """

    def __init__(self, attrs_dim, attention_dim, embed_dim, decoder_dim, attrs_size, vocab_size, device, encoder_dim=2048, dropout=0.5):
        '''
        :param attrs_dim: size of MIML's output: 1024
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param attrs_size: size of attr's vocabulary
        :param dropout: dropout
        '''
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.dropout = dropout
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.decoder_dim = decoder_dim
        self.device = device
        # 上路
        self.attrs_dim = attrs_dim

        self.embed_dim = embed_dim

        self.attrs_size = attrs_size

        self.init_x0 = nn.Linear(attrs_size, embed_dim)

        self.dropout1 = nn.Dropout(p=self.dropout)
        self.decode_step1 = nn.LSTMCell(
            embed_dim, decoder_dim, bias=True)  # decoding LSTMCell
        # linear layer to find initial hidden state of LSTMCell
        self.init1_h = nn.Linear(attrs_dim, decoder_dim)
        # linear layer to find initial cell state of LSTMCell
        self.init1_c = nn.Linear(attrs_dim, decoder_dim)
        # linear layer to find scores over vocabulary
        self.fc1 = nn.Linear(decoder_dim, vocab_size)

        # 下路
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim

        self.attention = Attention(
            encoder_dim, decoder_dim, attention_dim)  # attention network

        self.dropout2 = nn.Dropout(p=self.dropout)
        self.decode_step2 = aLSTMCell(
            embed_dim + encoder_dim, decoder_dim, decoder_dim)  # decoding LSTMCell
        # linear layer to find initial hidden state of LSTMCell
        self.init2_h = nn.Linear(encoder_dim, decoder_dim)
        # linear layer to find initial cell state of LSTMCell
        self.init2_c = nn.Linear(encoder_dim, decoder_dim)
        # linear layer to create a sigmoid-activated gate
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        # linear layer to find scores over vocabulary
        self.fc2 = nn.Linear(decoder_dim, vocab_size)

        #
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-0.1, 0.1)

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

    def init_hidden_state(self, attrs, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM
        """
        h1 = self.init1_h(attrs)
        c1 = self.init1_c(attrs)

        mean_encoder_out = encoder_out.mean(dim=1)
        # mean_encoder_out：[32,2048]转为512维
        h2 = self.init2_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c2 = self.init2_c(mean_encoder_out)

        return h1, c1, h2, c2

    def forward(self, attrs, encoder_out, encoded_captions, caption_lengths):
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
        encoder_dim = encoder_out.size(-1)

        # 下路
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        caption_lengths, sort_ind = caption_lengths.squeeze(
            1).sort(dim=0, descending=True)

        # 重新按照句子长度排序
        encoder_out = encoder_out[sort_ind]
        attrs = attrs[sort_ind]

        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions)

        h1, c1, h2, c2 = self.init_hidden_state(attrs, encoder_out)

        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(
            decode_lengths), vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max(
            decode_lengths), num_pixels).to(self.device)

        h1, c1 = self.decode_step1(x0, (h1, c1))  # (batch_size_t, decoder_dim)
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            # 上
            h1, c1 = self.decode_step1(
                embeddings[:batch_size_t, t, :], (h1[:batch_size_t], c1[:batch_size_t]))
            # 下
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h2[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h2[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h2, c2 = self.decode_step2(torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding,
                                                  ], dim=1), (h2[:batch_size_t], c2[:batch_size_t]), h1)  # (batch_size_t, decoder_dim)

            preds = self.fc2(self.dropout2(h2))

            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
