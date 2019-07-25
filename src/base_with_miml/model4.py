import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
from src.head import Head
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    '''
    head : { MIML }
           { conv }
    '''

    def __init__(self, basemodel='res101', encoded_image_size=14, K=20, L=1024, fine_tune1=True, blocks=1, fine_tune2=True, freeze1=False, freeze2=False):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        self.head = Head(model=basemodel)

        # for MIML
        model = torchvision.models.resnet101(
            pretrained=True)  # pretrained ImageNet ResNet-101
        self.miml_intermidate = torch.nn.Sequential(OrderedDict([
            ('layer2', model.layer2),
            ('layer3', model.layer3)]))

        self.miml_last = torch.nn.Sequential(OrderedDict([
            ('layer4', model.layer4)]))

        dim = 2048
        map_size = 64
        self.K = K
        self.L = L
        self.miml_sub_concept_layer = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(dim, 512, 1)),
            ('dropout1', nn.Dropout(0.5)),  # (-1,512,14,14)
            ('conv2', nn.Conv2d(512, K*L, 1)),
            # input need reshape to (-1,L,K,H*W)
            ('maxpool1', nn.MaxPool2d((K, 1))),
            # reshape input to (-1,L,H*W), # permute(0,2,1)
            ('softmax1', nn.Softmax(dim=2)),
            # permute(0,2,1) # reshape to (-1,L,1,H*W)
            ('maxpool2', nn.MaxPool2d((1, map_size)))
        ]))

        self.fine_tune1(fine_tune=fine_tune1, blocks=blocks)
        if freeze1:
            self.freeze1()
        # for image features
        model = torchvision.models.resnet101(
            pretrained=True)  # pretrained ImageNet ResNet-101
        self.features_model = torch.nn.Sequential(OrderedDict([
            ('layer2', model.layer2),
            ('layer3', model.layer3),
            ('layer4', model.layer4)
        ]))
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size))
        self.fine_tune2(fine_tune=fine_tune2)
        if freeze2:
            self.freeze2()

    def forward(self, images):
        """
        Forward propagation.
        """
        head_out = self.head(images)

        # miml
        miml_features_out = self.miml_last(self.miml_intermidate(head_out))
        # (-1,2048,8,8)
        _, C, H, W = miml_features_out.shape
        conv1_out = self.miml_sub_concept_layer.dropout1(
            self.miml_sub_concept_layer.conv1(miml_features_out))
        # shape -> (n_bags, (L * K), n_instances, 1)
        conv2_out = self.miml_sub_concept_layer.conv2(conv1_out)
        # shape -> (n_bags, L, K, n_instances)
        conv2_out = conv2_out.reshape(-1, self.L, self.K, H*W)
        # shape -> (n_bags, L, 1, n_instances),remove dim: 1
        maxpool1_out = self.miml_sub_concept_layer.maxpool1(
            conv2_out).squeeze(2)
        # softmax
        permute1 = maxpool1_out.permute(0, 2, 1)
        softmax1 = self.miml_sub_concept_layer.softmax1(permute1)
        permute2 = softmax1.permute(0, 2, 1)
        # reshape = permute2.unsqueeze(2)
        # predictions_instancelevel
        reshape = permute2.reshape(-1, self.L, 1, H*W)
        # shape -> (n_bags, L, 1, 1)
        maxpool2_out = self.miml_sub_concept_layer.maxpool2(reshape)
        attributes = maxpool2_out.squeeze()

        # extract image features
        # (batch_size, 2048, image_size/32, image_size/32)
        images_features = self.features_model(head_out)
        # (batch_size, 2048, encoded_image_size, encoded_image_size)
        imgs_features = self.adaptive_pool(images_features)
        # (batch_size, encoded_image_size, encoded_image_size, 2048)
        imgs_features = imgs_features.permute(0, 2, 3, 1)
        return attributes, imgs_features

    def fine_tune1(self, fine_tune=True, blocks=1):
        """
        fine tune block and concept_layer
        """
        for p in self.miml_intermidate.parameters():
            p.requires_grad = False
        for p in self.miml_last.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        if blocks == 1:  # 是最后一个block的最后一个块，最后一个block[-30:]
            for p in self.miml_last.parameters():
                p.requires_grad = fine_tune

            for p in self.miml_sub_concept_layer.parameters():
                p.requires_grad = fine_tune

        elif blocks == 3:
            for p in self.miml_intermidate.parameters():
                p.requires_grad = fine_tune
            for p in self.miml_last.parameters():
                p.requires_grad = fine_tune
            for p in self.miml_sub_concept_layer.parameters():
                p.requires_grad = fine_tune

    def fine_tune2(self, fine_tune=True):
        """
        3 blocks
        """
        for p in self.features_model.parameters():
            p.requires_grad = False

        for p in self.features_model.parameters():
            p.requires_grad = fine_tune

    def freeze1(self):
        for p in self.miml_intermidate.parameters():
            p.requires_grad = False
        for p in self.miml_last.parameters():
            p.requires_grad = False
        for p in self.miml_sub_concept_layer.parameters():
            p.requires_grad = False

    def freeze2(self):
        for p in self.features_model.parameters():
            p.requires_grad = False


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim, attrs_dim, dropout=0.5):

        super(Attention, self).__init__()
        # linear layer to transform encoded image
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # linear layer to transform decoder's output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)

        self.attr_att = nn.Linear(attrs_dim, attention_dim)

        # linear layer to calculate values to be softmax-ed
        self.dropout = nn.Dropout(p=dropout)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        #self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_uphidden, decoder_downhidden):
        """
        Forward propagation.
        """
        # 转换 image features, 两路隐藏层状态
        att1 = self.encoder_att(encoder_out)

        att2 = self.attr_att(decoder_uphidden)

        att3 = self.decoder_att(decoder_downhidden)

        att = self.full_att(self.dropout(
            self.relu(att1 + att3.unsqueeze(1)+att2.unsqueeze(1)))).squeeze(2)
        alpha = self.softmax(att)  # (batch_size, num_pixels, 1)
        attention_weighted_encoding = (
            encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class Decoder(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attrs_dim, attention_dim, embed_dim, decoder_dim, vocab_size, device, encoder_dim=2048, dropout=0.5):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.dropout = dropout
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.decoder_dim = decoder_dim
        self.device = device
        # 上路
        self.attrs_dim = attrs_dim

        self.embed_dim = embed_dim

        self.attrs_dim = attrs_dim

        self.init_x0 = nn.Linear(attrs_dim, embed_dim)

        #self.dropout1 = nn.Dropout(p=self.dropout)
        self.decode_step1 = nn.LSTMCell(
            embed_dim, decoder_dim, bias=True)  # decoding LSTMCell
        # linear layer to find initial hidden state of LSTMCell
        self.init1_h = nn.Linear(attrs_dim, decoder_dim)
        # linear layer to find initial cell state of LSTMCell
        self.init1_c = nn.Linear(attrs_dim, decoder_dim)
        # linear layer to find scores over vocabulary
        # self.fc1 = nn.Linear(decoder_dim, vocab_size)

        # 下路
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim

        self.attention = Attention(
            encoder_dim, decoder_dim, attention_dim, attrs_dim)  # attention network

        self.dropout2 = nn.Dropout(p=self.dropout)
        self.decode_step2 = nn.LSTMCell(
            embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        # linear layer to find initial hidden state of LSTMCell
        self.init2_h = nn.Linear(encoder_dim, decoder_dim)
        # linear layer to find initial cell state of LSTMCell
        self.init2_c = nn.Linear(encoder_dim, decoder_dim)
        # linear layer to create a sigmoid-activated gate
        # self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        # self.sigmoid = nn.Sigmoid()
        # linear layer to find scores over vocabulary
        self.fc2 = nn.Linear(decoder_dim, vocab_size)

        #
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        # self.fc1.bias.data.fill_(0)
        # self.fc1.weight.data.uniform_(-0.1, 0.1)
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

    def init_hidden_state(self, attrs, encoder_out, zero=False):
        """
        Creates the initial hidden and cell states for the decoder's LSTM
        """
        batch_size = attrs.shape[0]
        if zero:
            h1 = torch.zeros(batch_size, self.decoder_dim).to(
                device)  # (batch_size, decoder_dim)
            c1 = torch.zeros(batch_size, self.decoder_dim).to(device)

            h2 = torch.zeros(batch_size, self.decoder_dim).to(
                device)  # (batch_size, decoder_dim)
            c2 = torch.zeros(batch_size, self.decoder_dim).to(device)

        else:
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
        """
        batch_size = attrs.shape[0]

        vocab_size = self.vocab_size
        encoder_dim = encoder_out.size(-1)

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        caption_lengths, sort_ind = caption_lengths.squeeze(
            1).sort(dim=0, descending=True)

        attrs = attrs[sort_ind]
        x0 = self.init_x0(attrs)

        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions)

        h1, c1, h2, c2 = self.init_hidden_state(attrs, encoder_out, zero=True)

        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(
            decode_lengths), vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max(
            decode_lengths), num_pixels).to(self.device)

        h1, c1 = self.decode_step1(x0, (h1, c1))
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            # 上

            # 下
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h1[:batch_size_t],
                                                                h2[:batch_size_t])

            h1, c1 = self.decode_step1(
                embeddings[:batch_size_t, t, :], (h1[:batch_size_t], c1[:batch_size_t]))
            # gate = self.sigmoid(self.f_beta(h2[:batch_size_t]))
            # attention_weighted_encoding = gate * attention_weighted_encoding
            h2, c2 = self.decode_step2(torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding,
                                                  ], dim=1), (h2[:batch_size_t], c2[:batch_size_t]))  # (batch_size_t, decoder_dim)

            preds = self.fc2(self.dropout2(h2))

            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lengths, sort_ind
