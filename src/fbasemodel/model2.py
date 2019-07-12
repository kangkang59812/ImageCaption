import torch
import torch.nn as nn
import torchvision


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


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, features_dim, decoder_dim, attention_dim, , dropout=0.5):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        # linear layer to transform encoded image
        self.encoder_att = nn.Linear(features_dim, attention_dim)
        # linear layer to transform decoder's output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)

        self.attr_att = nn.Linear(decoder_dim, attention_dim)
        # linear layer to calculate values to be softmax-ed
        self.dropout = nn.Dropout(p=dropout)
        self.full_att = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, features, decoder_hidden1, decoder_hidden2):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(
            features)  # (batch_size, 36 , attention_dim=2048)

        att3 = self.attr_att(decoder_hidden1)

        att2 = self.decoder_att(decoder_hidden2)  # (batch_size, attention_dim)

        att = self.full_att(
            self.tanh(att1 + att2.unsqueeze(1) + att3.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (
            features * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attrs_dim, attention_dim, embed_dim, decoder_dim, attrs_size, vocab_size, device, features_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.vocab_size = vocab_size
        self.dropout = dropout
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.decoder_dim = decoder_dim
        self.device = device
        self.embed_dim = embed_dim

        # 上路
        self.attrs_dim = attrs_dim

        self.attrs_size = attrs_size

        self.init_x0 = nn.Linear(attrs_size, embed_dim)

        # self.dropout1 = nn.Dropout(p=self.dropout)
        self.decode_step1 = nn.LSTMCell(
            embed_dim, decoder_dim, bias=True)  # decoding LSTMCell
        # linear layer to find initial hidden state of LSTMCell
        self.init1_h = nn.Linear(attrs_dim, decoder_dim)
        # linear layer to find initial cell state of LSTMCell
        self.init1_c = nn.Linear(attrs_dim, decoder_dim)

        # 下路
        self.features_dim = features_dim
        self.attention_dim = attention_dim
        self.attention = Attention(
            features_dim, decoder_dim, attention_dim)  # attention network

        # embedding layer
        self.dropout2 = nn.Dropout(p=self.dropout)
        self.decode_step2 = nn.LSTMCell(
            embed_dim + features_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init2_h = nn.Linear(features_dim, decoder_dim)
        # linear layer to find initial cell state of LSTMCell
        self.init2_c = nn.Linear(features_dim, decoder_dim)
        # linear layer to find initial hidden state of LSTMCell

        # linear layer to find scores over vocabulary
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
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

    def init_hidden_state(self, attrs, image_features):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        h1 = self.init1_h(attrs)
        c1 = self.init1_c(attrs)

        mean_features = image_features.mean(dim=1)
        h2 = self.init2_h(mean_features)
        c2 = self.init2_c(mean_features)
        # h = torch.zeros(batch_size, self.decoder_dim).to(
        #     self.device)  # (batch_size, decoder_dim)
        # c = torch.zeros(batch_size, self.decoder_dim).to(self.device)
        return h1, c1, h2, c2

    def forward(self, attrs, image_features, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        # 获取像素数14*14， 和维度2048
        batch_size = image_features.size(0)
        x0 = self.init_x0(attrs)
        vocab_size = self.vocab_size

        # Sort input data by decreasing lengths; why? apparent below
        # 为了torch中的pack_padded_sequence，需要降序排列
        caption_lengths, sort_ind = caption_lengths.squeeze(
            1).sort(dim=0, descending=True)
        attrs = attrs[sort_ind]
        image_features = image_features[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        # (batch_size, max_caption_length, embed_dim)
        embeddings = self.embedding(encoded_captions)

        # Initialize LSTM state
        # (batch_size, decoder_dim)
        h1, c1, h2, c2 = self.init_hidden_state(attrs, image_features)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(
            decode_lengths), vocab_size).to(self.device)
        # alphas = torch.zeros(batch_size, max(
        #     decode_lengths), image_features.size(1)).to(self.device)

        h1, c1 = self.decode_step1(x0, (h1, c1))
        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            # 上
            h1, c1 = self.decode_step1(
                embeddings[:batch_size_t, t, :], (h1[:batch_size_t], c1[:batch_size_t]))
            # 下
            attention_weighted_encoding, alpha = self.attention(image_features[:batch_size_t],
                                                                h1[:batch_size_t], h2[:batch_size_t])
            # gating scalar, (batch_size_t, encoder_dim)
            # gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            # attention_weighted_encoding = gate * attention_weighted_encoding

            # embeddings[:batch_size_t, t, :]逐个取出单词
            h2, c2 = self.decode_step2(torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding,
                                                  ], dim=1), (h2[:batch_size_t], c2[:batch_size_t]))  # (batch_size_t, decoder_dim)

            preds = self.fc(self.dropout2(h2))  # (batch_size_t, vocab_size)

            predictions[:batch_size_t, t, :] = preds
            # alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, sort_ind
