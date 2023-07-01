import torch


class SelectiveNet(torch.nn.Module):
    """
    SelectiveNet for classification with rejection option.
    In the experiments of original papaer, variant of VGG-16 is used as body block for feature extraction.
    """

    def __init__(self, features, dim_features: int, num_classes: int, init_weights=True):
        """
        Args
            features: feature extractor network (called body block in the paper).
            dim_featues: dimension of feature from body block.
            num_classes: number of classification class.
        """
        super(SelectiveNet, self).__init__()
        self.features = features
        self.dim_features = dim_features
        self.num_classes = num_classes

        # represented as f() in the original paper
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, self.num_classes)
        )

        # represented as g() in the original paper
        self.selector = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, self.dim_features),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(self.dim_features),
            torch.nn.Linear(self.dim_features, 1),
            torch.nn.Sigmoid()
        )

        # represented as h() in the original paper
        self.aux_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, self.num_classes)
        )

        # initialize weights of heads
        if init_weights:
            self._initialize_weights(self.classifier)
            self._initialize_weights(self.selector)
            self._initialize_weights(self.aux_classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        prediction_out = self.classifier(x)
        selection_out = self.selector(x)
        auxiliary_out = self.aux_classifier(x)

        return prediction_out, selection_out, auxiliary_out

    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

class CrossModalSelectiveNet(torch.nn.Module):
    def __init__(self, slu, nlu, audio_fea_dim, text_fea_dim, com_fea_dim, dropout_rate=0.1):
        """
        Args
            slu: a slu model = slu_enc + slu_dec
            nlu: a nlu model = nlu_enc + nlu_enc
            audio_fea_dim: the dimensions of audio features from slu encoder
            text_fea_dim: the dimensions of text features from nlu encoder
            com_fea_dim: the dimensions of common space
        """
        super(CrossModalSelectiveNet, self).__init__()
        self.slu = slu
        self.nlu = nlu

        for parms in self.nlu.parameters():
            parms.require_grad = False

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.sel_audio_projector = torch.nn.Linear(audio_fea_dim, com_fea_dim)
        self.sel_text_projector = torch.nn.Linear(text_fea_dim, com_fea_dim)

        self.aux_audio_projector = torch.nn.Linear(audio_fea_dim, com_fea_dim)
        self.aux_text_projector = torch.nn.Linear(text_fea_dim, com_fea_dim)

        self.selector = torch.nn.Sequential(
            torch.nn.Linear(2 * com_fea_dim, 2 * com_fea_dim),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(2 * com_fea_dim),
            torch.nn.Linear(2 * com_fea_dim, 1),
            torch.nn.Sigmoid()
        )


    def forward(self, audio, text):
        audio_fea = self.slu.enc(audio)
        text_fea = self.nlu.enc(text)

        sel_prj_audio_fea = self.sel_audio_projector(self.dropout(audio_fea))
        sel_prj_text_fea = self.sel_text_projector(self.dropout(text_fea))

        aux_prj_audio_fea = self.aux_audio_projector(self.dropout(audio_fea))
        aux_prj_text_fea = self.aux_text_projector(self.dropout(text_fea))

        sel_score = self.selector(torch.cat((sel_prj_text_fea,sel_prj_audio_fea), dim=0)) # can use the audio_fea & text_fea

        return sel_prj_audio_fea, sel_prj_text_fea, aux_prj_audio_fea, aux_prj_text_fea, sel_score


class SpeechBrain_CrossModalSelectiveNet(torch.nn.Module):
    def __init__(self, audio_fea_dim, text_fea_dim, com_fea_dim, single_sample_per_batch, dropout_rate=0.1):
        """
        Args
            slu: a slu model = slu_enc + slu_dec
            nlu: a nlu model = nlu_enc + nlu_enc
            audio_fea_dim: the dimensions of audio features from slu encoder
            text_fea_dim: the dimensions of text features from nlu encoder
            com_fea_dim: the dimensions of common space
        """
        super(SpeechBrain_CrossModalSelectiveNet, self).__init__()


        self.dropout = torch.nn.Dropout(dropout_rate)
        self.sel_audio_projector = torch.nn.Linear(audio_fea_dim, com_fea_dim)
        self.sel_text_projector = torch.nn.Linear(text_fea_dim, com_fea_dim)

        self.aux_audio_projector = torch.nn.Linear(audio_fea_dim, com_fea_dim)
        self.aux_text_projector = torch.nn.Linear(text_fea_dim, com_fea_dim)

        if single_sample_per_batch:
            self.selector = torch.nn.Sequential(
            torch.nn.Linear(2 * com_fea_dim, 2 * com_fea_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(2 * com_fea_dim, 1),
            torch.nn.Sigmoid()
            )
        else:
            self.selector = torch.nn.Sequential(
            torch.nn.Linear(2 * com_fea_dim, 2 * com_fea_dim),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(2 * com_fea_dim),
            torch.nn.Linear(2 * com_fea_dim, 1),
            torch.nn.Sigmoid()
            )

    def forward(self, audio_fea, text_fea):


        sel_prj_audio_fea = self.sel_audio_projector(self.dropout(audio_fea))
        sel_prj_text_fea = self.sel_text_projector(self.dropout(text_fea))

        aux_prj_audio_fea = self.sel_audio_projector(self.dropout(audio_fea))
        aux_prj_text_fea = self.sel_text_projector(self.dropout(text_fea))

        sel_score = self.selector(
            torch.cat((sel_prj_text_fea, sel_prj_audio_fea), dim=1))  # can use the audio_fea & text_fea


        return sel_prj_audio_fea, sel_prj_text_fea, aux_prj_audio_fea, aux_prj_text_fea, sel_score

