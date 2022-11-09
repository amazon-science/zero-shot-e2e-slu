import torch
from transformers import RobertaTokenizer, RobertaModel


class RoBertEnc(torch.nn.Module):
    """
    SelectiveNet for classification with rejection option.
    In the experiments of original papaer, variant of VGG-16 is used as body block for feature extraction.
    """

    def __init__(self, model_name, robert_feature_dim, target_feature_dim, freeze_robert_enc=True):
        """
        Args
            features: feature extractor network (called body block in the paper).
            dim_featues: dimension of feature from body block.
            num_classes: number of classification class.
        """
        super(RoBertEnc, self).__init__()
        self.model_name = model_name
        self.robert_feature_dim = robert_feature_dim
        self.target_feature_dim = target_feature_dim

        self.robert_enc = RobertaModel.from_pretrained(self.model_name)
        self.enc_head = torch.nn.Sequential(
            torch.nn.Linear(self.robert_feature_dim, self.target_feature_dim)
        )

        if freeze_robert_enc:
            self.freeze_robert_enc_para()

    def freeze_robert_enc_para(self):
        for p in self.robert_enc.parameters():
            p.require_grad = False


    def forward(self, encoded_input):
        output = self.robert_enc(**encoded_input)
        output = self.enc_head(output)
        return output
