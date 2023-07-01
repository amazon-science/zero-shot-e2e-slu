import torch
# from transformers import RobertaTokenizer, RobertaModel
from transformers import BertModel


class BertEnc(torch.nn.Module):
    """
    SelectiveNet for classification with rejection option.
    In the experiments of original papaer, variant of VGG-16 is used as body block for feature extraction.
    """

    def __init__(self, model_name, robert_feature_dim, target_feature_dim, freeze_bert_enc=True):
        """
        Args
            features: feature extractor network (called body block in the paper).
            dim_featues: dimension of feature from body block.
            num_classes: number of classification class.
        """
        super(BertEnc, self).__init__()
        self.model_name = model_name
        self.robert_feature_dim = robert_feature_dim
        self.target_feature_dim = target_feature_dim
        self.freeze_bert_enc = freeze_bert_enc

        # self.robert_enc = RobertaModel.from_pretrained(self.model_name)
        self.robert_enc = BertModel.from_pretrained(self.model_name)

        self.enc_head = torch.nn.Sequential(
            torch.nn.Linear(self.robert_feature_dim, self.target_feature_dim * 2),
            torch.nn.ReLU(True),
            # torch.nn.BatchNorm1d(self.target_feature_dim * 2),
            torch.nn.Linear(self.target_feature_dim * 2, self.target_feature_dim)
        )


    def forward(self, encoded_input):
        if self.freeze_bert_enc:
            with torch.no_grad():
                output = self.robert_enc(**encoded_input)
        else:
            output = self.robert_enc(**encoded_input)
        output = self.enc_head(output['last_hidden_state'])
        return output
