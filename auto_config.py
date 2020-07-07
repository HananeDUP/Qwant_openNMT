import opennmt as onmt
from opennmt.utils import misc

def auto_config(self, num_replicas=1):
    config = super(onmt.models.SequenceTagger, self).auto_config(num_replicas=num_replicas)
    max_length = config["train"]["maximum_features_length"]
    return misc.merge_dict(config, {
        "train": {
            "maximum_features_length": [max_length, max_length]
        }
    })