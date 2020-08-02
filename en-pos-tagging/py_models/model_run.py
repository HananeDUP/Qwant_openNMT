import opennmt
import seq_tagger_updated as SequenceTagger

config = {
    "model_dir": "run/",
    "data": {
        "train_features_file": "train_words_bitext.txt",
        "train_labels_file": "train_tags_bitext.txt",
        "eval_features_file": "valid_words_bitext.txt",
        "eval_labels_file": "valid_tags_bitext.txt",
        "source_1_vocabulary": "src-train-vocab.txt",
        "source_2_vocabulary": "src-train-tkt-vocab.txt",
        "target_vocabulary": "tgt-train-vocab.txt",
    },

}

# model = SequenceTagger.model()

#model = opennmt.models.SequenceTagger()
model = opennmt.models.catalog.LstmCnnCrfTagger()
runner = opennmt.Runner(model, config,auto_config=True)
runner.train(num_devices=2, with_eval=True)