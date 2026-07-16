from triune.model.transformer import TriuneTransformer

def build_model(config):
    model = TriuneTransformer(
        vocab_size=config["vocab_size"],
        use_fp4=config["use_fp4"],
    )
    return model
