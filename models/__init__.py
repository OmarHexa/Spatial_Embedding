from models.BranchedERFNet import BranchedERFNet
from models.hypernet import HyperNet
def get_model(name, model_opts):
    if name == "branched_erfnet":
        model = BranchedERFNet(**model_opts)
        return model
    if name == "hypernet":
        model = HyperNet(**model_opts)
        return model
    else:
        raise RuntimeError("model \"{}\" not available".format(name))