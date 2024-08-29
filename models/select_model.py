from .unet import UNet


def select_model(model_name, n_classes):
    n_channels = 1
    if model_name == "UNet":
        return UNet(n_channels=n_channels, n_classes=n_classes, bilinear=False)
    else:
        return None


def load_model(model_path):
    # TODO
    return model_path
