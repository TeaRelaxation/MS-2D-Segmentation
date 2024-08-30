from .unet import UNet
import segmentation_models_pytorch as smp


def select_model(model_name, n_classes):
    n_channels = 1
    if model_name == "UNet":
        return UNet(n_channels=n_channels, n_classes=n_classes, bilinear=False)
    if model_name == "DeepLabV3Plus_ResNet34":
        return smp.DeepLabV3Plus(
            encoder_name="resnet34",
            in_channels=n_channels,
            classes=n_classes,
            encoder_weights=None,
        )
    else:
        return None


def load_model(model_path):
    # TODO
    return model_path
