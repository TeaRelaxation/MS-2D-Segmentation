from .unet import UNet
import segmentation_models_pytorch as smp


def select_model(model_name, n_classes):
    n_channels = 1
    segmentation_method, encoder_name = model_name.split('_', 1)
    segmentation_model = getattr(smp, segmentation_method)
    return segmentation_model(
        encoder_name=encoder_name.lower(), 
        in_channels=n_channels,
        classes=n_classes,
        encoder_weights=None,
    )


def load_model(model_path):
    # TODO
    return model_path
