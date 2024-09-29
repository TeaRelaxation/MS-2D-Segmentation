import segmentation_models_pytorch as smp


def select_model(model_name, n_classes, in_channels, is_imagenet):
    segmentation_method, encoder_name = model_name.split('_', 1)
    segmentation_model = getattr(smp, segmentation_method)
    encoder_weights = "imagenet" if is_imagenet == "True" else None
    return segmentation_model(
        encoder_name=encoder_name.lower(), 
        in_channels=in_channels,
        classes=n_classes,
        encoder_weights=encoder_weights
    )


def load_model(model_path):
    # TODO
    return model_path
