from tf.keras.preprocessing.image import ImageDataGenerator


def preprocessing(img_dir: str, pp_input, h, w, b):
    img_generator = ImageDataGenerator(preprocessing_function=pp_input)
    img_flow = img_generator.flow_from_directory(
        img_dir,
        target_size=(h, w),
        batch_size=b
    )
    return img_flow
