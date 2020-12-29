import os
import shutil

from PIL import Image

imagenette_path = "imagenette2"
clean_classes_names = ["n01440764"]  # , "n02979186", "n03028079", "n03417042", "n03445777"]
noisy_classes_names = ["n02102040"]  # , "n03000684", "n03394916", "n03425413", "n03888257"]
JPEG_QUALITY = 10
IMG_SIZE = 224

COMPRESSION_SUFFIX = "_compressed_{}"


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


# def change_to_png(folder: str):
#     i = 1
#     for fn in os.listdir(folder):
#         print(i, end="\r")
#         if not os.path.isfile(os.path.join(folder, fn)):  # in case we found a folder
#             continue
#
#         img = Image.open(os.path.join(folder, fn))
#         img.save(os.path.join(folder, os.path.splitext(fn)[0] + ".png"))
#         os.remove(os.path.join(folder, fn))
#         i += 1


def compress_single_image(input_folder: str, img_name: str, quality: int, output_folder: str):
    img = Image.open(os.path.join(input_folder, img_name))
    new_name = (os.path.splitext(img_name)[0] + COMPRESSION_SUFFIX + ".JPEG").format(quality)
    out_path = os.path.join(output_folder, new_name)
    img.save(out_path, quality=quality)
    return out_path


def get_one_squared_func(full_name: str, smaller_edge_size: int):
    img = Image.open(full_name)

    # resize
    width, height = img.size
    if width <= height:
        fraction = width/smaller_edge_size
        size = (smaller_edge_size, int(height/fraction + 0.5))  # 0.5 to round
    else:
        fraction = height/smaller_edge_size
        size = (int(width/fraction + 0.5), smaller_edge_size)
    img = img.resize(size)

    # cropping
    width, height = img.size  # Get dimensions
    left = (width - smaller_edge_size) / 2
    top = (height - smaller_edge_size) / 2
    right = (width + smaller_edge_size) / 2
    bottom = (height + smaller_edge_size) / 2

    img = img.crop((left, top, right, bottom))

    img.save(full_name)


def get_em_squared_func(folder_name: str, smaller_edge_size: int):
    for fn in os.listdir(folder_name):
        full_name = os.path.join(folder_name, fn)
        if os.path.isfile(full_name):
            get_one_squared_func(full_name, smaller_edge_size)


def preprocess_folder_of_images(input_folder: str, quality: int, output_folder: str = None):
    """ creates output_folder if not present """
    # take care of output folder
    if output_folder is None:
        output_folder = input_folder if not input_folder[-1] == "\\" else input_folder[:-1]
        output_folder += COMPRESSION_SUFFIX
        output_folder = output_folder.format(quality)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(output_folder)

    # actual code
    i = 1
    for f in os.listdir(input_folder):
        print(i, end="\r")
        if not os.path.isfile(os.path.join(input_folder, f)):  # in case we found a folder
            continue
        full_name = compress_single_image(input_folder, f, quality, output_folder)
        get_one_squared_func(full_name, IMG_SIZE)
        i += 1
    # change_to_png(output_folder)
    print(i)


if __name__ == "__main__":
    os.makedirs("data")
    os.makedirs("data/train")
    os.makedirs("data/test")
    os.makedirs("data/test_denoised_only")

    ## training data
    train_source_folder = os.path.join(imagenette_path, "train")
    # copy the clean training data
    for dn in clean_classes_names:
        src = os.path.join(train_source_folder, dn)
        dst = "data/train/"+dn
        os.makedirs(dst)
        src_files = os.listdir(src)
        for file_name in src_files:
            full_file_name = os.path.join(src, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dst)
        get_em_squared_func(dst, IMG_SIZE)
        # change_to_png(dst)
        print(dst)

    # get noisy training data
    for dn in noisy_classes_names:
        preprocess_folder_of_images(os.path.join(train_source_folder, dn), JPEG_QUALITY, "data/train/"+dn)

    ## normal test data
    test_source_folder = os.path.join(imagenette_path, "val")
    # copy the clean test data
    for dn in clean_classes_names:
        src = os.path.join(test_source_folder, dn)
        dst = "data/test/"+dn
        os.makedirs(dst)
        src_files = os.listdir(src)
        for file_name in src_files:
            full_file_name = os.path.join(src, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dst)
        get_em_squared_func(dst, IMG_SIZE)
        # change_to_png(dst)
        print(dst)

    for dn in noisy_classes_names:
        preprocess_folder_of_images(os.path.join(test_source_folder, dn), JPEG_QUALITY, "data/test/"+dn)

    ## denoised test
    # copy denoised test data
    for dn in noisy_classes_names:
        src = os.path.join(test_source_folder, dn)
        dst = "data/test_denoised_only/"+dn
        os.makedirs(dst)
        src_files = os.listdir(src)
        for file_name in src_files:
            full_file_name = os.path.join(src, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dst)
        get_em_squared_func(dst, IMG_SIZE)
        # change_to_png(dst)
        print(dst)
