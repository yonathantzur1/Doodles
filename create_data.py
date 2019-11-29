import ndjson
from PIL import Image
import os
import shutil

dataset_dir = './dataset/'
train_dir =  dataset_dir + 'train/'
test_dir = dataset_dir + 'test/'
validation_dir = dataset_dir + 'validation/'
ndjson_dir = './ndjson/'
images_per_class = 1000
skip = 0
ratio = 0.15


def create_folder(folder=None):
    if folder is None:
        return

    if not os.path.exists(folder):
        os.makedirs(folder)


def create_image(image, filename):
    img = Image.new('RGB', (256, 256), "white")
    pixels = img.load()

    x = -1
    y = -1

    for stroke in image:
        for i in range(len(stroke[0])):
            if x != -1:
                for point in get_line(stroke[0][i], stroke[1][i], x, y):
                    pixels[point[0], point[1]] = (0, 0, 0)
            pixels[stroke[0][i], stroke[1][i]] = (0, 0, 0)
            x = stroke[0][i]
            y = stroke[1][i]
        x = -1
        y = -1
    img.save(filename)


def get_line(x1, y1, x2, y2):
    points = []
    is_steep = abs(y2 - y1) > abs(x2 - x1)
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    delta_x = x2 - x1
    delta_y = abs(y2 - y1)
    error = int(delta_x / 2)
    y = y1
    if y1 < y2:
        y_step = 1
    else:
        y_step = -1
    for x in range(x1, x2 + 1):
        if is_steep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= delta_y
        if error < 0:
            y += y_step
            error += delta_x
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points


def create_dataset():
    print("loading...")
    for file_name in os.listdir(ndjson_dir):
        with open(ndjson_dir + file_name) as f:
            data = ndjson.load(f)
            create_folder(dataset_dir)
            total_images_amount = min(images_per_class, len(data))
            validation_images_amount = int(total_images_amount * ratio)
            test_images_amount = int((total_images_amount - validation_images_amount) * ratio)
            validation_range = range(0, validation_images_amount)
            test_range = range(validation_images_amount, validation_images_amount + test_images_amount)
            train_range = range(validation_images_amount + test_images_amount, total_images_amount)

            create_dataset_part(data, file_name, validation_range, validation_dir)
            create_dataset_part(data, file_name, test_range, test_dir)
            create_dataset_part(data, file_name, train_range, train_dir)

    print("done")


def create_dataset_part(data, file_name, range_obj, dir_path):
    for i in range_obj:
        class_name = file_name.split('_')[-1].split('.')[0]
        class_dir = dir_path + class_name + "/"
        create_folder(class_dir)
        img_data = data[i + skip]["drawing"]
        img_name = data[i + skip]['key_id']
        create_image(img_data, class_dir + img_name + '.jpg')


if __name__ == '__main__':
    create_dataset()
