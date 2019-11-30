import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model, load_model
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from flask import Flask
from flask import request

model_path = "./model.h5"
image_size = [256, 256]
app = Flask(__name__)
dogs = ['flamingo', 'bread', 'wristwatch', 'firetruck', 'blackberry', 'pencil', 'shovel', 'eye', 'matches', 'marker',
        'lighthouse', 'snorkel', 'bathtub', 'hammer', 'underwear', 'bee', 'ocean', 'skull', 'rake', 'onion', 'sun',
        'piano', 'baseball', 'hockey stick', 'toaster', 'nose', 'zigzag', 'floor lamp', 'horse', 'screwdriver', 'crown',
        'lollipop', 'guitar', 'purse', 'basket', 'butterfly', 'lantern', 'helmet', 'jacket', 'teddy-bear', 'foot',
        'sheep', 'fireplace', 'bush', 'door', 'fan', 'skyscraper', 'broccoli', 'binoculars', 'hexagon', 'mountain',
        'ant', 'scorpion', 'drums', 'telephone', 'frog', 'candle', 'wheel', 'table', 'paper clip', 'rifle', 'stove',
        'pig', 'diving board', 'sailboat', 'line', 'cooler', 'jail', 'star', 'police car', 'pants', 'toothbrush',
        'television', 'parrot', 'soccer ball', 'house plant', 'hospital', 'spoon', 'bird', 'square', 'snowman',
        'eyeglasses', 'ladder', 'microphone', 'map', 'pear', 'hockey puck', 'knee', 'goatee', 'tennis racquet', 'angel',
        'feather', 'river', 'penguin', 'cookie', 'mug', 'dishwasher', 'tiger', 'circle', 'stairs', 'flying saucer',
        'mailbox', 'sword', 'octagon', 'ceiling fan', 'bandage', 'finger', 'hand', 'suitcase', 'laptop',
        'roller coaster', 'bottlecap', 'bench', 'belt', 'bracelet', 'grapes', 'cat', 'mouth', 'boomerang', 'tractor',
        'alarm clock', 'remote control', 'yoga', 'baseball bat', 'strawberry', 'squirrel', 'cow', 'hot air balloon',
        'hamburger', 'ear', 'stereo', 'campfire', 'octopus', 'see saw', 'dresser', 'banana', 'flower', 'bus', 'tooth',
        'house', 'spreadsheet', 'barn', 'trombone', 'ambulance', 'nail', 'picture frame', 'bulldozer', 'light bulb',
        'blueberry', 'traffic light', 'sea turtle', 'headphones', 'moon', 'rainbow', 'key', 'trumpet', 'stitches',
        'knife', 'oven', 'camouflage', 'chair', 'postcard', 'apple', 'cactus', 'face', 'fish', 'animal migration',
        'cup', 'cello', 'cloud', 'swan', 'elephant', 'power outlet', 'pineapple', 'toilet', 'clarinet', 'kangaroo',
        'canoe', 'wine glass', 'book', 'hurricane', 'palm tree', 'hot dog', 'pizza', 'speedboat', 'waterslide', 'sink',
        'axe', 'teapot', 'stethoscope', 'cruise ship', 'couch', 'fence', 'popsicle', 'rhinoceros', 'The Eiffel Tower',
        'bridge', 'owl', 'ice cream', 'sandwich', 'hot tub', 'stop sign', 'sock', 'necklace', 'submarine', 'harp',
        'hourglass', 't-shirt', 'calculator', 'windmill', 'cannon', 'snowflake', 'microwave', 'potato', 'hedgehog',
        'toothpaste', 'school bus', 'The Great Wall of China', 'lipstick', 'motorbike', 'smiley face', 'skateboard',
        'squiggle', 'bear', 'leg', 'envelope', 'scissors', 'radio', 'eraser', 'flip flops', 'sleeping bag', 'crab',
        'lighter', 'raccoon', 'bat', 'hat', 'mermaid', 'pliers', 'clock', 'dolphin', 'panda', 'whale', 'backpack',
        'saw', 'calendar', 'aircraft carrier', 'syringe', 'washing machine', 'keyboard', 'mouse', 'paintbrush', 'duck',
        'saxophone', 'beard', 'zebra', 'passport', 'fire hydrant', 'crocodile', 'bowtie', 'lobster', 'grass',
        'lightning', 'snail', 'mosquito', 'monkey', 'flashlight', 'airplane', 'frying pan', 'pillow', 'leaf',
        'string bean', 'garden', 'toe', 'parachute', 'wine bottle', 'donut', 'crayon', 'paint can', 'bed', 'snake',
        'camera', 'golf club', 'tent', 'triangle', 'dog', 'pool', 'basketball', 'tree', 'peas', 'watermelon',
        'computer', 'bicycle', 'brain', 'birthday cake', 'umbrella', 'The Mona Lisa', 'giraffe', 'shorts', 'train',
        'diamond', 'moustache', 'spider', 'tornado', 'shoe', 'pickup truck', 'streetlight', 'lion', 'truck',
        'chandelier', 'anvil', 'fork', 'arm', 'elbow', 'broom', 'car', 'sweater', 'carrot', 'dumbbell', 'cell phone',
        'rain', 'steak', 'mushroom', 'castle', 'church', 'camel', 'drill', 'pond', 'garden hose', 'cake', 'bucket',
        'van', 'dragon', 'peanut', 'asparagus', 'coffee cup', 'shark', 'beach', 'swing set', 'megaphone', 'rabbit',
        'vase', 'violin', 'rollerskates', 'compass', 'helicopter']


def get_doodle(img_str):
    img_arr = img_to_array(img_str)
    output = list(list(model.predict(img_arr))[0])
    max_value = max(output)
    max_index = output.index(max_value)

    return {
        "doodle": dogs[max_index],
        "accuracy": str(max_value * 100) + "%"
    }


def img_to_array(base64_img):
    img = Image.open(BytesIO(base64.b64decode(base64_img))).resize(image_size, Image.ANTIALIAS)
    return np.array(img.getdata(),
                    np.float32).reshape(1, img.size[1], img.size[0], 3)


@app.route('/doodle', methods=["POST"])
def find():
    return get_doodle(request.json["img"])


if __name__ == '__main__':
    model = load_model(model_path)
    app.run(host='0.0.0.0', port=5000)
