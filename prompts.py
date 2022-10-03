categories_list = {
    'PACS':['dog','elephant','giraffe','guitar', 'horse', 'house', 'person'],
    'VLCS':['car', 'person', 'dog', 'bird', 'chair'],
    'OfficeHome':['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
        'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
        'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
        'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
        'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
        'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
        'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker'],
    'Terra':['coyote', 'dog', 'raccoon', 'opossum', 'bird', 'bobcat', 'squirrel', 'cat', 'rabbit', 'empty'],
    'DomainNet':['aircraft_carrier', 'airplane', 'alarm_clock', 'ambulance', 'angel', 'animal_migration', 'ant', 'anvil',
        'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball_bat',
        'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench',
        'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book', 'boomerang',
        'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket',
        'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera',
        'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan',
        'cello', 'cell_phone', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud',
        'coffee_cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile',
        'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 'donut',
        'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant',
        'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire_hydrant',
        'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip_flops', 'floor_lamp', 'flower',
        'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 'garden', 'garden_hose', 'giraffe', 'goatee',
        'golf_club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones',
        'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck', 'hockey_stick', 'horse', 'hospital',
        'hot_air_balloon', 'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant', 'hurricane', 'ice_cream',
        'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf',
        'leg', 'light_bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster',
        'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave',
        'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom',
        'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paintbrush',
        'paint_can', 'palm_tree', 'panda', 'pants', 'paper_clip', 'parachute', 'parrot', 'passport', 'peanut',
        'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup_truck', 'picture_frame', 'pig', 'pillow',
        'pineapple', 'pizza', 'pliers', 'police_car', 'pond', 'pool', 'popsicle', 'postcard', 'potato',
        'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote_control',
        'rhinoceros', 'rifle', 'river', 'roller_coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw',
        'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw', 'shark',
        'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag',
        'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat',
        'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo',
        'stethoscope', 'stitches', 'stop_sign', 'stove', 'strawberry', 'streetlight', 'string_bean', 'submarine',
        'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword', 'syringe', 'table', 'teapot', 'teddy-bear',
        'telephone', 'television', 'tennis_racquet', 'tent', 'The_Eiffel_Tower', 'The_Great_Wall_of_China',
        'The_Mona_Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado',
        'tractor', 'traffic_light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 't-shirt',
        'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine', 'watermelon', 'waterslide',
        'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']
        }

# Empty: standard prompt, with no special domain style
# Task: name with dataset name, domain bank consists of the domains of the dataset
# Combined: consist of domains of all the used dataset
# Expanded: adds 19 additional domains based on "Combined"
domain_banks_list = {
    'Empty': [''],
    'PACS':['photo', 'art', 'cartoon', 'sketch'],
    'DomainNet':['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
    'OfficeHome':['art', 'clipart', 'product', 'real'],
    # for VLCS and Terra, since the domains share the same style, we use "Empty" as the "Task" bank.
    'VLCS':[''],
    'Terra':[''],
    'Combined':['', 'photo', 'art', 'cartoon', 'sketch', 'clipart', 'infograph', 'painting', 'quickdraw', 'real','product'],
    'Expanded':['', 'photo', 'art', 'cartoon', 'sketch', 'clipart', 'infograph', 'painting', 'quickdraw', 'real','product', 
    'realistic', 'pixel_style', 'abstract', 'fantasy', 'mysterious', 'science_fiction', 'blurry', 'geometric', 'hazy', 'mosaic', 
    'doodle', 'graffiti', 'art_nouveau', 'expressionism', 'cubism', 'fauvism', 'futurism', 'surrealism', 'minimalism']  

}

def get_prompts(dataset, bank_type):
    """
    generate (class_num x domain_num) prompts from the combinations of categories and domain_bank
    """
    categories = categories_list[dataset]
    domain_bank = domain_banks_list[bank_type] if bank_type!='Task' else domain_banks_list[dataset]

    text_target = [[a] for a in categories]
    text_domain = [[a] for a in domain_bank]

    text_all = []

    for t in text_target:
        for s in text_domain:
            text_all.append([t,s])

    return [f"a {s[0]} photo of a {t[0]}" for t,s in text_all], len(categories), len(domain_bank)
