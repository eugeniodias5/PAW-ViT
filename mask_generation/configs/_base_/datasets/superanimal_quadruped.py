dataset_info = dict(
    dataset_name='superanimal_quadruped',
    paper_info=dict(
        author='Ye, Shaokai and Filippova, Anastasiia and Lauer, Jessy and Schneider, Steffen and Vidal, Maxime and Qiu, Tian and Mathis, Alexander and Mathis, Mackenzie Weygandt',
        title='SuperAnimal pretrained pose estimation models for behavioral analysis',
        container='Nature Communications',
        year='2024',
        homepage='https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html',
    ),
    keypoint_info={
        # --- FACE/MOUTH REGION (dark blue) ---
        2:  dict(name='lower_jaw',       id=2,  color=[0,   0, 153],  type='face',  swap=''),
        4:  dict(name='mouth_end_left',  id=4,  color=[0,   0, 153],  type='face',  swap='mouth_end_right'),
        3:  dict(name='mouth_end_right', id=3,  color=[0,   0, 153],  type='face',  swap='mouth_end_left'),
        1:  dict(name='upper_jaw',       id=1,  color=[0,   0, 153],  type='face',  swap=''),

        # --- NOSE (medium/bright blue) ---
        0:  dict(name='nose',            id=0,  color=[0,   0, 255],  type='face',  swap=''),

        # --- EYES/EARS/ANTLERS (teal/green) ---
        5:  dict(name='right_eye',         id=5,  color=[0, 200, 100], type='face', swap='left_eye'),
        6:  dict(name='right_earbase',     id=6,  color=[0, 200, 100], type='face', swap='left_earbase'),
        7:  dict(name='right_earend',      id=7,  color=[0, 200, 100], type='face', swap='left_earend'),
        8:  dict(name='right_antler_base', id=8,  color=[0, 200, 100], type='face', swap='left_antler_base'),
        9:  dict(name='right_antler_end',  id=9,  color=[0, 200, 100], type='face', swap='left_antler_end'),
        10: dict(name='left_eye',          id=10, color=[0, 200, 100], type='face', swap='right_eye'),
        11: dict(name='left_earbase',      id=11, color=[0, 200, 100], type='face', swap='right_earbase'),
        12: dict(name='left_earend',       id=12, color=[0, 200, 100], type='face', swap='right_earend'),
        13: dict(name='left_antler_base',  id=13, color=[0, 200, 100], type='face', swap='right_antler_base'),
        14: dict(name='left_antler_end',   id=14, color=[0, 200, 100], type='face', swap='right_antler_end'),

        # --- NECK (green) ---
        15: dict(name='neck_base', id=15, color=[0, 255, 0],   type='body', swap=''),
        16: dict(name='neck_end',  id=16, color=[0, 255, 0],   type='body', swap=''),

        # --- THROAT (lime / yellow-green) ---
        17: dict(name='throat_base', id=17, color=[153, 255, 51], type='body', swap=''),
        18: dict(name='throat_end',  id=18, color=[153, 255, 51], type='body', swap=''),

        # --- BACK/TORSO (yellow) ---
        19: dict(name='back_base',   id=19, color=[255, 255, 0],   type='body', swap=''),
        21: dict(name='back_middle', id=21, color=[255, 255, 0],   type='body', swap=''),
        20: dict(name='back_end',    id=20, color=[255, 255, 0],   type='body', swap=''),

        # --- TAIL (orange) ---
        22: dict(name='tail_base', id=22, color=[255, 165, 0], type='body', swap=''),
        23: dict(name='tail_end',  id=23, color=[255, 165, 0], type='body', swap=''),

        # --- FRONT LEFT LIMB (red/orange) ---
        24: dict(name='front_left_thigh', id=24, color=[255,  80,  0], type='limb', swap='front_right_thigh'),
        25: dict(name='front_left_knee', id=25, color=[255,  80,  0], type='limb', swap='front_right_knee'),
        26: dict(name='front_left_paw',  id=26, color=[255,   0,  0], type='limb', swap='front_right_paw'),

        # --- FRONT RIGHT LIMB (pink) ---
        27: dict(name='front_right_thigh', id=27, color=[255, 105, 180], type='limb', swap='front_left_thigh'),
        28: dict(name='front_right_knee', id=28, color=[255, 105, 180], type='limb', swap='front_left_knee'),
        29: dict(name='front_right_paw',  id=29, color=[255, 105, 180], type='limb', swap='front_left_paw'),

        # --- BACK LEFT LIMB (magenta) ---
        31: dict(name='back_left_thigh', id=31, color=[255,   0, 255], type='limb', swap='back_right_thigh'),
        33: dict(name='back_left_knee', id=33, color=[255,   0, 255], type='limb', swap='back_right_knee'),
        30: dict(name='back_left_paw',  id=30, color=[255,   0, 255], type='limb', swap='back_right_paw'),

        # --- BACK RIGHT LIMB (purple) ---
        32: dict(name='back_right_thigh', id=32, color=[128,   0, 128], type='limb', swap='back_left_thigh'),
        34: dict(name='back_right_knee', id=34, color=[128,   0, 128], type='limb', swap='back_left_knee'),
        35: dict(name='back_right_paw',  id=35, color=[128,   0, 128], type='limb', swap='back_left_paw'),

        # --- UNDERSIDE / BELLY (pink & purple) ---
        36: dict(name='belly_bottom',      id=36, color=[255, 105, 180], type='body', swap=''),
        37: dict(name='body_middle_right', id=37, color=[128,   0, 128], type='body', swap='body_middle_left'),
        38: dict(name='body_middle_left',  id=38, color=[0,     0, 128], type='body', swap='body_middle_right'),
    },

    skeleton_info={
        # MOUTH REGION
        0: dict(link=('lower_jaw','mouth_end_left'),  id=0,  color=[0, 0, 153]),
        1: dict(link=('lower_jaw','mouth_end_right'), id=1,  color=[0, 0, 153]),
        2: dict(link=('lower_jaw','upper_jaw'),       id=2,  color=[0, 0, 153]),

        # NOSE to EYES
        3: dict(link=('nose','left_eye'),             id=3,  color=[0, 200, 100]),
        4: dict(link=('nose','right_eye'),            id=4,  color=[0, 200, 100]),
        5: dict(link=('left_eye','right_eye'),            id=5,  color=[0, 200, 100]),

        # LEFT EYE to EAR & ANTLER
        6: dict(link=('left_eye','left_earbase'),      id=6,  color=[0, 200, 100]),
        7: dict(link=('left_earbase','left_earend'),   id=7,  color=[0, 200, 100]),
        8: dict(link=('left_antler_base','left_antler_end'),  id=8,  color=[0, 200, 100]),
        
        9: dict(link=('left_earbase','right_earbase'),   id=9,  color=[0, 200, 100]),

        # RIGHT EYE to EAR & ANTLER
        10: dict(link=('right_eye','right_earbase'),     id=10, color=[0, 200, 100]),
        11: dict(link=('right_earbase','right_earend'),  id=11, color=[0, 200, 100]),
        12: dict(link=('right_antler_base','right_antler_end'), id=12, color=[0, 200, 100]),

        # NECK & THROAT
        13: dict(link=('neck_base','neck_end'),          id=13, color=[0, 255, 0]),
        14: dict(link=('throat_base','throat_end'),      id=14, color=[153, 255, 51]),
        16: dict(link=('throat_end','front_left_thigh'),       id=16, color=[150, 150, 150]),  # dotted line
        17: dict(link=('throat_end','front_right_thigh'),         id=17, color=[150, 150, 150]),  # dotted line

        # BACK / TORSO
        18: dict(link=('neck_end','back_base'),         id=18, color=[255, 255, 0]),
        19: dict(link=('back_base','back_middle'),       id=19, color=[255, 255, 0]),
        20: dict(link=('back_middle','back_end'),        id=20, color=[255, 255, 0]),

        # “MIDDLE” BODY POINTS
        21: dict(link=('back_middle','body_middle_left'),  id=21, color=[80,   0, 160]),
        22: dict(link=('back_middle','body_middle_right'), id=22, color=[80,   0, 160]),
        23: dict(link=('body_middle_right','belly_bottom'),         id=23, color=[255, 105, 180]),

        # TAIL
        24: dict(link=('back_end','tail_base'), id=24, color=[255, 165, 0]),
        25: dict(link=('tail_base','tail_end'), id=25, color=[255, 165, 0]),

        # FRONT LEFT LIMB
        26: dict(link=('back_base','front_left_thigh'),  id=26, color=[255, 80, 0]),
        27: dict(link=('front_left_thigh','front_left_knee'), id=27, color=[255, 80, 0]),
        28: dict(link=('front_left_knee','front_left_paw'),  id=28, color=[255, 0, 0]),

        # FRONT RIGHT LIMB
        29: dict(link=('back_base','front_right_thigh'),    id=29, color=[255, 105, 180]),
        30: dict(link=('front_right_thigh','front_right_knee'), id=30, color=[255, 105, 180]),
        31: dict(link=('front_right_knee','front_right_paw'),  id=31, color=[255, 105, 180]),

        # BACK LEFT LIMB
        32: dict(link=('tail_base','back_left_thigh'),   id=32, color=[255, 0, 255]),
        33: dict(link=('back_left_thigh','back_left_knee'), id=33, color=[255, 0, 255]),
        34: dict(link=('back_left_knee','back_left_paw'),  id=34, color=[255, 0, 255]),

        # BACK RIGHT LIMB
        35: dict(link=('tail_base','back_right_thigh'),    id=35, color=[128, 0, 128]),
        36: dict(link=('back_right_thigh','back_right_knee'), id=36, color=[128, 0, 128]),
        37: dict(link=('back_right_knee','back_right_paw'),  id=37, color=[128, 0, 128]),
    },
    joint_weights = [
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
    ],

    sigmas = [
        0.026, 0.067, 0.067, 0.067, 0.067, 0.025, 0.067, 0.067, 0.067, 0.067, 0.025, 0.067, 0.067, 
        0.067, 0.067, 0.035, 0.067, 0.067, 0.067, 0.067, 0.067, 0.035, 0.067, 0.079, 0.072, 0.062, 
        0.079, 0.072, 0.062, 0.089, 0.107, 0.107, 0.087, 0.087, 0.089, 0.067, 0.067, 0.067, 0.067
    ]
)