import numpy as np

# Parameters for drawing figure.
display_input_size = (10, 10)
overall_fig_size = (18, 24)

# Parameters for drawing figure.
line_thickness = 1
fig_size_w = 35
# fig_size_h = min(max(5, int(len(category_names) / 2.5) ), 10)
mask_color = 'red'
alpha = 0.5

#@markdown **Global constants:** pick and place objects, colors, workspace bounds.
PIXEL_SIZE = 0.00267857 # The size of each pixel in real-world units (such as meters)

BOUNDS = np.float32([[-0.25, 0.25], [-0.75, -0.25], [0, 0.15]])  # (X, Y, Z) (width, depth, height)

# [-6.28318530718, -6.28318530718, -3.14159265359, -6.28318530718, -6.28318530718, -6.28318530718]
# [6.28318530718, 6.28318530718, 3.14159265359, 6.28318530718, 6.28318530718, 6.28318530718]
JOINT_LIMITS = [[- 2 * np.pi, 2 * np.pi], [- 2 * np.pi, 2 * np.pi], [-np.pi, np.pi],
				[- 2 * np.pi, 2 * np.pi], [- 2 * np.pi, 2 * np.pi], [- 2 * np.pi, 2 * np.pi]]

# ALLOWED_POSITION = [[0.2, 2.4], [-2.4, -0.3], [0.5, 3.0], [-2.9, -0.9], [4.711, 4.712], [0.0, 0.01], [0.0, 0.06]]  # suction
ALLOWED_POSITION = [[0.3, 2.2], [-2.7, -0.5], [0.0, 2.9], [-2.9, -1.5], [4.711, 4.712], [-0.9, 3.7], [0.0, 0.83]]

ALLOWED_VELOCITY = [[-3.1, 3.1], [-3.1, 3.1], [-3.1, 3.1], [-3.1, 3.1], [-3.1, 3.1], [-3.1, 3.1], [-3.1, 3.1]]

COLORS = {
	'black': (0, 0, 0),
	'white': (255, 255, 255),
	'blue': (78 / 255, 121 / 255, 167 / 255, 255 / 255),
	'red': (255 / 255, 87 / 255, 89 / 255, 255 / 255),
	'green': (89 / 255, 169 / 255, 79 / 255, 255 / 255),
	'orange': (242 / 255, 142 / 255, 43 / 255, 255 / 255),
	'yellow': (237 / 255, 201 / 255, 72 / 255, 255 / 255),
	'purple': (176 / 255, 122 / 255, 161 / 255, 255 / 255),
	'pink': (255 / 255, 157 / 255, 167 / 255, 255 / 255),
	'cyan': (118 / 255, 183 / 255, 178 / 255, 255 / 255),
	'brown': (156 / 255, 117 / 255, 95 / 255, 255 / 255),
	'gray': (186 / 255, 176 / 255, 172 / 255, 255 / 255),
}

PICK_TARGETS = {
	'red apple': None,
	'red strawberry': None,
	'yellow banana': None,
	'pink peach': None,
	'white egg': None,
	'green pear': None,
	'orange orange': None,
	'yellow lemon': None,
	'black spoon': None,
	'black spatula': None,
	'black knife': None,
	'red gelatin_box': None,

	'blue block': None,
	'red block': None,
	'green block': None,
	'orange block': None,
	'yellow block': None,
	'purple block': None,
	'pink block': None,
	'cyan block': None,
	'brown block': None,
	'gray block': None,
}

PLACE_TARGETS = {
	'blue block': None,
	'red block': None,
	'red gelatin_box': None,
	'green block': None,
	'orange block': None,
	'yellow block': None,
	'purple block': None,
	'pink block': None,
	'cyan block': None,
	'brown block': None,
	'gray block': None,

	'blue bowl': None,
	'red bowl': None,
	'green bowl': None,
	'orange bowl': None,
	'yellow bowl': None,
	'purple bowl': None,
	'pink bowl': None,
	'cyan bowl': None,
	'brown bowl': None,
	'gray bowl': None,

	'top left corner': (-0.25, -0.25, 0),
	'top side': (0, -0.25, 0),
	'top right corner': (0.25, -0.25, 0),
	'left side': (-0.25, -0.5, 0),
	'middle': (0, -0.5, 0),
	'right side': (0.25, -0.5, 0),
	'bottom left corner': (-0.25, -0.75, 0),
	'bottom side': (0, -0.75, 0),
	'bottom right corner': (0.25, -0.75, 0),
}

fixed_destination = {
	'top left corner': (-0.25, -0.25, 0),
	'top side': (0, -0.25, 0),
	'top right corner': (0.25, -0.25, 0),
	'left side': (-0.25, -0.5, 0),
	'middle': (0, -0.5, 0),
	'right side': (0.25, -0.75, 0),
	'bottom left corner': (-0.25, -0.75, 0),
	'bottom side': (0, -0.75, 0),
	'bottom right corner': (0.25, -0.75, 0),
}


instruction_form = [
	"Touch the {}",
	"Pick the {} up",
	"Move the {} to the {}",
	"Put the {} onto the {}",
]

action_state = {
	'Touch': '0',
	'Pick': '1',
	'Move': '2',
	'Put': '3',
}

object_type = {
	'red apple': 0,
	'red strawberry': 0,
	'yellow banana': 0,
	'pink peach': 0,
	'white egg': 0,
	'green pear': 0,
	'orange orange': 0,
	'yellow lemon': 0,
	'red gelatin_box': 0,

	'blue block': 1,
	'red block': 1,
	'green block': 1,
	'orange block': 1,
	'yellow block': 1,
	'purple block': 1,
	'pink block': 1,
	'cyan block': 1,
	'brown block': 1,
	'gray block': 1,
}  #  soft: 0, hard: 1