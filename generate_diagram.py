import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
import visualkeras
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

# 1. Define the CNN Model architecture as requested by the user
model = Sequential([
    Input(shape=(28, 28, 1), name="Input"),
    
    # First set of Conv2D and MaxPooling2D layers
    Conv2D(32, (3, 3), activation='relu', name="Conv2D_1"),
    MaxPooling2D((2, 2), name="MaxPool2D_1"),
    
    # Second set of Conv2D and MaxPooling2D layers
    Conv2D(64, (3, 3), activation='relu', name="Conv2D_2"),
    MaxPooling2D((2, 2), name="MaxPool2D_2"),
    
    # Flatten layer
    Flatten(name="Flatten"),
    
    # Dense ReLU layer
    Dense(128, activation='relu', name="Dense_ReLU"),
    
    # Final Dense Softmax layer with 10 outputs
    Dense(10, activation='softmax', name="Dense_Softmax")
])

# Keras 3 compatibility patch for visualkeras
for layer in model.layers:
    if not hasattr(layer, 'output_shape'):
        layer.output_shape = layer.output.shape

# Helper to load fonts safely with fallback options
def get_font(font_list, size):
    for name in font_list:
        try:
            return ImageFont.truetype(name, size)
        except IOError:
            continue
    return ImageFont.load_default()

# Load Segoe UI or Arial (common modern sans-serif fonts)
font_title = get_font(["segoeui.ttf", "arial.ttf", "calibri.ttf"], 26)
font_subtitle = get_font(["segoeuib.ttf", "arialbd.ttf", "calibrib.ttf"], 16)
font_header = get_font(["segoeuib.ttf", "arialbd.ttf", "calibrib.ttf"], 18)
font_bold = get_font(["segoeuib.ttf", "arialbd.ttf", "calibrib.ttf"], 14)
font_regular = get_font(["segoeui.ttf", "arial.ttf", "calibri.ttf"], 14)
font_small = get_font(["segoeui.ttf", "arial.ttf", "calibri.ttf"], 12)

# Define configurations for both themes
themes = [
    {
        "name": "dark",
        "output_path": "model_architecture.png",
        "bg_color": (15, 23, 42),          # Slate 900 (#0f172a)
        "text_main": (248, 250, 252),      # Slate 50
        "text_muted": (148, 163, 184),     # Slate 400
        "text_accent": (56, 189, 248),     # Sky 400
        "border_color": (51, 65, 85),      # Slate 700
        "table_header_bg": (30, 41, 59),   # Slate 800
        "table_header_text": (241, 245, 249), # Slate 100
        "table_header_border": (71, 85, 105), # Slate 600
        "row_bg_even": (21, 27, 38),       # Dark Slate alternating
        "row_bg_odd": (15, 23, 42),
        "row_text_main": (226, 232, 240),  # Slate 200
        "row_text_muted": (148, 163, 184), # Slate 400
        "3d_outline": None                 # Default visualkeras outline
    },
    {
        "name": "light",
        "output_path": "model_architecture_white.png",
        "bg_color": (255, 255, 255),       # Pure White (#FFFFFF)
        "text_main": (0, 0, 0),            # Black (#000000)
        "text_muted": (71, 85, 105),       # Slate 600
        "text_accent": (0, 0, 0),          # Black
        "border_color": (0, 0, 0),         # Black
        "table_header_bg": (241, 245, 249),# Light Slate Grey (#F1F5F9)
        "table_header_text": (15, 23, 42), # Dark Slate Grey
        "table_header_border": (0, 0, 0),  # Black
        "row_bg_even": (248, 250, 252),    # Alternating row white/grey
        "row_bg_odd": (255, 255, 255),
        "row_text_main": (0, 0, 0),        # Black
        "row_text_muted": (71, 85, 105),   # Slate 600
        "3d_outline": "#000000"            # Thin black border for blocks
    }
]

for theme in themes:
    print(f"Generating diagram for theme: {theme['name']}...")
    
    # Custom premium colors for the 3D block diagram
    color_map = defaultdict(dict)
    color_map[Conv2D]['fill'] = '#2563eb'       # Royal Blue
    color_map[MaxPooling2D]['fill'] = '#e11d48' # Crimson Red
    color_map[Flatten]['fill'] = '#059669'      # Emerald Green
    color_map[Dense]['fill'] = '#7c3aed'        # Purple
    
    # Add borders to blocks if specified
    if theme["3d_outline"]:
        color_map[Conv2D]['outline'] = theme["3d_outline"]
        color_map[MaxPooling2D]['outline'] = theme["3d_outline"]
        color_map[Flatten]['outline'] = theme["3d_outline"]
        color_map[Dense]['outline'] = theme["3d_outline"]

    # 2. Generate the 3D layered view using visualkeras
    try:
        img_3d = visualkeras.layered_view(
            model, 
            legend=True, 
            font=font_regular, 
            color_map=color_map,
            spacing=20
        )
    except Exception as e:
        print(f"Warning: Failed to generate 3D layered view for {theme['name']} theme:", e)
        img_3d = visualkeras.layered_view(model, legend=True)

    # 3. Create a large, high-res canvas
    canvas_width = 1200
    canvas_height = 800
    canvas = Image.new('RGB', (canvas_width, canvas_height), color=theme["bg_color"])
    draw = ImageDraw.Draw(canvas)

    # Paste the 3D representation (centered horizontally)
    paste_x = (canvas_width - img_3d.width) // 2
    paste_y = 130
    canvas.paste(img_3d, (paste_x, paste_y))

    # Draw bounding box for the 3D visualization section
    draw.rectangle(
        [paste_x - 10, paste_y - 10, paste_x + img_3d.width + 10, paste_y + img_3d.height + 10],
        outline=theme["border_color"],
        width=2
    )

    # Label for the 3D visualization
    draw.text((50, 100), "3D Layer Block Flow (Left to Right)", fill=theme["text_accent"], font=font_subtitle)

    # 4. Draw Header / Titles
    draw.text((50, 25), "CNN Model Architecture", fill=theme["text_main"], font=font_title)

    # Calculate parameter count and other model stats
    total_params = model.count_params()
    draw.text(
        (50, 70), 
        f"Input Shape: (28, 28, 1)  |  Output: 10 Classes  |  Total Trainable Parameters: {total_params:,}", 
        fill=theme["text_muted"], 
        font=font_regular
    )

    # Draw a divider line below the title
    draw.line([(50, 90), (canvas_width - 50, 90)], fill=theme["border_color"], width=1)

    # 5. Draw Layer Details Table
    table_start_y = 350
    draw.text((50, table_start_y), "Detailed Layer Specifications", fill=theme["text_accent"], font=font_header)

    table_y = table_start_y + 35
    headers = ["Layer Name", "Layer Type", "Input Shape", "Output Shape", "Details / Specifications"]
    col_widths = [150, 180, 160, 160, 450]
    col_xs = []
    current_x = 50
    for w in col_widths:
        col_xs.append(current_x)
        current_x += w

    # Draw table header background
    header_height = 40
    draw.rectangle(
        [50, table_y, canvas_width - 50, table_y + header_height],
        fill=theme["table_header_bg"]
    )

    # Write table headers
    for idx, header in enumerate(headers):
        draw.text((col_xs[idx] + 10, table_y + 10), header, fill=theme["table_header_text"], font=font_bold)

    # Draw header bottom line
    draw.line([(50, table_y + header_height), (canvas_width - 50, table_y + header_height)], fill=theme["table_header_border"], width=2)

    # Construct table data dynamically from model
    table_data = []

    # Helper to clean up shape representation
    def format_shape(shape):
        if shape is None:
            return "None"
        if isinstance(shape, list):
            shape = shape[0]
        return str(shape)

    # Input Row (Explicitly added to show the starting point)
    table_data.append((
        "Input_Layer", 
        "InputLayer", 
        "(None, 28, 28, 1)", 
        "(None, 28, 28, 1)", 
        "Raw Grayscale Digit Image Input"
    ))

    # Populate from Keras layers
    for layer in model.layers:
        name = layer.name
        l_type = layer.__class__.__name__
        
        in_shape = format_shape(layer.input_shape) if hasattr(layer, 'input_shape') else "N/A"
        out_shape = format_shape(layer.output_shape) if hasattr(layer, 'output_shape') else "N/A"
        
        # Custom description details based on layer class
        details = ""
        if isinstance(layer, Conv2D):
            act_name = getattr(layer.activation, '__name__', 'linear')
            details = f"{layer.filters} filters, {layer.kernel_size} kernel, activation='{act_name}'"
        elif isinstance(layer, MaxPooling2D):
            details = f"Pool size: {layer.pool_size}, strides: {layer.strides}"
        elif isinstance(layer, Flatten):
            details = f"Flattens 3D tensor to 1D vector"
        elif isinstance(layer, Dense):
            act_name = getattr(layer.activation, '__name__', 'linear')
            details = f"{layer.units} hidden units, activation='{act_name}'"
        else:
            details = f"Trainable params: {layer.count_params():,}"
            
        table_data.append((name, l_type, in_shape, out_shape, details))

    row_height = 35
    current_row_y = table_y + header_height

    # Draw rows with alternating backgrounds
    for row_idx, row in enumerate(table_data):
        bg_color = theme["row_bg_even"] if row_idx % 2 == 0 else theme["row_bg_odd"]
        draw.rectangle(
            [50, current_row_y, canvas_width - 50, current_row_y + row_height],
            fill=bg_color
        )
        
        # Write cells
        for col_idx, cell_value in enumerate(row):
            text_color = theme["row_text_main"] if col_idx != 4 else theme["row_text_muted"]
            draw.text((col_xs[col_idx] + 10, current_row_y + 8), cell_value, fill=text_color, font=font_regular)
            
        # Draw horizontal row separator
        draw.line([(50, current_row_y + row_height), (canvas_width - 50, current_row_y + row_height)], fill=theme["border_color"], width=1)
        current_row_y += row_height

    # Draw vertical lines for the table columns
    table_end_y = current_row_y
    for col_x in col_xs[1:]:
        draw.line([(col_x, table_y), (col_x, table_end_y)], fill=theme["border_color"], width=1)

    # Draw outer border around the whole table
    draw.rectangle(
        [50, table_y, canvas_width - 50, table_end_y],
        outline=theme["border_color"],
        width=1
    )

    # 6. Draw Footer
    draw.text(
        (50, canvas_height - 40), 
        "Digit Recognizer CNN Architecture Diagram  |  Generated using TensorFlow Keras & Visualkeras", 
        fill=theme["text_muted"], 
        font=font_small
    )

    # 7. Save the final diagram
    canvas.save(theme["output_path"], "PNG")
    print(f"Successfully saved {theme['name']} theme to '{theme['output_path']}'")
