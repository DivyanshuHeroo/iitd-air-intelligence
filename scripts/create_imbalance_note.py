from PIL import Image, ImageDraw, ImageFont
import os

os.makedirs("linkedin_assets/performance", exist_ok=True)
WIDTH, HEIGHT = 1080, 500
BG_COLOR = "#0D1117"
TEXT_COLOR = "#FFFFFF"

img = Image.new("RGB", (WIDTH, HEIGHT), color=BG_COLOR)
draw = ImageDraw.Draw(img)

try:
    font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 45)
    font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 35)
except:
    font_large = ImageFont.load_default()
    font_small = ImageFont.load_default()

title = "A Note on Class Imbalance"
draw.text((50, 50), title, font=font_large, fill="#FF5555")

text_lines = [
    "Binary unsafe-air accuracy is high (99.4%)",
    "but heavily affected by class imbalance in Delhi.",
    "",
    "To evaluate the model honestly, the project",
    "highlights 3-class severity and adjacent-category",
    "metrics instead."
]

y_pos = 150
for line in text_lines:
    draw.text((50, y_pos), line, font=font_small, fill=TEXT_COLOR)
    y_pos += 50
    
img.save("linkedin_assets/performance/class_imbalance_note.png")
print("Saved note image.")
