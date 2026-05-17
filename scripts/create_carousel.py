from PIL import Image, ImageDraw, ImageFont
import os

# Create directory if it doesn't exist
os.makedirs("linkedin_assets/carousel", exist_ok=True)

WIDTH, HEIGHT = 1080, 1080
BG_COLOR = "#0D1117"
TEXT_COLOR = "#FFFFFF"
ACCENT_COLOR = "#58A6FF"

def create_slide(filename, title, text_lines, footer="Streamlit + FastAPI + Route Exposure Advisor"):
    img = Image.new("RGB", (WIDTH, HEIGHT), color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 75)
        text_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 45)
        footer_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 35)
    except:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        footer_font = ImageFont.load_default()

    # Draw Title
    draw.text((100, 150), title, font=title_font, fill=ACCENT_COLOR)
    
    # Draw Text
    y_pos = 350
    for line in text_lines:
        draw.text((100, y_pos), line, font=text_font, fill=TEXT_COLOR)
        y_pos += 70
        
    # Draw Footer
    draw.text((100, HEIGHT - 100), footer, font=footer_font, fill="#8B949E")
    
    img.save(f"linkedin_assets/carousel/{filename}")

# Slide 1
create_slide(
    "slide_01_title.png",
    "IITD Air Intelligence",
    ["", "ML for Hyperlocal PM2.5 Prediction", "in Delhi-NCR", "", "", ""]
)

# Slide 2
create_slide(
    "slide_02_problem.png",
    "Why This Project?",
    ["Delhi-NCR air pollution changes", "by time and location.", "", "This project explores hyperlocal", "PM2.5 prediction around", "IIT Delhi and South Delhi."]
)

# Slide 3
create_slide(
    "slide_03_pipeline.png",
    "ML Pipeline",
    ["✅ Data Cleaning", "✅ Feature Engineering", "✅ Time-Based Validation", "✅ Model Benchmarking", "✅ Dashboard + API", "", "* Temporal + spatial + weather", "* Lag + rolling features"]
)

# Slide 4
create_slide(
    "slide_04_classification_results.png",
    "Classification Results",
    ["Target: Next-Hour PM2.5 Category", "", "➡️ 6-class exact: 59.3%", "➡️ Adjacent-category: 95.6%", "➡️ 3-class severity: 84.9%", "", "Note: Fine-grained PM2.5 categories", "are hard; adjacent/severity metrics", "better capture practical trend prediction."]
)

# Slide 5
create_slide(
    "slide_05_dashboard.png",
    "Interactive Website",
    ["Streamlit dashboard", "", "Features:", "- IIT Delhi default demo", "- PM2.5 + category interpretation", "- Real-time inference via FastAPI"]
)

# Slide 6
create_slide(
    "slide_06_route_advisor.png",
    "Route Exposure Advisor",
    ["Compares routes around IIT Delhi", "", "Estimates predicted exposure", "", "Turns ML output into decision support!"]
)

print("Carousel slides generated.")
