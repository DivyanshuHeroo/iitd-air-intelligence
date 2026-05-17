from PIL import Image, ImageDraw, ImageFont
import os

# Create directory if it doesn't exist
os.makedirs("linkedin_assets/carousel", exist_ok=True)

WIDTH, HEIGHT = 1080, 1080
BG_COLOR = "#0D1117"
TEXT_COLOR = "#FFFFFF"
ACCENT_COLOR = "#58A6FF"

def create_slide(filename, title, text_lines, footer="IITD Air Intelligence"):
    img = Image.new("RGB", (WIDTH, HEIGHT), color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 80)
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
        y_pos += 80
        
    # Draw Footer
    draw.text((100, HEIGHT - 100), footer, font=footer_font, fill="#8B949E")
    
    img.save(f"linkedin_assets/carousel/{filename}")

# Slide 1
create_slide(
    "slide_01_title.png",
    "IITD Air Intelligence",
    ["", "Hyperlocal PM2.5 Prediction", "for Delhi-NCR", "", "", "ML + Streamlit + FastAPI", "+ Route Exposure Advisor"]
)

# Slide 2
create_slide(
    "slide_02_problem.png",
    "Why This Project?",
    ["Delhi-NCR air pollution changes", "by time and location.", "", "This project explores ML-based", "hyperlocal PM2.5 prediction.", "", "Focus: IIT Delhi / South Delhi"]
)

# Slide 3
create_slide(
    "slide_03_pipeline.png",
    "ML Pipeline",
    ["✅ Data Cleaning", "✅ Feature Engineering", "✅ Time-Based Split", "✅ Baseline Comparison", "✅ Model Evaluation", "✅ Deployment", "", "* Temporal + spatial + weather", "* Lag + expanding mean features"]
)

# Slide 4
create_slide(
    "slide_04_performance.png",
    "Model Benchmarking",
    ["Target: Next-Hour PM2.5 Prediction", "", "Best Model: Ridge Regression", "", "Improvement over", "Persistence Baseline:", "➡️ RMSE: -19.6%", "➡️ MAE: -16.4%", "", "Category Classification: 62% Accuracy"]
)

# Slide 5
create_slide(
    "slide_05_dashboard.png",
    "Interactive Dashboard",
    ["Streamlit app for PM2.5 prediction", "", "Features:", "- IIT Delhi default demo", "- Location-based prediction", "- Real-time inference via FastAPI"]
)

# Slide 6
create_slide(
    "slide_06_route_advisor.png",
    "Route Exposure Advisor",
    ["Compares routes around", "IIT Delhi & South Delhi.", "", "Estimates predicted PM2.5 exposure.", "", "Turns ML prediction into", "decision support!"]
)

print("Carousel slides generated.")
