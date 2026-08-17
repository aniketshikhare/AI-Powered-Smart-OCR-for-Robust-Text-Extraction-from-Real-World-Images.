"""
Create test images for OCR demonstration
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_test_image(output_path, text, size=(800, 400)):
    """Create a test image with text"""
    # Create white background
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font
    try:
        # Try to use a common font
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        # Fall back to default font
        font = ImageFont.load_default()
    
    # Add text to image
    draw.text((50, 150), text, fill='black', font=font)
    
    # Save the image
    img.save(output_path)
    print(f"Created test image: {output_path}")

def create_multiple_test_images():
    """Create multiple test images with different text"""
    output_dir = "data/images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test images
    test_texts = [
        ("Hello World - AI Powered OCR", "test1.jpg"),
        ("This is a sample text for extraction", "test2.jpg"),
        ("Multi-language support for OCR systems", "test3.jpg"),
        ("Robust text extraction from images", "test4.jpg"),
        ("1234567890 - Numbers and Text", "test5.jpg")
    ]
    
    for text, filename in test_texts:
        output_path = os.path.join(output_dir, filename)
        create_test_image(output_path, text)
    
    print(f"\nCreated {len(test_texts)} test images in {output_dir}")

if __name__ == "__main__":
    create_multiple_test_images()