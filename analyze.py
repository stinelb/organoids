import click
import exif
import json
import os
import shapely
import tqdm
from PIL import Image  # For pixel data
import easyocr  # Import EasyOCR
import re  # Import regex module for stricter filtering

from ..utils import end, start, status

@click.group()
def _analyze():
    pass

@_analyze.command()
@click.argument("directory", type=click.Path(exists=True), nargs=-1)
@click.option("--ext", default=".json", help="File extension to search for (default: .json)")
@click.option("--exif-ext", default=".jpg", help="File extension to extract EXIF from (default: .jpg)")
def analyze(directory, ext, exif_ext):
    start("Scanning for files")
    todo = list(directory)
    found = []
    
    # Initialize EasyOCR Reader (English is used here, adjust languages as needed)
    reader = easyocr.Reader(['en'])
    
    while todo:
        dir = todo.pop()
        for entry in os.listdir(dir):
            entry_path = os.path.join(dir, entry)
            if os.path.isdir(entry_path):
                todo.append(entry_path)
            elif entry.endswith(ext):
                found.append(entry_path)
    status(len(found), end='')
    end()
    
    data = {}
    for entry in tqdm.tqdm(found, desc="Parsing JSON and checking for shapes data"):
        with open(entry, 'rt') as f:
            d = json.load(f)
            if "shapes" in d:
                data[entry] = d
            else:
                print(f"Warning: {entry} has no shapes data")
    status(len(data), end='')
    end()
    
    start("Computing areas and detecting text")
    for entry, d in tqdm.tqdm(data.items(), desc="Computing areas"):
        image_path = os.path.join(os.path.dirname(entry), d["imagePath"])
        
        pix_size = None
        mag = None
        exif_found = False
        
        # Check if the image has EXIF data
        if image_path.endswith(exif_ext):
            e = exif.Image(open(image_path, 'rb'))
            if e.has_exif and hasattr(e, "user_comment"):
                exif_found = True
                user_comment = json.loads(e.user_comment)
                pix_size = user_comment["effectivePixelSize"] / 10**6
                mag = user_comment["objectiveMag"]
            else:
                print(f"Warning: {image_path} referenced from {entry} has no valid EXIF data")

        # If no valid EXIF data, use pixel-based fallback
        if not exif_found:
            with Image.open(image_path) as img:
                width, height = img.size  # Get image dimensions
                pixel_area = width * height  # Calculate total pixel area
                pix_size = 1  # Assume each pixel has a size of 1 unit if no EXIF data
                mag = 1  # Default to no magnification if not specified
                print(f"Using pixel dimensions for {image_path}: {width}x{height} pixels")
                
                # Use EasyOCR to read text from the image
                results = reader.readtext(img, detail=1)  # Set detail to 1 for bounding box info
                
                # Dictionary to store text and associated coordinates, and set to check for unique numbers
                ocr_text_map = {}
                unique_numbers = set()  # Track detected numbers from 1 to 12
                
                # Regular expression pattern for numbers between 1 and 12
                pattern = re.compile(r'^(1[0-2]|[1-9])$')
                
                # Process OCR results for unique numbers in the range 1-12
                for (bbox, text, _) in results:
                    # Attempt to match the text strictly to numbers 1-12
                    if pattern.match(text):  # Check if text matches "1" to "12"
                        number = int(text)
                        # Ensure uniqueness by skipping if the number is already in unique_numbers
                        if number not in unique_numbers:
                            unique_numbers.add(number)
                            # Calculate center of bounding box
                            (top_left, top_right, bottom_right, bottom_left) = bbox
                            center_x = (top_left[0] + bottom_right[0]) / 2
                            center_y = (top_left[1] + bottom_right[1]) / 2
                            ocr_text_map[number] = (center_x, center_y)
                        else:
                            print(f"Duplicate detected and skipped: {number}")
                    else:
                        # If text does not match our pattern, it is ignored
                        print(f"Ignored non-matching OCR result: {text}")
        
        # Compute areas and correlate OCR text with shapes
        for s in d["shapes"]:
            poly = shapely.geometry.Polygon(s["points"])
            area = (poly.area * pix_size) / mag
            
            # Find closest OCR text to shape center, if applicable
            shape_center = poly.centroid
            closest_text = None
            min_distance = float("inf")
            
            # Compare shape center to each OCR-detected text center to find the closest
            for number, (ocr_x, ocr_y) in ocr_text_map.items():
                distance = ((ocr_x - shape_center.x)**2 + (ocr_y - shape_center.y)**2)**0.5
                if distance < min_distance:
                    min_distance = distance
                    closest_text = number

            # Update shape label to include only the OCR-detected number and area
            area_label = f"Area: {area:.2f} mm²" if exif_found else f"Area: {area:.2f} pixels²"
            label_text = area_label  # Start with just the area

            if closest_text:
                label_text = f"{closest_text} - {area_label}"  # Only OCR number and area

            s["label"] = label_text  # Set label to OCR number and area only
            s["area"] = area

    end()
    start("Writing areas to disk")
    for entry, d in tqdm.tqdm(data.items(), desc="Writing areas to disk"):
        with open(entry, 'wt') as f:
            json.dump(d, f, indent=2)
    end()
