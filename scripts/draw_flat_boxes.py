# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def annotate_image(json_data, image_path, output_image_path):
    # Load the JSON data
    data = json.loads(json_data)

    # Load the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    font_height = 32
    border = 5

    # Optionally load a font. If unavailable, the default PIL font will be used
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_height
        )  # You might need to adjust the path to a valid font and size
    except IOError:
        font = ImageFont.load_default()

    # Process each annotation and draw rectangles and text
    for item in data["ann"]:
        bbox = item["bbox"]
        category_id = item["category_id"]
        content = item["content"]

        is_tbc = "<tbc>" in content

        box_color = "green" if is_tbc else "red"

        if is_tbc:
            label = f"{category_id} TBC"
        elif category_id == "Text":
            label = ""
        else:
            label = category_id

        # Draw the rectangle with the specified color
        # Let's make the box larger by `border` pixels
        larger_bbox = [bbox[0] - border, bbox[1] - border, bbox[2] + border, bbox[3] + border]

        draw.rectangle(larger_bbox, outline=box_color, width=border)

        # Draw text (category_id) near the bbox
        text_position = (bbox[0], bbox[1] - font_height - border)
        draw.text(text_position, label, font=font, fill="blue")

    # Save the modified image
    image.save(output_image_path)
    print(f"Annotated image saved as {output_image_path}")


def create_pdf_from_images(folder_path, output_pdf_path):
    def natural_keys(text):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(text))]

    # Create a Path object for the folder
    folder = Path(folder_path)

    image_files = list(folder.glob("*_annotated.png"))

    # Sort files using natural sorting
    image_files.sort(key=lambda x: natural_keys(x.stem))

    # Initialize a list to hold the images
    images = []

    # Open and append each image to the list
    for image_file in image_files:
        image = Image.open(image_file)
        images.append(image.convert("RGB"))  # Convert images to RGB color mode

    # Save all images as a single PDF file
    if images:
        images[0].save(output_pdf_path, save_all=True, append_images=images[1:])
        print(f"PDF created successfully at {output_pdf_path}")
    else:
        print("No images found to create PDF.")


def main(folder_path):
    # Create a Path object for the folder
    folder = Path(folder_path)

    # Glob for all image files and corresponding JSON files
    for image_file in folder.glob("nvpdftex_page_*.png"):
        json_file = image_file.with_suffix(".flat_layout.json")

        if "annotated.png" in image_file.name:
            continue

        # Check if the JSON file exists
        if not json_file.exists():
            print(f"No corresponding JSON file {str(json_file)} found for {image_file.name}")
            continue

        # Read the JSON file
        with open(json_file, "r") as file:
            json_data = file.read()

        # Set output image path
        output_image_path = image_file.with_name(image_file.stem + "_annotated.png")

        # Annotate the image
        annotate_image(json_data, str(image_file), str(output_image_path))

    # Create a PDF from all annotated images
    output_pdf_path = folder / "annotated_images.pdf"
    create_pdf_from_images(folder, str(output_pdf_path))


# Example usage
if __name__ == "__main__":
    assert len(sys.argv) == 2, "Please provide the folder path as the first argument"
    folder_path = sys.argv[1]

    main(folder_path)
