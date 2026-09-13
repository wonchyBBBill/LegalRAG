"""
pdf_to_markdown.py (Marker 2.0)
=======================================
Converts legal PDFs into clean Markdown format.
Dependencies: pip install marker-pdf>=2.0.0
"""

import os
import argparse
from pathlib import Path
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


def convert_pdfs_to_md(input_folder, output_folder):
    # 1. Create output directory
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # 2. Load Marker models (only once) —— 2.0 用 create_model_dict()
    print("Loading Marker AI models... (this may take a minute)")
    model_dict = create_model_dict()

    # 3. Initialize converter (only once, reuse across files)
    converter = PdfConverter(artifact_dict=model_dict)

    # 4. Process each PDF
    pdf_files = [f for f in os.listdir(input_folder) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in the input folder.")
        return

    for filename in pdf_files:
        pdf_path = os.path.join(input_folder, filename)
        md_filename = Path(filename).stem + ".md"
        md_path = os.path.join(output_folder, md_filename)

        print(f"\nConverting {filename}...")
        try:
            # 2.0: converter 直接返回 rendered 对象
            rendered = converter(pdf_path)

            # 用 text_from_rendered 提取文本、元数据和图片
            full_text, metadata, images = text_from_rendered(rendered)

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            # 可选：保存提取出来的图片
            if images:
                img_dir = os.path.join(output_folder, Path(filename).stem + "_images")
                os.makedirs(img_dir, exist_ok=True)
                for img_name, img_data in images.items():
                    img_data.save(os.path.join(img_dir, img_name))
                print(f"  Saved {len(images)} images to {img_dir}")

            print(f"Successfully saved to {md_path}")
        except Exception as e:
            print(f"  Error converting {filename}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert legal PDFs to Markdown for RAG")
    parser.add_argument("--input", default="./law", help="Folder containing PDF files")
    parser.add_argument("--output", default="./law_md", help="Folder to save Markdown files")
    args = parser.parse_args()

    convert_pdfs_to_md(args.input, args.output)
    print("\nConversion complete. You can now use these .md files for your index!")


if __name__ == "__main__":
    main()