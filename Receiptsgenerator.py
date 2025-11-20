from PIL import Image, ImageDraw, ImageFont
import os
import json
import random
import string

# Configurations
RECEIPT_WIDTH = 512
RECEIPT_HEIGHT = 1024
PADDING = 20
BG_COLOR = 'white'
TEXT_COLOR = 'black'
FONT_PATH = 'cour.ttf'  # Set this to a monospaced font on your system, e.g., 'C:/Windows/Fonts/cour.ttf'
FONT_SIZE = 20

def get_text_height(font, text):
    bbox = font.getbbox(text)
    height = bbox[3] - bbox[1]
    return height

def draw_dashed_line(draw, y, width, font, dash="-"):
    # Repeat dash for width
    n = int(width/font.getlength(dash))
    line = dash * n
    draw.text((PADDING, y), line, fill=TEXT_COLOR, font=font)

def generate_receipt_image(receipt, output_folder):
    # Spacing controls (adjust as needed)
    address_line_gap = 8  # pixels between address lines
    item_line_gap = 8     # pixels between item rows
    total_line_gap = 8    # pixels between total/summary fields
    section_gap = 14      # pixels between major sections

    RECEIPT_WIDTH = 512
    RECEIPT_HEIGHT = 1024
    PADDING = 20
    BG_COLOR = 'white'
    TEXT_COLOR = 'black'
    FONT_PATH = 'cour.ttf'   # Make sure you have a monospaced font TTF on your system
    FONT_SIZE = 20

    dash_line = "-" * 44

    image = Image.new('RGB', (RECEIPT_WIDTH, RECEIPT_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    y = PADDING

    # Merchant and address
    draw.text((PADDING, y), receipt['Merchant'], fill=TEXT_COLOR, font=font)
    y += get_text_height(font, receipt['Merchant']) + address_line_gap

    for addr in receipt['Address'].split(', '):
        draw.text((PADDING, y), addr, fill=TEXT_COLOR, font=font)
        y += get_text_height(font, addr) + address_line_gap

    # Dash line directly under address
    draw.text((PADDING, y), dash_line, fill=TEXT_COLOR, font=font)
    y += get_text_height(font, dash_line) + section_gap

    # Receipt info
    draw.text((PADDING, y), f"Slip:  {receipt['Receipt ID']}", fill=TEXT_COLOR, font=font)
    y += get_text_height(font, f"Slip:  {receipt['Receipt ID']}") + address_line_gap
    draw.text((PADDING, y), f"Date:  {receipt['Date']}  {receipt['Time']}", fill=TEXT_COLOR, font=font)
    y += get_text_height(font, f"Date:  {receipt['Date']}  {receipt['Time']}") + section_gap

    # Dash line before items
    draw.text((PADDING, y), dash_line, fill=TEXT_COLOR, font=font)
    y += get_text_height(font, dash_line) + address_line_gap

    # Item header
    draw.text((PADDING, y), "Description".ljust(30) + "Amount".rjust(14), fill=TEXT_COLOR, font=font)
    y += get_text_height(font, 'X') + item_line_gap

    # Line items (with even spacing)
    items = receipt['Line Items'].split(';')
    for item in items:
        item = item.strip()
        if '@' in item:
            desc, amt = item.rsplit('@', 1)
            desc, amt = desc.strip(), amt.strip()
        else:
            desc, amt = item, ""
        draw.text((PADDING, y), desc.ljust(30) + amt.rjust(14), fill=TEXT_COLOR, font=font)
        y += get_text_height(font, desc) + item_line_gap

    y += item_line_gap // 2
    # Dash line below items
    draw.text((PADDING, y), dash_line, fill=TEXT_COLOR, font=font)
    y += get_text_height(font, dash_line) + section_gap

    # Totals/summary section (with spacing)
    for (label, key) in [
        ("Subtotal", 'Subtotal'),
        ("Tax", 'Tax'),
        ("Total", 'Total Amount'),
        ("Card", 'Card Payment'),
        ("Cash", 'Cash Payment'),
    ]:
        draw.text((PADDING, y), label.ljust(29) + receipt.get(key, '-').rjust(15), fill=TEXT_COLOR, font=font)
        y += get_text_height(font, label) + total_line_gap

    y += section_gap // 2
    # Dash line at end
    draw.text((PADDING, y), dash_line, fill=TEXT_COLOR, font=font)
    y += get_text_height(font, dash_line) + section_gap

    # Footer -- Welcome/Random Codes
    welcome = "Welcome again"
    w = draw.textlength(welcome, font=font)
    draw.text(((RECEIPT_WIDTH - w) // 2, y), welcome, fill=TEXT_COLOR, font=font)
    y += get_text_height(font, welcome) + section_gap // 2

    # Random bottom code (hash + AT code)
    code_chars = string.digits + string.ascii_uppercase
    rand_code = '#' + ''.join(random.choices(code_chars, k=18))
    trans_code = '{AT' + ''.join(random.choices(string.digits, k=16)) + '}'
    w1 = draw.textlength(rand_code, font=font)
    w2 = draw.textlength(trans_code, font=font)
    draw.text(((RECEIPT_WIDTH - w1) // 2, y), rand_code, fill=TEXT_COLOR, font=font)
    y += get_text_height(font, rand_code)
    draw.text(((RECEIPT_WIDTH - w2) // 2, y), trans_code, fill=TEXT_COLOR, font=font)

    # Save image
    output_filename = f"{receipt['Receipt ID']}.png"
    output_path = os.path.join(output_folder, output_filename)
    image.save(output_path)
    print(f"Saved receipt image: {output_path}")

def main():
    json_file = 'receipts_complete.json'
    output_folder = 'receipt_images'
    os.makedirs(output_folder, exist_ok=True)

    # Load receipts JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        receipts = json.load(f)

    for receipt in receipts:
        generate_receipt_image(receipt, output_folder)

if __name__ == '__main__':
    main()
