from PIL import Image, ImageDraw, ImageFont
import os
import json
import random
import string

# Configurations
RECEIPT_WIDTH = 750
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
    RECEIPT_WIDTH = 700
    RECEIPT_HEIGHT = 1100
    PADDING = 28
    BG_COLOR = 'white'
    TEXT_COLOR = 'black'
    FONT_PATH = 'cour.ttf'
    FONT_SIZE = 20

    address_line_gap = 8
    item_line_gap = 8
    total_line_gap = 8
    section_gap = 16
    dash_line = "-" * 60

    image = Image.new('RGB', (RECEIPT_WIDTH, RECEIPT_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    y = PADDING

    def get_text_height(font, text):
        bbox = font.getbbox(text)
        return bbox[3] - bbox[1]

    amount_width_max = 140
    amount_x = RECEIPT_WIDTH - PADDING - amount_width_max

    # Merchant and address
    draw.text((PADDING, y), receipt['Merchant'], fill=TEXT_COLOR, font=font)
    y += get_text_height(font, receipt['Merchant']) + address_line_gap

    for addr in receipt['Address'].split(', '):
        draw.text((PADDING, y), addr, fill=TEXT_COLOR, font=font)
        y += get_text_height(font, addr) + address_line_gap

    draw.text((PADDING, y), dash_line, fill=TEXT_COLOR, font=font)
    y += get_text_height(font, dash_line) + section_gap

    # Info section
    draw.text((PADDING, y), f"Slip:  {receipt['Receipt ID']}", fill=TEXT_COLOR, font=font)
    y += get_text_height(font, f"Slip:  {receipt['Receipt ID']}") + address_line_gap
    draw.text((PADDING, y), f"Date:  {receipt['Date']}  {receipt['Time']}", fill=TEXT_COLOR, font=font)
    y += get_text_height(font, f"Date:  {receipt['Date']}  {receipt['Time']}") + section_gap

    draw.text((PADDING, y), dash_line, fill=TEXT_COLOR, font=font)
    y += get_text_height(font, dash_line) + address_line_gap

    # Table header, pixel-aligned
    draw.text((PADDING, y), "Description", fill=TEXT_COLOR, font=font)
    draw.text((amount_x, y), "Amount", fill=TEXT_COLOR, font=font)
    y += get_text_height(font, 'X') + item_line_gap

    # Item lines, pixel-aligned
    items = receipt['Line Items'].split(';')
    for item in items:
        item = item.strip()
        if '@' in item:
            desc, amt = item.rsplit('@', 1)
            desc, amt = desc.strip(), amt.strip()
        else:
            desc, amt = item, ""
        draw.text((PADDING, y), desc, fill=TEXT_COLOR, font=font)
        draw.text((amount_x, y), amt, fill=TEXT_COLOR, font=font)
        y += get_text_height(font, desc) + item_line_gap

    y += item_line_gap // 2
    draw.text((PADDING, y), dash_line, fill=TEXT_COLOR, font=font)
    y += get_text_height(font, dash_line) + section_gap

    # Totals and card number
    for (label, key) in [
        ("Subtotal", 'Subtotal'),
        ("Tax", 'Tax'),
        ("Total", 'Total Amount'),
        ("Card", 'Card Payment'),
        ("Cash", 'Cash Payment'),
    ]:
        draw.text((PADDING, y), label, fill=TEXT_COLOR, font=font)
        draw.text((amount_x, y), receipt.get(key, '-'), fill=TEXT_COLOR, font=font)
        y += get_text_height(font, label) + total_line_gap

        # Card Number (only after Card line, only if present)
        if label == "Card" and receipt.get("Card Number", ""):
            # shift left by 20 pixels (make this bigger if needed)
            card_num_x = RECEIPT_WIDTH - PADDING - draw.textlength(receipt["Card Number"], font=font) - 5
            draw.text((card_num_x, y), receipt["Card Number"], fill=TEXT_COLOR, font=font)
            y += get_text_height(font, receipt["Card Number"]) + 2

    y += section_gap // 2
    draw.text((PADDING, y), dash_line, fill=TEXT_COLOR, font=font)
    y += get_text_height(font, dash_line) + section_gap

    # Footer
    welcome = "Welcome again"
    w = draw.textlength(welcome, font=font)
    draw.text(((RECEIPT_WIDTH - w) // 2, y), welcome, fill=TEXT_COLOR, font=font)
    y += get_text_height(font, welcome) + section_gap // 2

    # Random code
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
    json_file = 'receipts_augmented_with_card.json'
    output_folder = 'receipt_images'
    os.makedirs(output_folder, exist_ok=True)

    # Load receipts JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        receipts = json.load(f)

    for receipt in receipts:
        generate_receipt_image(receipt, output_folder)

if __name__ == '__main__':
    main()
