#!/usr/bin/env python3

import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv('.env')

def extract_brand(description):
    if not description:
        return None
    if "Brand:" in description:
        brand_part = description.split("Brand:")[1].split("|")[0].strip()
        return brand_part
    return None

def get_clean_names_prompt(product_names, brands, category):
    names_with_brands = []
    for i, (name, brand) in enumerate(zip(product_names, brands)):
        if brand:
            names_with_brands.append(f"{i+1}. Name: {name} | Brand: {brand}")
        else:
            names_with_brands.append(f"{i+1}. Name: {name}")
    names_list = "\n".join(names_with_brands)
    
    return f"""Clean these e-commerce product names by keeping only: Brand + Model/Series + Product Type.

Product Category: {category}

CRITICAL RULES (YOU MUST FOLLOW THESE - NO EXCEPTIONS):
1. Uniqueness: The first three words of each cleaned name must be unique across all products. If two products would have the same first three words, adjust one of them to make them distinct.
2. Format: Brand first, then Model/Series, then Product Type. Title Case (first letter of each word capitalized), spaces only, typically 3-8 words
3. The cleaned name MUST start with the brand name (use the Brand field if the original name doesn't start with a brand). **The brand name's first letter MUST be capitalized, even if the original brand name starts with lowercase** 
4. The cleaned name MUST end with the product type that matches the category "{category}". 
5. Remove: colors, special symbols (®, ™, –, commas, parentheses), functional descriptions, feature lists, "by [brand]" phrases, and redundant words

Examples:
Input: 1. Name: Ingenuity 3D Mini Convenience Stroller – Lightweight Stroller with Compact Fold, Multi-Position Recline, Canopy with Pop Out Sun Visor and More – Umbrella Stroller for Travel and More, Gray | Brand: Ingenuity | Category: baby stroller
Output: Ingenuity 3D Mini Convenience Stroller

Input: 2. Name: 12 Color Cream Lip Gloss, 2025 New Cream Texture Lipstick | Brand: Wegodal | Category: lipstick
Output: Wegodal 12 Color Cream Lipstick

Input: 3. Name: Logitech K400 Plus Wireless Touch TV Keyboard With Easy Media Control and Built-in Touchpad, HTPC Keyboard for PC-connected TV, Windows, Android, ChromeOS, Laptop, Tablet - Black | Brand: Logitech | Category: keyboard
Output: Logitech K400 Plus Keyboard

Product names to clean (Category: {category}):
{names_list}

Return the cleaned names as a JSON object with keys "1", "2", "3", etc. (matching the numbers above), where each value is the cleaned product name starting with the brand. Only return the JSON object, no additional text.

Example output format:
{{
  "1": "Ingenuity 3D Mini Convenience Stroller",
  "2": "Wegodal 12 Color Cream Lipstick",
  "3": "Logitech K400 Plus Keyboard"
}}"""

def extract_category_from_path(file_path):
    filename = Path(file_path).stem
    category = filename.replace('_', ' ')
    return category

def clean_product_names_with_chatgpt(product_names, brands, category, api_key):
    client = OpenAI(api_key=api_key)
    prompt = get_clean_names_prompt(product_names, brands, category)
    print(f"📤 Sending {len(product_names)} product names to ChatGPT...")
    
    generated_text = None
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        generated_text = response.choices[0].message.content.strip()
        
        if "```json" in generated_text:
            generated_text = generated_text.split("```json")[1].split("```")[0].strip()
        elif "```" in generated_text:
            generated_text = generated_text.split("```")[1].split("```")[0].strip()
        
        cleaned_names_dict = json.loads(generated_text)
        cleaned_names = [cleaned_names_dict[str(i+1)] for i in range(len(product_names))]
        
        print(f"✅ Received {len(cleaned_names)} cleaned names")
        return cleaned_names
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON response: {e}")
        if generated_text:
            print(f"Response was: {generated_text[:500]}")
        raise
    except Exception as e:
        print(f"❌ Error calling ChatGPT API: {e}")
        if generated_text:
            print(f"Response was: {generated_text[:500] if len(generated_text) > 500 else generated_text}")
        raise

def process_jsonl_file(file_path, api_key):
    print(f"\n📄 Processing: {file_path}")
    
    products = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                products.append(json.loads(line))
    
    if not products:
        print(f"⚠️  No products found in {file_path}")
        return
    
    print(f"📊 Found {len(products)} products")
    original_names = [product['Name'] for product in products]
    brands = [extract_brand(product.get('Description', '')) for product in products]
    category = extract_category_from_path(file_path)
    
    try:
        cleaned_names = clean_product_names_with_chatgpt(original_names, brands, category, api_key)
    except Exception as e:
        print(f"❌ Failed to clean names: {e}")
        return
    
    print("📝 Updating names:")
    for i, product in enumerate(products):
        product['Name'] = cleaned_names[i]
        if len(products) <= 6 or i < 3 or i >= len(products) - 3:
            print(f"  {i+1}. {original_names[i][:50]}... → {cleaned_names[i]}")
        elif i == 3:
            print(f"  ... ({len(products) - 6} more changes) ...")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for product in products:
            f.write(json.dumps(product, ensure_ascii=False) + '\n')
    
    print(f"✅ Updated {file_path}")

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Check .env file.")
    
    base_dir = Path('/multi-stealth-rank/data_new_simplified')
    jsonl_files = [f for f in base_dir.rglob('*.jsonl') if not f.name.endswith('.backup')]
    
    if not jsonl_files:
        print("⚠️  No JSONL files found in data_new_simplified")
        return
    
    print(f"🔍 Found {len(jsonl_files)} JSONL files to process")
    
    for jsonl_file in jsonl_files:
        try:
            process_jsonl_file(jsonl_file, api_key)
        except Exception as e:
            print(f"❌ Error processing {jsonl_file}: {e}")
            continue
    
    print("\n✅ All files processed!")

if __name__ == "__main__":
    main()
