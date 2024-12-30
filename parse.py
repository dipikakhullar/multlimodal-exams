import argparse
import requests
import json
import os
import re
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import time
import openai
from openai import (
    OpenAI,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
)


APP_ID = "cohere_app_id"
APP_KEY = "cohere_app_key"

GPT_PROMPT = """
You will be provided a piece of text containing multiple questions with answer options. 
Each question may include images, diagrams, or equations represented by placeholders such as
`<jee_hindi_2024_p2_0.png>`. Extract the multiple-choice questions in {lang} language. Return a json with question number as question_num key, question text as "question_text", choices as a list with "choices". If it includes any placeholder, put it with the "image" key as a list, else leave empty. If the image placeholder are part of choices, put them inside the "choices" key.
Below is an example of the expected output format:

{{
    "question_num": 1,
    "question_text": "What is the capital of France?",
    "choices": ["Paris", "London", "Berlin", "Madrid"],
    "image": [<jee_hindi_2024_p2_0.png>]
}}

{processed_output}
"""
GPT_PROMPT.format(lang="Hindi", processed_output="")
def process_image_with_mathpix(image_path):
    """Process an image using Mathpix API and return the response."""
    r = requests.post(
        "https://api.mathpix.com/v3/text",
        files={"file": open(image_path, "rb")},
        data={
            "options_json": json.dumps({
                "math_inline_delimiters": ["$", "$"],
                "rm_spaces": True,
                "include_line_data": True
            })
        },
        headers={
            "app_id": APP_ID,
            "app_key": APP_KEY,
        }
    )
    return r.json()

def process_line_data(pdf_response, crops_dir):
    """Processes the line data from the response to extract text and insert placeholders for non-text elements."""
    processed_output = []
    img_counter = 0
    
    for page in pdf_response:
        response = pdf_response[page]
        page_output = ""
        for line in response.get("line_data", []):
            if line.get("type") == "page_info":
                continue
            response["exam_name"] = response["exam_name"].split("/")[-1]
            if line.get("cnt") and line.get("type") not in ["text", "math", "column"]:
                output_dir = os.path.join(crops_dir, f'mm_data/{response["exam_name"]}')
                img_prefix = response["exam_name"]
                os.makedirs(output_dir, exist_ok=True)
                image_filename = f"{output_dir}/{img_prefix}_{img_counter}.png"
                crop_image(response["page_image_path"], line, image_filename)
                page_output += f" <{img_prefix}_{img_counter}.png>"
                img_counter += 1
            else:
                page_output += f"{line.get('text', '')} "
        processed_output.append(page_output.strip())
        
    return processed_output

def crop_image(page_image_path, line, cropped_img_path):
    """Crop the image based on the bounding box coordinates."""
    page_image = Image.open(page_image_path)
    cnt = line["cnt"]
    x_min = min(point[0] for point in cnt)
    y_min = min(point[1] for point in cnt)
    x_max = max(point[0] for point in cnt)
    y_max = max(point[1] for point in cnt)
    cropped_image = page_image.crop((x_min, y_min, x_max, y_max))
    cropped_image.save(cropped_img_path)

def process_pdf(pdf_path, json_output_dir):
    """Convert PDF to images and process each page with Mathpix."""
    pages = convert_from_path(pdf_path, dpi=300, fmt="jpeg")
    parent_dir = os.path.dirname(os.path.dirname(pdf_path))
    pages_output_dir = os.path.join(parent_dir, "imgs")
    os.makedirs(pages_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)
    pdf_response = {}

    for i, page in enumerate(pages[:15]):
        image_path = os.path.join(pages_output_dir, f"page_{i + 1}.jpg")
        page.save(image_path, "JPEG")
        response = process_image_with_mathpix(image_path)
        response["page_image_path"] = image_path
        response["exam_name"] = pdf_path.split(".pdf")[0]
        pdf_response[f"page_{i + 1}"] = response
        output_file = os.path.join(json_output_dir, f"page_{i + 1}.json")
        with open(output_file, "w") as f:
                json.dump(response, f, indent=4)
    print(f"All pages processed. Response saved to {output_file}")
    return pdf_response

def chat_completion(client, messages, model, return_text=True, return_usage=True, model_args=None):
    """Calls OpenAI API with the provided messages and model."""
    if model_args is None:
        model_args = {}

    while True:
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, **model_args
            )
            text = response.choices[0].message.content.strip()
            usage = response.usage

            if return_text and return_usage:
                return text, dict(usage)
            if return_text:
                return text
            if return_usage:
                return usage
            return response
        except (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError) as e:
            print(f"OpenAI error: {str(e)}. Waiting for 1 minute.")
            time.sleep(60)
            continue

def process_questions(api_key, lang, processed_output, model="gpt-4o"):
    """Process questions using OpenAI API."""
    client = OpenAI(api_key=api_key)
    prompt = GPT_PROMPT.format(lang=lang, processed_output=processed_output)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    model_args = {"max_tokens": 1024, "temperature": 0.7}

    try:
        response_text, usage = chat_completion(
            client,
            messages=messages,
            model=model,
            return_text=True,
            return_usage=True,
            model_args=model_args
        )
        return response_text
    except Exception as e:
        print(f"Error processing questions: {e}")
        return None

def plot_bounding_boxes(image_path, response):
    """Plots bounding boxes from Mathpix API response on top of the input image."""
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font_path = "DejaVuSans-Bold.ttf"
    font_size = 30
    font = ImageFont.truetype(font_path, font_size)

    image_width, image_height = image.size
    mathpix_width = response.get("image_width", None)
    mathpix_height = response.get("image_height", None)

    if mathpix_width and mathpix_height:
        if image_width != mathpix_width or image_height != mathpix_height:
            print(
                f"Warning: Image dimensions mismatch. "
                f"Image: {image_width}x{image_height}, "
                f"Mathpix: {mathpix_width}x{mathpix_height}"
            )

    for line in response["line_data"]:
        if "cnt" in line:
            cnt = line["cnt"]
            flat_cnt = [coord for point in cnt for coord in point]
            draw.polygon(flat_cnt, outline="red", width=2)
            text = line.get("type", "")
            if text:
                x, y = cnt[0]
                draw.text((x, y), text, fill="blue", font=font)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def get_options(choices_matches):
    option_pattern = r"\n\(\d+\) ([^\n]+)"
    final_matches = []
    for choice_match in choices_matches:
        options = re.findall(option_pattern, choice_match)
        final_matches.append(options)
    return final_matches

def parse_gpt_output(response):
    question_pattern = re.compile(r"<question>(.*?)</question>", re.DOTALL)
    instruction_pattern = re.compile(r"<instruction>(.*?)</instruction>", re.DOTALL)
    choices_pattern = re.compile(r"<choices>(.*?)</choices>", re.DOTALL)
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    requires_image_pattern = re.compile(r"<image>(.*?)</image>", re.DOTALL)
    # category_pattern = re.compile(r"<category>(.*?)</category>", re.DOTALL)
    # context_pattern = re.compile(r"<context>(.*?)</context>", re.DOTALL)
    question_num = re.compile(r"<question_num>(.*?)</question_num>", re.DOTALL)

    # Find all matches for questions
    question_matches = question_pattern.findall(response)
    instruction_matches = instruction_pattern.findall(response)
    choices_matches = choices_pattern.findall(response)
    crop_images = requires_image_pattern.findall(response)
    # contexts = context_pattern.findall(response)
    # categories = category_pattern.findall(response)
    q_nums = question_num.findall(response)
    if len(instruction_matches) == len(question_matches):
        for i in range(len(question_matches)):
            question_matches[i] = instruction_matches[i] + "\n" + question_matches[i]
    else:
        if len(instruction_matches) < len(question_matches):
            instruction_matches += [''] * (len(question_matches) - len(instruction_matches))
        else:
            raise ValueError()
    final_matches = get_options(choices_matches)

    answer_matches = answer_pattern.findall(response)
    return (
        question_matches,
        final_matches,
        crop_images,
        # categories,
        # contexts,
        q_nums,
    )

def main():
    parser = argparse.ArgumentParser(description="Process PDF and extract questions using Mathpix and OpenAI APIs.")
    parser.add_argument("--pdf_path", type=str, default="sample.pdf", help="Default path to the PDF file.")
    parser.add_argument("--lang", type=str, default="sample.pdf", help="Default path to the PDF file.")
    parser.add_argument("--mathpix_out_dir", type=str, default="exams/JEE_Main/2013/mathpix_results")
    parser.add_argument("--crops_dir", type=str, default="exams/JEE_Main/2013/crops")
    parser.add_argument("--processed_out_dir", type=str, default="exams/JEE_Main/2013/processed_results")
    parser.add_argument("--api_key", type=str, default="", help="Default OpenAI API key.")
    args = parser.parse_args()

    pdf_response = process_pdf(args.pdf_path, args.mathpix_out_dir)
    
    processed_output = process_line_data(pdf_response, args.crops_dir)
    for page in processed_output:
        response_text = process_questions(args.api_key, args.lang, page)
        print(response_text)
        # print(response_text)
        # questions, choices, crop_image_paths, q_nums, = parse_gpt_output(response_text)
        # for i, question in enumerate(questions):
        #     print(f"Question {q_nums[i]}: {question}")
        #     print(f"Choices: {choices[i]}")
        #     # print(f"Question Number: {q_nums[i]}")
        #     print(f"Image: {crop_image_paths[i]}")
        #     print("\n")


if __name__ == "__main__":
    main()

# python parse.py --pdf_path exams/JEE_Main/2013/Papers/jee-main-paper-1-2013-hindi-p.pdf --lang Hindi --mathpix_out_dir exams/JEE_Main/2013/mathpix_results --crops_dir exams/JEE_Main/2013/crops --processed_out_dir exams/JEE_Main/2013/processed_results  --api_key 
# {
#     "language": args.lang,
#     "country": "India",
#     "file_name": pdf_name_in_database,
#     "source": "https://m.shsbnu.net/pluginfile.php/38738/mod_resource/content/1/%5B2018-Official-AP%20Practice%20Exam%5D%20%28With%20Answers%29.pdf",
#     "license": "Unknown",
#     "level": "University Entrance",
#     "category_en": “Chemistry",
#     "category_original_lang": "Chemistry",
#     "original_question_num": 2,
#     "question": "",
#     "options": [],
#     "answer": "",
#     "image_png": png_file,
#     "image_information" : "essential",
#     "image_type": 
#     "parallel_question_id": None, #This exam does not contain questions duplicated in different languages.
# }