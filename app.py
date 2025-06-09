import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import StableDiffusionPipeline
import os
from huggingface_hub import login
from transformers import pipeline, BitsAndBytesConfig # Import BitsAndBytesConfig for potential quantization

import re # Import the regular expression module for HTML stripping

# Authenticate with Hugging Face using the token from environment variables.
try:
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Successfully logged into Hugging Face Hub.")
    else:
        print("Warning: HF_TOKEN environment variable not found. Gated models may not load. Please ensure it's set as a secret in your Hugging Face Space or as an environment variable locally.")
        hf_token = None
except Exception as e:
    print(f"Hugging Face login failed: {e}. Ensure HF_TOKEN is correctly set as a secret.")
    hf_token = None

# Load the Gemma text-generation model.
gemma_lm = None
try:
    print("Attempting to load Gemma model (google/gemma-2-2b)...")
    # Using torch_dtype=torch.bfloat16 can help with memory and potentially speed on modern CPUs
    gemma_lm = pipeline(
        "text-generation",
        model="google/gemma-2-2b",
        device_map="auto",
        # torch_dtype=torch.bfloat16 # Use bfloat16 for potential CPU speedup and memory reduction
    )
    print("Gemma model loaded successfully.")
except Exception as e:
    print(f"Error loading Gemma model: {e}")
    print("!!! IMPORTANT: Double-check that your HF_TOKEN is set correctly as an environment variable. Also, make sure you have accepted the terms of use for 'google/gemma-2-2b' on its Hugging Face model page. !!!")
    print("Consider your system's RAM: Gemma can be memory intensive even when quantized for CPU. If you have limited RAM, this could be the cause.")
    gemma_lm = None

# Load the Stable Diffusion image-generation model for CPU.
image_model = None
try:
    print("Attempting to load Stable Diffusion model...")
    # Using float16 (half-precision) for CPU can sometimes offer a speedup and reduces memory usage.
    image_model = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        # torch_dtype=torch.float16 # Use float16 for reduced memory and potential speedup on CPU
    )
    image_model.to("cpu")
    print("Stable Diffusion model loaded successfully on CPU.")
except Exception as e:
    print(f"Error loading Stable Diffusion model: {e}")
    print("Ensure you have sufficient RAM for CPU processing for Stable Diffusion.")
    image_model = None

# --- Story and Dialogue Generation Functions ---

def generate_story(scene_description: str) -> str:
    """Generate a short comic story based on the given scene description."""
    if gemma_lm is None:
        return "Story generation service is unavailable: Gemma model failed to load. Check console logs for details (HF_TOKEN, terms, RAM)."

    prompt = f"Write a brief, engaging, and imaginative comic book scene description, focusing on action and character interaction: {scene_description}\n\nStory:"

    try:
        response = gemma_lm(prompt, max_new_tokens=100, do_sample=True, truncation=True, max_length=512)
        if isinstance(response, list) and response and 'generated_text' in response[0]:
            generated_text = response[0]['generated_text'].strip()
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            return generated_text
        else:
            return "Failed to generate story: Unexpected output format from Gemma model."
    except Exception as e:
        return f"Story generation error: {e}"

def generate_dialogue(story_text: str) -> str:
    """Generate dialogue for a comic panel based on the generated story."""
    if gemma_lm is None:
        return "Dialogue generation service is unavailable: Gemma model failed to load."

    prompt = f"Given the following comic scene story, extract a concise, impactful dialogue for a single speech bubble. Make it brief and punchy, as if a character is speaking: '{story_text}'\n\nDialogue:"

    try:
        response = gemma_lm(prompt, max_new_tokens=20, do_sample=True, truncation=True, max_length=512)
        if isinstance(response, list) and response and 'generated_text' in response[0]:
            generated_text = response[0]['generated_text'].strip()
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            lines = generated_text.split('\n')
            dialogue = lines[0].strip() if lines else "No dialogue generated."

            # Remove HTML tags from the dialogue
            dialogue = re.sub(r'<[^>]*>', '', dialogue)
            
            return dialogue[:70] + "..." if len(dialogue) > 70 else dialogue
        else:
            return "Failed to generate dialogue: Unexpected output format from Gemma model."
    except Exception as e:
        return f"Dialogue generation error: {e}"

# --- Image Generation and Manipulation Functions ---

def generate_comic_image(scene_description: str) -> Image.Image:
    """Generate a comic panel based on the scene description."""
    if image_model is None:
        print("Image generation model is unavailable.")
        img = Image.new("RGB", (400, 300), color="lightgray")
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        d.text((50, 140), "Image model unavailable", fill=(0,0,0), font=font)
        return img

    try:
        image = image_model(prompt=scene_description, num_inference_steps=20).images[0]
        return image
    except Exception as e:
        print(f"Image generation error: {e}")
        img = Image.new("RGB", (400, 300), color="red")
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        d.text((50, 140), f"Image Gen Error: {str(e)[:50]}...", fill=(255,255,255), font=font)
        return img

def add_speech_bubble(image: Image.Image, dialogue_text: str) -> Image.Image:
    """Overlay dialogue onto the comic image with a speech bubble effect."""
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    
    initial_font_size = 20
    selected_font_path = None # Store the path if a TrueType font is found
    
    # --- Step 1: Try to find a usable TrueType font path on the system ---
    font_paths = [
        "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf", # macOS path
        "/Library/Fonts/Arial.ttf", # macOS common path
        "/usr/share/fonts/opentype/noto/NotoSans-Regular.ttf", # Common Linux path
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf" # Another common Linux path
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                # Test if PIL can actually load it at base size
                ImageFont.truetype(path, size=initial_font_size)
                selected_font_path = path
                break # Found a working TrueType font, stop searching
            except IOError:
                print(f"Warning: Could not load TrueType font from {path}, trying next.")
                pass # This path didn't work, try the next one

    # --- Step 2: Define a helper to get text dimensions based on font object ---
    def get_text_dimensions(text, current_font_obj):
        if not text:
            return 0, 0
        try:
            bbox = current_font_obj.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            # Fallback for font issues (e.g., if current_font_obj somehow invalid)
            if hasattr(current_font_obj, 'size'):
                return len(text) * current_font_obj.size * 0.6, current_font_obj.size * 1.2
            else: # Even if it's a completely broken font object
                return len(text) * 10, 15 # Default rough estimate

    # --- Step 3: Initialize font parameters for the resizing loop ---
    current_font_size = initial_font_size
    font_to_use = None # This will hold the final font object for drawing

    # --- Step 4: Iterate and wrap words, dynamically resizing font if needed ---
    words = dialogue_text.split(' ')
    wrapped_lines = []

    padding_x = 18
    padding_y = 12
    margin = 15
    max_bubble_width_percent = 0.5
    max_bubble_height_percent = 0.3
    max_bubble_width = image_copy.width * max_bubble_width_percent
    max_bubble_height = image_copy.height * max_bubble_height_percent
    min_bubble_dim = 60


    for _ in range(5): # Allow up to 5 attempts to fit text by reducing font size
        current_font_obj = None
        if selected_font_path:
            try:
                # Always create a new font object for the current size using the selected path
                current_font_obj = ImageFont.truetype(selected_font_path, size=current_font_size)
            except IOError:
                # If loading fails here for this specific size, fall back to default
                print(f"Warning: Failed to load TrueType font from '{selected_font_path}' at size {current_font_size}, falling back to default.")
                selected_font_path = None # Prevent trying TrueType again in subsequent iterations
                current_font_obj = ImageFont.load_default()
        else:
            current_font_obj = ImageFont.load_default() # Use default if no TrueType path found or failed

        # Ensure we always have a font object
        if current_font_obj is None:
            current_font_obj = ImageFont.load_default()

        test_wrapped_lines = []
        current_line = ""
        available_width_for_text = max_bubble_width - 2 * padding_x

        if available_width_for_text <= 0:
            break

        for word in words:
            temp_line = current_line + " " + word if current_line else word
            line_width, _ = get_text_dimensions(temp_line, current_font_obj)
            
            if line_width <= available_width_for_text:
                current_line = temp_line
            else:
                test_wrapped_lines.append(current_line)
                current_line = word
        test_wrapped_lines.append(current_line)

        approx_line_height = get_text_dimensions("Ag", current_font_obj)[1]
        total_text_height_with_padding = len(test_wrapped_lines) * approx_line_height + 2 * padding_y

        if total_text_height_with_padding <= max_bubble_height:
            wrapped_lines = test_wrapped_lines
            font_to_use = current_font_obj # This is the font that fits
            break # Text fits, we're good
        elif current_font_size > 10: # If text doesn't fit and font is still large enough to shrink
            current_font_size -= 2 # Reduce font size
            if current_font_size < 10: current_font_size = 10 # Don't go too small
        else: # Cannot fit, even with smallest font, just use what we have
            wrapped_lines = test_wrapped_lines
            font_to_use = current_font_obj
            break
            
    # Final fallback if somehow wrapped_lines is empty or font_to_use is still None
    if not wrapped_lines:
        wrapped_lines = [dialogue_text]
    if font_to_use is None:
        font_to_use = ImageFont.load_default()


    # Calculate final bubble dimensions based on wrapped text and updated font size
    final_bubble_width = 0
    final_line_height = 0
    for line in wrapped_lines:
        line_w, line_h = get_text_dimensions(line, font_to_use)
        final_bubble_width = max(final_bubble_width, line_w)
        final_line_height = max(final_line_height, line_h)

    final_bubble_width += 2 * padding_x
    final_bubble_height = len(wrapped_lines) * final_line_height + 2 * padding_y

    # Apply min/max constraints one last time
    final_bubble_width = min(final_bubble_width, max_bubble_width)
    final_bubble_height = min(final_bubble_height, max_bubble_height)
    final_bubble_width = max(final_bubble_width, min_bubble_dim)
    final_bubble_height = max(final_bubble_height, min_bubble_dim // 2) # Ensure some minimum height

    # Position the bubble (bottom-left)
    x1 = margin
    y1 = image_copy.height - final_bubble_height - margin
    x2 = x1 + final_bubble_width
    y2 = y1 + final_bubble_height

    # Draw the bubble (more elliptical)
    draw.ellipse([x1, y1, x2, y2], fill="black")

    # Draw the wrapped text
    text_current_y = y1 + padding_y
    for line in wrapped_lines:
        line_w, _ = get_text_dimensions(line, font_to_use)
        text_x = x1 + (final_bubble_width - line_w) / 2
        draw.text((text_x, text_current_y), line, font=font_to_use, fill="white") # Use font_to_use
        text_current_y += final_line_height # Move to the next line

    # Add a simple triangular tail
    # Tail points from the bottom of the bubble, slightly inwards, towards the character.
    tail_base_x1 = x1 + (final_bubble_width * 0.25) # 25% into the bubble from left
    tail_base_y1 = y2 # Bottom edge of the bubble

    tail_base_x2 = x1 + (final_bubble_width * 0.75) # 75% into the bubble from left
    tail_base_y2 = y2 # Bottom edge of the bubble

    tail_tip_x = x1 + (final_bubble_width * 0.5) # Middle of the base
    tail_tip_y = y2 + 15 # 15 pixels below the bubble

    draw.polygon([
        (tail_base_x1, tail_base_y1),
        (tail_tip_x, tail_tip_y),
        (tail_base_x2, tail_base_y2)
    ], fill="black")

    return image_copy

def generate_comic_panel(scene_description: str) -> list:
    """
    Creates a comic panel by combining AI-generated story, dialogue, and image.
    Returns a list containing the story text, dialogue text, and the final image with the speech bubble.
    """
    if not scene_description or not scene_description.strip():
        return [
            "Please provide a description for your comic scene to get started!",
            "No dialogue generated for an empty scene.",
            Image.new("RGB", (400, 300), color="lightgray")
        ]

    story = generate_story(scene_description)
    dialogue = generate_dialogue(story)
    image = generate_comic_image(scene_description)

    is_placeholder_image = False
    if image.width == 400 and image.height == 300:
        # Check if it's the lightgray (initial unavailable) or red (error) placeholder
        # Check a pixel that should represent the background color
        if image.getpixel((image.width // 2, image.height // 2)) == (192, 192, 192) or \
           image.getpixel((image.width // 2, image.height // 2)) == (255, 0, 0):
            is_placeholder_image = True

    if is_placeholder_image:
        final_image = image
    else:
        final_image = add_speech_bubble(image, dialogue)

    return [story, dialogue, final_image]

# --- Load Custom CSS ---
custom_css = ""
try:
    with open("style.css", "r") as f:
        custom_css = f.read()
except FileNotFoundError:
    print("Warning: style.css not found. Running without custom CSS. Please ensure 'style.css' is in the same directory.")

# --- "Clear" Button Function ---
def clear_all():
    """Function to clear all inputs and outputs."""
    # Reset image to a default blank (lightgray) image
    # Return 4 values: scene_input, story_output, dialogue_output, image_output
    return "", "", "", Image.new("RGB", (400, 300), color="lightgray")


# In your gr.Textbox for scene_input:
scene_input = gr.Textbox(
    lines=2,
    placeholder="Describe your scene, e.g., 'A cat wearing a superhero cape flying over a city at sunset. (Try adding \"comic book art\" or \"graphic novel style\")'",
    label="Scene Description for Comic Panel",
    scale=3
)

# --- Gradio Interface ---
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown(
        """
        # 🦸‍♂️ Fantastic AI 🤖 Comic Generator 💥
        <p style="font-size: 1.25em; color: var(--teal-text); display: block; margin-bottom: 15px; font-weight: 600;">
        Generate a unique comic scene using advanced AI models! Provide a scene description, and the AI will craft a short story, generate a punchy dialogue, and create a stylized comic panel image.
        </p>
        <br><b>Note:</b> This app uses free resources. Thus, this version is optimized for CPU, so generation will be <span style="color: var(--red-text);">quite slow</span> compared to GPU due to model size and computational demands, and images are not top-tier. The primary bottleneck for the application on a CPU is the Stable Diffusion image generation model, and large language models like Gemma 2 are also quite demanding.
        <br><b>Tip:</b> To get more comic-like images, try adding style words to your description, such as "comic book art," "graphic novel style," or "bold outlines!"
        <br><br>
        """
    )

    with gr.Row():
        scene_input = gr.Textbox(
            lines=2,
            placeholder="Describe your scene, e.g., 'A cat wearing a superhero cape flying over a city at sunset. (Try adding \"comic book art\" or \"graphic novel style\")'",
            label="Scene Description for Comic Panel",
            scale=3
        )
        submit_btn = gr.Button("Generate Comic", variant="primary", scale=1)
        
    with gr.Row():
        story_output = gr.Textbox(label="AI-Generated Story", lines=5, interactive=False, max_lines=10)
        dialogue_output = gr.Textbox(label="AI-Generated Dialogue", lines=3, interactive=False, max_lines=5)
    
    image_output = gr.Image(label="Comic Panel Art with Dialogue", type="pil")

    submit_btn.click(
        fn=generate_comic_panel,
        inputs=scene_input,
        outputs=[story_output, dialogue_output, image_output],
        # Corrected: Use show_progress instead of loading_indicator
        show_progress="full" # Options: "full", "minimal", "hidden"
    )

    # Explicitly link ClearButton to a clearing function
    gr.Button("Clear All", variant="secondary").click(
        fn=clear_all,
        inputs=[],
        outputs=[scene_input, story_output, dialogue_output, image_output]
    )

    gr.Examples(
        examples=[
            ["Comic book art: A mischievous robot trying to bake a cake, making a huge sugary mess, with exaggerated facial expressions."],
            ["Graphic novel style: A clever detective dog investigating a missing bone in a spooky old mansion, dramatic lighting, dynamic pose."],
            ["Pop art comic panel: Two alien astronauts playing a strategic game of chess on the surface of the moon, with Earth shimmering in the background, bold outlines, vibrant colors."],
            ["Action comic scene: A fantasy hero battling a giant marshmallow monster in a candy forest, dynamic action lines, intense perspective."],
            ["Superhero comic style: A superhero squirrel saving a lost acorn from a black hole, city skyline in background, heroic pose."],
            ["Fantasy comic illustration: A wizard accidentally turns his cat into a fire-breathing dragon during a spell gone wrong in his cozy library, magical effects, expressive characters."],
            ["Cyberpunk comic art: An ancient samurai warrior facing off against a towering, futuristic mecha in a neon-lit cyber city, rain, reflections."],
            ["Adventure comic style: A group of diverse friends discovering a shimmering, hidden portal beneath an old oak tree in a mundane city park, sense of wonder, detailed foliage."],
            ["Humorous comic strip: A grumpy old gnome trying to win a village-wide pie-eating contest against a joyful, impossibly fast fairy, humorous expressions, food flying."],
            ["Sci-fi comic cover: A lone astronaut tending to a small, vibrant garden growing from strange alien soil on a barren, red planet, epic scale, distant nebulae."],
        ],
        inputs=scene_input,
        outputs=[story_output, dialogue_output, image_output],
        fn=generate_comic_panel,
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch(share=False)
    