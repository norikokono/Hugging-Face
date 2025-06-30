import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import StableDiffusionPipeline # Consolidated diffusers imports
import os
from huggingface_hub import login
from transformers import pipeline, AutoTokenizer # Consolidated transformers imports
import re
import logging

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Authenticate with Hugging Face using the token from environment variables.
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    print("HF_TOKEN environment variable found. Models will attempt to load with this token.")
else:
    print("Warning: HF_TOKEN environment variable not found. Gated models may not load. Please ensure it's set as a secret in your Hugging Face Space or as an environment variable locally.")

# --- Global Model Variables (initialized to None) ---
gemma_lm_pipeline = None
gemma_tokenizer_instance = None
stable_diffusion_pipeline = None


# --- Utility Functions ---

# Corrected clear_all to accept inputs and return empty values for outputs
def clear_all(scene_input_val, story_output_val, dialogue_output_val, image_output_val):
    """
    Function to clear all input and output fields in the Gradio interface.
    It must accept the current values of the components it's clearing as inputs,
    and return empty values for all the components it's setting as outputs.
    """
    logging.info("Clearing all fields.")
    # Return empty strings for textboxes and None for gr.Image
    return "", "", "", None

# --- Model Loading Functions ---
def load_gemma_model():
    """Load the Gemma text-generation model and its tokenizer into global variables."""
    global gemma_lm_pipeline, gemma_tokenizer_instance
    if gemma_lm_pipeline is None: # Only load if not already loaded
        try:
            print("Attempting to load Gemma model (google/gemma-2-2b) and tokenizer...")
            gemma_tokenizer_instance = AutoTokenizer.from_pretrained("google/gemma-2-2b")
            gemma_lm_pipeline = pipeline(
                "text-generation",
                model="google/gemma-2-2b",
                device_map="auto", # This will correctly send it to "cpu"
                tokenizer=gemma_tokenizer_instance
                # REMOVE: model_kwargs={"quantization_config": quantization_config} if quantization_config else {}
                # REMOVE: the quantization_config variable definition entirely from your code
            )
            print("Gemma model and tokenizer loaded successfully on CPU.")
        except Exception as e:
            print(f"Error loading Gemma model and tokenizer: {e}")
            print("!!! IMPORTANT: Double-check that your HF_TOKEN is set correctly as an environment variable. Also, make sure you have accepted the terms of use for 'google/gemma-2-2b' on its Hugging Face model page. !!!")
            print("Consider your system's RAM: Gemma can be memory intensive on CPU. If you have limited RAM, this could be the cause.")
            gemma_lm_pipeline = None
            gemma_tokenizer_instance = None

def load_stable_diffusion_model():
    """Load the Stable Diffusion image-generation model into a global variable."""
    global stable_diffusion_pipeline
    if stable_diffusion_pipeline is None: # Only load if not already loaded
        try:
            print("Attempting to load Stable Diffusion model...")
            # Note: The safety checker warning is informational. For public services, consider enabling it.
            stable_diffusion_pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32 # Use float32 for CPU compatibility
            )
            stable_diffusion_pipeline.to("cpu") # Explicitly move to CPU
            print("Stable Diffusion model loaded successfully on CPU.")
        except Exception as e:
            print(f"Error loading Stable Diffusion model: {e}")
            print("Ensure you have sufficient RAM for CPU processing for Stable Diffusion.")
            stable_diffusion_pipeline = None


# --- Story and Dialogue Generation Functions ---

def generate_story(scene_description: str) -> str:
    """Generate a short comic story based on the given scene description."""
    load_gemma_model() # Ensure Gemma is loaded before use
    if gemma_lm_pipeline is None: # Changed to gemma_lm_pipeline
        return "Story generation service is unavailable: Gemma model failed to load. Check console logs for details (HF_TOKEN, terms, RAM)."

    # Modified prompt to explicitly ask for a finished sentence/thought
    prompt = f"Write a brief, engaging, and imaginative comic book scene description, focusing on action and character interaction. End the story with a complete sentence that sets up a dramatic moment: {scene_description}\n\nStory:"

    try:
        # max_length takes precedence over max_new_tokens if both are set.
        # It's generally better to use max_new_tokens when generating a specific amount of new text.
        response = gemma_lm_pipeline(prompt, max_new_tokens=100, do_sample=True)
        if isinstance(response, list) and response and 'generated_text' in response[0]:
            generated_text = response[0]['generated_text'].strip()

            logging.debug(f"Raw story generated by Gemma (before stripping prompt):\n{generated_text}")

            # Find the last occurrence of the prompt to avoid issues with repeated prompt parts
            prompt_end_index = generated_text.rfind(prompt)
            if prompt_end_index != -1:
                generated_text = generated_text[prompt_end_index + len(prompt):].strip()

            generated_text = generated_text.replace("<unk>", "").strip()
            logging.debug(f"Cleaned story for display (first 500 chars):\n{generated_text[:500]}...")

            return generated_text
        else:
            return "Failed to generate story: Unexpected output format from Gemma model."
    except Exception as e:
        logging.error(f"Story generation error: {e}", exc_info=True)
        return f"Story generation error: {e}"

def generate_dialogue(story_text: str) -> str:
    """Generate dialogue for a comic panel based on the generated story."""
    load_gemma_model() # Ensure Gemma is loaded before use
    if gemma_lm_pipeline is None: # Changed to gemma_lm_pipeline
        return "Dialogue generation service is unavailable: Gemma model failed to load."

    # Prompt for dialogue extraction/generation
    prompt = f"Given the following comic scene story, extract a concise, impactful dialogue for a single speech bubble. Make it brief and punchy, as if a character is speaking: '{story_text}'\n\nDialogue:"

    try:
        # Pass the prompt directly to the pipeline
        response = gemma_lm_pipeline(prompt, max_new_tokens=20, do_sample=True) # Changed max_length to max_new_tokens
        if isinstance(response, list) and response and 'generated_text' in response[0]:
            generated_text = response[0]['generated_text'].strip()

            # Remove the prompt if it's repeated
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            # Further refine to get only a short dialogue
            lines = generated_text.split('\n')
            dialogue = lines[0].strip() if lines else "No dialogue generated."

            # --- Start of Moved and Corrected Dialogue Cleaning Logic ---
            dialogue = re.sub(r"Dialogue:\s*", "", dialogue, flags=re.IGNORECASE).strip()
            dialogue = re.sub(r"The dialogue is:\s*", "", dialogue, flags=re.IGNORECASE).strip()
            dialogue = re.sub(r"Here is the dialogue:\s*", "", dialogue, flags=re.IGNORECASE).strip()
            dialogue = re.sub(r"Generated dialogue:\s*", "", dialogue, flags=re.IGNORECASE).strip()
            dialogue = re.sub(r"This scene features:\s*", "", dialogue, flags=re.IGNORECASE).strip()
            dialogue = re.sub(r"Given the story, here's a dialogue:\s*", "", dialogue, flags=re.IGNORECASE).strip()

            # This line might remove too much if the story text is literally part of the dialogue
            # Consider if this is truly necessary after previous prompt stripping
            # dialogue = dialogue.replace(story_text.strip(), "").strip()

            dialogue = re.sub(r"<start_of_turn>\s*model\s*", "", dialogue, flags=re.IGNORECASE).strip()
            dialogue = re.sub(r'<[^>]*>', '', dialogue).strip() # Remove any remaining HTML/XML like tags
            dialogue = dialogue.replace("<eos>", "").replace("<pad>", "").strip()
            dialogue = dialogue.replace("<end_of_turn>", "").strip()

            dialogue = dialogue.split('\n')[0].strip() # Take only the first line after cleaning

            problematic_phrases = ["", ".", ",", "...", "(no dialogue)", "error", "dialogue unavailable", "none", "n/a", "no clear dialogue", "the dialogue is", "that", "this", "i cannot generate dialogue", "i can't generate dialogue", "story:", "brave"]
            dialogue_lower = dialogue.lower()
            if len(dialogue) < 5 or \
               any(phrase in dialogue_lower for phrase in problematic_phrases if len(phrase) > 1):
                logging.warning(f"Extracted dialogue too short/generic or problematic: '{dialogue}'. Setting to default.")
                dialogue = "No clear dialogue."

            # Limit dialogue length to avoid overflowing the bubble
            dialogue = dialogue[:70] + "..." if len(dialogue) > 70 else dialogue
            # --- End of Moved and Corrected Dialogue Cleaning Logic ---

            return dialogue
        else:
            return "Failed to generate dialogue: Unexpected output format from Gemma model."
    except Exception as e:
        logging.error(f"Dialogue generation (inference) error: {e}", exc_info=True)
        return f"Dialogue generation error: {e}. Check Space logs for details."

# --- Image Generation and Manipulation Functions ---

def generate_comic_image(scene_description: str) -> Image.Image:
    """Generate a comic panel based on the scene description."""
    load_stable_diffusion_model() # Ensure Stable Diffusion is loaded before use
    if stable_diffusion_pipeline is None: # Changed to stable_diffusion_pipeline
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
        image = stable_diffusion_pipeline(prompt=scene_description, num_inference_steps=20).images[0] # Changed to stable_diffusion_pipeline
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
    selected_font_path = None

    font_paths = [
        "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf"
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                ImageFont.truetype(path, size=initial_font_size)
                selected_font_path = path
                break
            except IOError:
                logging.warning(f"Could not load TrueType font from {path}, trying next.")
                pass

    def get_text_dimensions(text, current_font_obj):
        if not text:
            return 0, 0
        try:
            bbox = current_font_obj.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception as e:
            logging.warning(f"Error getting text dimensions with getbbox, falling back: {e}")
            # Fallback for older Pillow versions or unusual fonts
            if hasattr(current_font_obj, 'size'):
                return len(text) * current_font_obj.size * 0.6, current_font_obj.size * 1.2
            else:
                return len(text) * 10, 15

    current_font_size = initial_font_size
    font_to_use = None
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

    for _ in range(5): # Iteratively try smaller font sizes
        current_font_obj = None
        if selected_font_path:
            try:
                current_font_obj = ImageFont.truetype(selected_font_path, size=current_font_size)
            except IOError:
                logging.warning(f"Failed to load TrueType font from '{selected_font_path}' at size {current_font_size}, falling back to default.")
                selected_font_path = None # Don't try this path again
                current_font_obj = ImageFont.load_default()
        else:
            current_font_obj = ImageFont.load_default()

        if current_font_obj is None: # Fallback if for some reason default also fails
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
            font_to_use = current_font_obj
            break
        elif current_font_size > 10: # Only decrease font size if it's above 10
            current_font_size -= 2
            if current_font_size < 10: current_font_size = 10 # Don't go below 10
        else: # If font size is already 10 or less and still too big, just use it
            wrapped_lines = test_wrapped_lines
            font_to_use = current_font_obj
            break

    if not wrapped_lines: # Fallback if text wrapping fails for some reason
        wrapped_lines = [dialogue_text]
    if font_to_use is None:
        font_to_use = ImageFont.load_default()

    final_bubble_width = 0
    final_line_height = 0
    for line in wrapped_lines:
        line_w, line_h = get_text_dimensions(line, font_to_use)
        final_bubble_width = max(final_bubble_width, line_w)
        final_line_height = max(final_line_height, line_h)

    final_bubble_width += 2 * padding_x
    final_bubble_height = len(wrapped_lines) * final_line_height + 2 * padding_y

    final_bubble_width = min(final_bubble_width, max_bubble_width)
    final_bubble_height = min(final_bubble_height, max_bubble_height)
    final_bubble_width = max(final_bubble_width, min_bubble_dim)
    final_bubble_height = max(final_bubble_height, min_bubble_dim // 2)


    # Bubble position (bottom left corner)
    x1 = margin
    y1 = image_copy.height - final_bubble_height - margin
    x2 = x1 + final_bubble_width
    y2 = y1 + final_bubble_height

    # Draw the white circular bubble with a black outline.
    # Changed from ellipse to rounded rectangle for a more comic-like bubble
    radius = 20 # Adjust for roundness
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill="white", outline="black", width=2)


    text_current_y = y1 + padding_y
    for line in wrapped_lines:
        line_w, _ = get_text_dimensions(line, font_to_use)
        text_x = x1 + (final_bubble_width - line_w) / 2
        draw.text((text_x, text_current_y), line, font=font_to_use, fill="black") # Changed fill to black for contrast
        text_current_y += final_line_height

    # Draw the speech bubble tail
    tail_base_x1 = x1 + (final_bubble_width * 0.25)
    tail_base_y1 = y2

    tail_base_x2 = x1 + (final_bubble_width * 0.75)
    tail_base_y2 = y2

    tail_tip_x = x1 + (final_bubble_width * 0.5)
    tail_tip_y = y2 + 15 # Extends 15 pixels below the bubble

    draw.polygon([
        (tail_base_x1, tail_base_y1),
        (tail_tip_x, tail_tip_y),
        (tail_base_x2, tail_base_y2)
    ], fill="black") # Fill for the tail

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
            None # Return None for image when empty scene
        ]

    story = generate_story(scene_description)
    dialogue = generate_dialogue(story)
    image = generate_comic_image(scene_description)

    is_placeholder_image = False
    if image is not None and image.width == 400 and image.height == 300:
        # Check a pixel in the middle to be more robust
        try:
            mid_pixel_color = image.getpixel((image.width // 2, image.height // 2))
            if mid_pixel_color in [(192, 192, 192), (255, 0, 0)]: # Lightgray or Red
                is_placeholder_image = True
        except Exception: # In case getpixel fails on some image types
            pass

    if is_placeholder_image:
        final_image = image
    elif image is not None: # Only add bubble if image was successfully generated
        final_image = add_speech_bubble(image, dialogue)
    else:
        final_image = None # No image to add bubble to

    return [story, dialogue, final_image]

# --- Load Custom CSS ---
custom_css = ""
try:
    with open("style.css", "r") as f:
        custom_css = f.read()
except FileNotFoundError:
    print("Warning: style.css not found. Running without custom CSS. Please ensure 'style.css' is in the same directory.")

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Citrus(), css=custom_css) as demo:
    gr.Markdown(
        """
        # 🦸‍♂️ Fantastic AI 🤖 Comic Generator 💥
        <p style="font-size: 1.25em; color: var(--teal-text); display: block; margin-bottom: 15px; font-weight: 600;">
        Generate a unique comic scene using advanced AI models! Provide a scene description, and the AI will craft a short story, generate a punchy dialogue, and create a stylized comic panel image.
        </p>
        <br>
        <b>Important Note on Performance:</b> This app runs on free servers, using only CPU power. Thus, this results in <span style="color: var(--red-text); font-weight: bold;">significantly slower</span> generation compared to GPU due to the model's size and computational requirements.
        <br><b>Tip:</b> To get more comic-like images, try adding style words to your description, such as "comic book art," "graphic novel style," or "bold outlines!"
        <br><br>
        """,
        elem_id="note-section"
    )

    with gr.Row():
        scene_input = gr.Textbox(
            lines=2,
            placeholder="Describe your scene, e.g., 'A cat wearing a superhero cape flying over a city at sunset.'",
            label="Scene Description for Comic Panel",
            scale=3
        )
        submit_btn = gr.Button("Generate Comic", variant="primary", scale=1)

    with gr.Row():
        story_output = gr.Textbox(label="AI-Generated Story", lines=5, interactive=False, max_lines=10)
        dialogue_output = gr.Textbox(label="AI-Generated Dialogue", lines=3, interactive=False, max_lines=5)

    image_output = gr.Image(label="Comic Panel Art with Dialogue", type="pil", interactive=False)

    submit_btn.click(
        fn=generate_comic_panel,
        inputs=scene_input,
        outputs=[story_output, dialogue_output, image_output]
    )

    gr.Button("Clear All", variant="secondary").click(
        fn=clear_all,
        # Inputs should match the components you want to clear (their current values)
        inputs=[scene_input, story_output, dialogue_output, image_output],
        # Outputs should match the components you want to update (set to empty)
        outputs=[scene_input, story_output, dialogue_output, image_output]
    )

    gr.Examples(
        examples=[
            ["A futuristic bounty hunter cyborg on a neon-lit rooftop, overlooking a dystopian city, in a cyberpunk comic art style, detailed, dark, atmospheric."],
            ["A brave knight and a fire-breathing dragon sharing a cup of tea in a whimsical, enchanted forest, fantasy comic style, bright, joyful."],
            ["A group of diverse superheroes soaring above a city skyline at dawn, preparing for battle against a giant robot, classic comic book art, dynamic, heroic."],
            ["A mischievous alien chef attempting to bake an intergalactic pizza, causing a chaotic explosion of cosmic ingredients, cartoon comic style, messy, hilarious."],
            ["An ancient vampire detective in a foggy London alley, investigating a mysterious case, gothic comic art, shadowy, suspenseful."],
            ["A joyful space explorer discovering a new planet filled with sentient, singing plants, sci-fi comic art, vibrant, lush."],
            ["A secret agent cat infiltrating a high-tech villain's lair, navigating laser grids with agility, spy comic style, sleek, intense."],
            ["Two elderly grandmothers in a fierce magical duel using knitting needles as wands, in a cozy village square, humorous fantasy comic, lighthearted, quirky."],
            ["A giant, friendly robot protecting a tiny, scared kitten from a thunderstorm in a modern city, heartwarming comic style, gentle, protective."],
            ["A pirate crew of anthropomorphic animals searching for buried treasure on a deserted island, adventure comic style, sandy, lush."],
        ],
        inputs=scene_input,
        outputs=[story_output, dialogue_output, image_output],
        fn=generate_comic_panel, # Crucial: examples must specify a function to run
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch(share=False)
