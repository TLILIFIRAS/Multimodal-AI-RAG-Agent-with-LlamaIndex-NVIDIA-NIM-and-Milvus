import os
import base64
import fitz
from io import BytesIO
from PIL import Image
import requests
from llama_index.llms.nvidia import NVIDIA


def set_environment_variables():
    """Set necessary environment variables."""
    # Setting the environment variable for NVIDIA API key.
    # Replace the empty string with your actual NVIDIA API key.
    os.environ["NVIDIA_API_KEY"] = "" # Set API key here


def get_b64_image_from_content(image_content):
    """Convert image content to a base64 encoded string."""
    
    # Open the image content using BytesIO to read it in memory.
    img = Image.open(BytesIO(image_content))
    
    # Convert the image to RGB format if it's not already in that mode.
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Create a buffer to save the image in memory in JPEG format.
    buffered = BytesIO()
    
    # Save the image into the buffer as a JPEG.
    img.save(buffered, format="JPEG")
    
    # Convert the buffer contents to base64 and return it as a UTF-8 string.
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def is_graph(image_content):
    """Determine if an image is a graph, plot, chart, or table."""
    
    # Get the description or label of the image by passing its content.
    res = describe_image(image_content)
    
    # Check if the description contains any of the keywords: "graph", "plot", "chart", or "table"
    # Convert the description to lowercase to ensure the search is case-insensitive.
    return any(keyword in res.lower() for keyword in ["graph", "plot", "chart", "table"])

def process_graph(image_content):
    """Process a graph image and generate a description."""
    
    # Process the image content using a deplotting function, which likely converts
    # the graph into a textual description (linearized table).
    deplot_description = process_graph_deplot(image_content)
    
    # Initialize the NVIDIA Mixtral model (LLaMA 3.1 70B) for generating text completions.
    mixtral = NVIDIA(model_name="meta/llama-3.1-70b-instruct")
    
    # Generate a prompt for Mixtral to explain the linearized table in plain English.
    # The AI is tasked with explaining charts, specifically focusing on linearized tables.
    response = mixtral.complete(
        "Your responsibility is to explain charts. You are an expert in describing the "
        "responses of linearized tables into plain English text for LLMs to use. Explain the "
        "following linearized table. " + deplot_description
    )
    
    # Return the generated explanation text.
    return response.text

def describe_image(image_content):
    """Generate a description of an image using NVIDIA API."""
    
    # Convert the image content to a base64 encoded string for sending to the NVIDIA API.
    image_b64 = get_b64_image_from_content(image_content)
    
    # Set the NVIDIA API endpoint for the image description model.
    invoke_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b"
    
    # Retrieve the NVIDIA API key from environment variables.
    api_key = os.getenv("NVIDIA_API_KEY")
    
    # Check if the API key is set; if not, raise an error.
    if not api_key:
        raise ValueError("NVIDIA API Key is not set. Please set the NVIDIA_API_KEY environment variable.")

    # Prepare the headers for the API request, including the authorization token.
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }

    # Prepare the payload for the API request, which includes the image (as base64) and the query prompt.
    payload = {
        "messages": [
            {
                "role": "user",
                "content": f'Describe what you see in this image. <img src="data:image/png;base64,{image_b64}" />'
            }
        ],
        # Define parameters for the model such as max tokens, temperature, and top-p.
        "max_tokens": 1024,
        "temperature": 0.20,  # Controls randomness in the model's output
        "top_p": 0.70,  # Controls diversity of the generated output
        "seed": 0,  # Fixed seed for reproducibility
        "stream": False  # No streaming of responses
    }

    # Send a POST request to the NVIDIA API with the headers and payload.
    response = requests.post(invoke_url, headers=headers, json=payload)
    
    # Return the image description from the response, extracted from the first choice in the model's output.
    return response.json()["choices"][0]['message']['content']

def process_graph_deplot(image_content):
    """Process a graph image using NVIDIA's Deplot API."""
    
    # Set the NVIDIA Deplot API endpoint URL.
    invoke_url = "https://ai.api.nvidia.com/v1/vlm/google/deplot"
    
    # Convert the image content to a base64 encoded string to include in the request.
    image_b64 = get_b64_image_from_content(image_content)
    
    # Retrieve the NVIDIA API key from the environment variables.
    api_key = os.getenv("NVIDIA_API_KEY")
    
    # Check if the API key is set; if not, raise a ValueError to notify the user.
    if not api_key:
        raise ValueError("NVIDIA API Key is not set. Please set the NVIDIA_API_KEY environment variable.")

    # Prepare the headers for the API request, including the authorization token.
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }

    # Create the payload for the API request, which includes the image and a request to extract data from the graph.
    payload = {
        "messages": [
            {
                "role": "user",
                "content": f'Generate underlying data table of the figure below: <img src="data:image/png;base64,{image_b64}" />'
            }
        ],
        # Define model parameters, including maximum tokens and sampling temperature.
        "max_tokens": 1024,  # Limit on the length of the generated response
        "temperature": 0.20,  # Controls randomness (lower means more deterministic)
        "top_p": 0.20,  # Controls diversity in sampling (lower means focusing on high-probability responses)
        "stream": False  # No streaming of the response
    }

    # Send a POST request to the NVIDIA API with the headers and payload.
    response = requests.post(invoke_url, headers=headers, json=payload)
    
    # Return the content of the first response choice, which contains the extracted data.
    return response.json()["choices"][0]['message']['content']

def extract_text_around_item(text_blocks, bbox, page_height, threshold_percentage=0.1):
    """
    Extract text blocks that are above and below a given bounding box on a page.
    
    Args:
        text_blocks (list): A list of text blocks, each with a bounding box and corresponding text.
        bbox (fitz.Rect): The bounding box of the item of interest.
        page_height (float): The height of the page where the bounding box is located.
        threshold_percentage (float): Percentage of the page height and bbox width used to define 
                                      vertical and horizontal proximity thresholds.
    
    Returns:
        before_text (str): The text block directly above the item.
        after_text (str): The text block directly below the item.
    """
    
    before_text, after_text = "", ""  # Initialize variables for text above and below the bbox.
    
    # Define vertical and horizontal thresholds for proximity to the bbox.
    vertical_threshold_distance = page_height * threshold_percentage
    horizontal_threshold_distance = bbox.width * threshold_percentage

    # Iterate through each text block and calculate its position relative to the bbox.
    for block in text_blocks:
        # Get the bounding box of the current text block.
        block_bbox = fitz.Rect(block[:4])
        
        # Calculate the vertical distance between the block and the bbox.
        vertical_distance = min(abs(block_bbox.y1 - bbox.y0), abs(block_bbox.y0 - bbox.y1))
        
        # Calculate the horizontal overlap between the block and the bbox.
        horizontal_overlap = max(0, min(block_bbox.x1, bbox.x1) - max(block_bbox.x0, bbox.x0))

        # If the block is within the vertical threshold and horizontally overlaps with the bbox:
        if vertical_distance <= vertical_threshold_distance and horizontal_overlap >= -horizontal_threshold_distance:
            
            # Check if the block is above the bbox and store the text if it's the first found above.
            if block_bbox.y1 < bbox.y0 and not before_text:
                before_text = block[4]
            
            # Check if the block is below the bbox and store the text if it's the first found below.
            elif block_bbox.y0 > bbox.y1 and not after_text:
                after_text = block[4]
                break  # Stop the loop after finding the first text block below.

    # Return the text blocks found above and below the bbox.
    return before_text, after_text

def process_text_blocks(text_blocks, char_count_threshold=500):
    """
    Group text blocks together based on a character count threshold.
    
    Args:
        text_blocks (list): A list of text blocks where each block contains a bounding box and corresponding text.
        char_count_threshold (int): The maximum character count allowed for each grouped block of text.
    
    Returns:
        grouped_blocks (list): A list of tuples where each tuple contains the first block's bounding box 
                               and the grouped text content from multiple blocks.
    """
    
    current_group = []  # Stores the current group of text blocks.
    grouped_blocks = []  # Stores the final list of grouped text blocks.
    current_char_count = 0  # Keeps track of the character count in the current group.

    # Iterate through each text block.
    for block in text_blocks:
        if block[-1] == 0:  # Check if the block represents text (not an image or other content).
            block_text = block[4]  # Get the actual text content of the block.
            block_char_count = len(block_text)  # Calculate the character count of the block.

            # If adding this block's character count does not exceed the threshold, add it to the current group.
            if current_char_count + block_char_count <= char_count_threshold:
                current_group.append(block)
                current_char_count += block_char_count
            else:
                # If the threshold is exceeded, save the current group and start a new one.
                if current_group:
                    grouped_content = "\n".join([b[4] for b in current_group])  # Combine texts in the current group.
                    grouped_blocks.append((current_group[0], grouped_content))  # Save the first block's bbox and content.
                
                # Reset the current group with the new block.
                current_group = [block]
                current_char_count = block_char_count

    # Append the last group after processing all blocks.
    if current_group:
        grouped_content = "\n".join([b[4] for b in current_group])
        grouped_blocks.append((current_group[0], grouped_content))

    return grouped_blocks

def save_uploaded_file(uploaded_file):
    """
    Save an uploaded file to a temporary directory for further processing.

    Args:
        uploaded_file (UploadedFile): The file object uploaded by the user.

    Returns:
        temp_file_path (str): The full path of the saved temporary file.
    """
    
    # Create the path to the temporary directory where the file will be saved.
    temp_dir = os.path.join(os.getcwd(), "vectorstore", "ppt_references", "tmp")
    
    # Ensure the temporary directory exists, create it if it doesn't.
    os.makedirs(temp_dir, exist_ok=True)
    
    # Construct the full path for the temporary file using the uploaded file's name.
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    
    # Open the file in binary write mode and save the contents of the uploaded file.
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    
    # Return the full path of the saved temporary file.
    return temp_file_path
