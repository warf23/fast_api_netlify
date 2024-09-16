from flask import Flask, request, jsonify
from langchain_community.document_loaders import YoutubeLoader
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
import os
import logging
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize the language model
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY not found in environment variables")

llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")

# Define the prompt template
prompt_template = """
Please provide a concise and informative summary of the content found at the following URL in {language}. The summary should be approximately 300 words and should highlight the main points, key arguments, and any significant conclusions or insights presented in the content. Ensure that the summary is clear and easy to understand for someone who has not accessed the original content.

URL Content:
{text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text", "language"])
summarize_chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)

@app.route('/api/summarize', methods=['POST'])
def summarize_youtube_video():
    logger.info("Received request to /api/summarize endpoint")
    try:
        logger.info("Attempting to parse JSON data from request")
        data = request.get_json()
        logger.info(f"Received data: {data}")

        # Extract YouTube URL and language
        youtube_url = data.get('youtube_url')
        language = data.get('language', 'English')  # Default to English if not specified

        logger.info(f"YouTube URL: {youtube_url}")
        logger.info(f"Language: {language}")

        if not youtube_url:
            logger.warning("No YouTube URL provided in the request")
            return jsonify({"error": "YouTube URL is required"}), 400

        logger.info("Initializing YoutubeLoader")
        loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
        
        logger.info("Loading video content")
        docs = loader.load()

        # Check if any documents were loaded
        if len(docs) == 0:
            logger.error("No content loaded from the YouTube video")
            return jsonify({"error": "Failed to load video content"}), 500

        logger.info(f"Loaded {len(docs)} document(s) from YouTube")

        logger.info("Generating summary")
        summary = summarize_chain.invoke({"input_documents": docs, "language": language})
        logger.info("Summary generated successfully")

        logger.info("Returning summary")
        return jsonify({"summary": summary['output_text']})

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return jsonify({"error": "An internal error occurred"}), 500
@app.route('/api/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

# This is used when running locally. Ignored by Vercel.
if __name__ == '__main__':
    app.run(debug=True)