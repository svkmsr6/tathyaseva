import os
import logging
from flask import Flask, request, jsonify, render_template
from ai_router import AIRouter
from agents import ResearchCrew

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")

# Initialize AI Router
ai_router = AIRouter()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/fact-check', methods=['POST'])
def fact_check():
    try:
        data = request.get_json()
        if not data or 'content' not in data:
            return jsonify({'error': 'No content provided'}), 400

        content = data['content']
        research_depth = data.get('research_depth', 'medium')

        # Get appropriate model from router
        model_type = ai_router.get_model(research_depth)
        
        # Initialize crew with selected model
        crew = ResearchCrew(model_type)
        
        # Run fact checking process
        result = crew.run_fact_check(content)
        
        return jsonify({
            'veracity_score': result['score'],
            'details': result['details']
        })

    except Exception as e:
        logger.error(f"Error in fact-check endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-content', methods=['POST'])
def generate_content():
    try:
        data = request.get_json()
        if not data or 'topic' not in data:
            return jsonify({'error': 'No topic provided'}), 400

        topic = data['topic']
        research_depth = data.get('research_depth', 'medium')
        
        # Get appropriate model from router
        model_type = ai_router.get_model(research_depth)
        
        # Initialize crew with selected model
        crew = ResearchCrew(model_type)
        
        # Generate content
        result = crew.generate_content(topic)
        
        return jsonify({
            'content': result['content'],
            'metadata': result['metadata']
        })

    except Exception as e:
        logger.error(f"Error in generate-content endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500
