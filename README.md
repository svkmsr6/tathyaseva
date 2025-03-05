# Tathya Seva Playground

A Flask-based AI content verification and generation platform using CrewAI for intelligent agent orchestration and OpenAI model processing.

## Features

- 🔍 **Content Verification**: Fact-check any content with detailed veracity scoring
- ✍️ **Content Generation**: Generate well-researched content on any topic
- 🤖 **AI-Powered**: Utilizes OpenAI's latest GPT-4o model for accurate results
- 🎯 **Intelligent Agents**: CrewAI for orchestrating specialized AI agents
- 🌐 **Web Interface**: Clean, responsive UI for easy interaction

## Quick Start

### Prerequisites

- Python 3.11 or higher
- OpenAI API key (GPT-4o access required)

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd tathya-seva-playground
```

2. Set up environment variables:
```bash
# Required
OPENAI_API_KEY='your-api-key-here'
SESSION_SECRET='your-session-secret'  # For Flask session management
```

3. Dependencies:
All required packages are automatically installed in the Replit environment:
- flask
- crewai
- openai
- python-dotenv
- gunicorn

### Running the Application

The application will automatically start in your Replit environment. If you need to restart it:

1. Stop any running processes
2. Click the "Run" button in Replit

The application will be available at the provided Replit URL.

## API Documentation

### Fact Checking API

**Endpoint**: `/api/fact-check`
**Method**: POST
**Request Body**:
```json
{
    "content": "Text to fact check"
}
```
**Response**:
```json
{
    "veracity_score": 85,
    "details": "Veracity Score: 85 - Detailed analysis of the content's accuracy..."
}
```

### Content Generation API

**Endpoint**: `/api/generate-content`
**Method**: POST
**Request Body**:
```json
{
    "topic": "Topic for content generation"
}
```
**Response**:
```json
{
    "content": "Generated content...",
    "metadata": {
        "topic": "Original topic",
        "timestamp": "2024-03-05T17:40:41",
        "word_count": 500
    }
}
```

## Architecture

The application uses a multi-agent system powered by CrewAI:
- Researcher Agent: Gathers comprehensive information
- Fact Checker Agent: Verifies content accuracy
- Content Writer Agent: Generates high-quality content

All agents utilize OpenAI's GPT-4o model for optimal performance.

## Error Handling and Troubleshooting

### Common Issues

1. JSON Parsing Errors
   - **Symptom**: "JSON.parse: unexpected character" error
   - **Solution**: The application includes robust JSON extraction. If you encounter this error, try:
     - Ensure your input content is properly formatted
     - Wait a few seconds and retry the request
     - Check the server logs for detailed error information

2. API Key Issues
   - **Symptom**: "Invalid API Key" or authentication errors
   - **Solution**: 
     - Verify your OpenAI API key is correctly set in environment variables
     - Ensure your API key has access to GPT-4o
     - Check if you've reached your API rate limits

3. Content Generation Issues
   - **Symptom**: Empty or error responses
   - **Solution**:
     - Keep your input text clear and focused
     - Avoid extremely long inputs (stay under 4000 characters)
     - For fact-checking, provide specific claims to verify

### Logging

The application includes comprehensive logging:
- Detailed JSON parsing logs
- Request/response validation
- AI model interactions
- Error tracking with stack traces

## Development Guidelines

1. Code Structure:
   - Flask routes in `app.py`
   - AI agents in `agents.py`
   - Templates in `/templates`
   - Static assets in `/static`

2. Adding New Features:
   - Follow the modular architecture
   - Add comprehensive error handling
   - Include detailed logging
   - Update tests if applicable

3. Best Practices:
   - Use the provided feedback tools for testing
   - Follow the RESTful API patterns
   - Maintain consistent error handling
   - Keep the UI responsive and user-friendly

## Support

For issues related to:
- Application functionality: Open an issue in the repository
- Replit platform: Contact Replit support
- OpenAI API: Refer to OpenAI documentation

## License

MIT License - feel free to use this project for your own purposes.