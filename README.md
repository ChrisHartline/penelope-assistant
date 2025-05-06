# Penelope Assistant

An AI research assistant focused on Bitcoin, AI, and related technologies. Penelope helps you search, summarize, and understand research papers while maintaining a knowledge base of information.

## Features

- 📚 arXiv paper search and summarization
- 🧠 Knowledge base for storing and retrieving information
- 💬 Natural language chat interface
- 🔊 Text-to-speech capabilities
- 🔍 Perplexity integration for general questions

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/penelope-assistant.git
cd penelope-assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```
ANTHROPIC_API_KEY=your_anthropic_key_here
ELEVENLABS_API_KEY=your_elevenlabs_key_here
PPLX_API_KEY=your_perplexity_key_here
OPENAI_API_KEY=your_openai_key_here
```

## Usage

1. Start the interface:
```bash
python ui.py
```

2. Access the interface at `http://localhost:7860`

3. Available commands:
- `search arxiv: [query]` - Search for papers
- `summarize arxiv: [paper_id]` - Get paper summary
- `search kb: [query]` - Search knowledge base
- `check kb` - View knowledge base stats
- `perplexity: [query]` - Ask general questions

## Configuration

The system uses a `config.json` file for settings. This file is automatically created with defaults and can be customized. Sensitive information should be stored in the `.env` file.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

