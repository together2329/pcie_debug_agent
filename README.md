# UVM Debug Agent

An AI-powered Universal Verification Methodology (UVM) error analysis and debugging tool that helps identify and fix errors in UVM simulation logs.

## Features

- 🔍 **Intelligent Error Analysis**: Uses advanced AI models to analyze UVM simulation logs and identify root causes
- 📊 **Interactive Dashboard**: Visualize error distributions, patterns, and trends
- 🤖 **AI-Powered Recommendations**: Get suggested fixes and prevention guidelines
- 📄 **Comprehensive Reports**: Generate detailed HTML and Markdown reports
- 🔄 **Real-time Analysis**: Process logs as they are generated
- 🎯 **Context-Aware**: Uses RAG (Retrieval-Augmented Generation) to provide relevant context for analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/uvm-debug-agent.git
cd uvm-debug-agent
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your API keys:
Create a `.env` file in the project root and add your API keys:
```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Configure the analysis settings in the sidebar:
   - Select embedding and LLM models
   - Configure processing parameters
   - Set log directories

4. Run the analysis:
   - Click "Start Analysis" to begin processing
   - View results in the dashboard
   - Download generated reports

## Project Structure

```
uvm-debug-agent/
├── app.py                 # Streamlit application
├── requirements.txt       # Project dependencies
├── README.md             # Project documentation
├── src/
│   ├── collectors/       # Log collection modules
│   ├── processors/       # Document and code processing
│   ├── vectorstore/      # Vector store implementation
│   ├── rag/             # RAG components
│   ├── reports/         # Report generation
│   ├── models/          # Model management
│   └── config/          # Configuration settings
├── data/
│   ├── specs/           # Specification documents
│   └── testbench/       # SystemVerilog testbench files
└── reports/             # Generated reports
```

## Configuration

The application can be configured through the Streamlit interface or by modifying the configuration files:

- **Model Selection**: Choose from various embedding and LLM models
- **Processing Settings**: Configure chunk sizes, overlap, and retrieval parameters
- **Log Directories**: Specify directories containing UVM simulation logs
- **Advanced Settings**: Fine-tune batch sizes, parallel processing, and other parameters

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Uses [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- Powered by [OpenAI](https://openai.com/) and [Anthropic](https://www.anthropic.com/) models