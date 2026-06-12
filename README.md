## Directory Structure
```text
Agentic-RAG-QA-System/
├── code/                        # Source code directory [cite: 131]
│   ├── backend/                 # Backend logic package
│   │   ├── __init__.py
│   │   ├── retrieval.py         # Retrieval System 
│   │   ├── generation.py        # Generation System 
│   │   └── data_loader.py      
│   ├── web_ui.py                # Main Streamlit Application
│   ├── requirements.txt             # Python dependencies   
│   ├── mock_web_ui.py           # Standalone UI for demo
│   └── rag_bridge.py            # Bridge controller connecting UI and Backend
├── data/                        
│   └── hq_small_train.json     
└── README.md
```

## Running the Mock_Web
```text
streamlit run code/mock_web_ui.py
