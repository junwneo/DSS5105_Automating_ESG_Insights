# 🧠 AI-Powered ESG Dashboard

## 🌍 Project Overview

The **AI-Powered ESG Dashboard** is an interactive web application designed to analyze and visualize Environmental, Social, and Governance (ESG) performance across companies. It combines real-time data processing, machine learning predictions, and advanced data visualization to provide insights into corporate sustainability, trends, and their financial impact.

---

## 🎥 Website Walkthrough

To get a quick overview of how the ESG Dashboard works, check out the tutorial video below:

➡️ [📽️ Watch the Tutorial on YouTube](https://youtu.be/0GBvJovISb4)

> _This video walks you through the dashboard’s main features and how to get started using it._

---

### ✨ Features

- View and compare ESG scores across top-performing companies.
- Explore ESG trends over time.
- Analyze correlations between ESG performance and financial metrics.
- Generate AI-powered forecasts and insights.
- Upload ESG reports and receive summary breakdowns and predictions.

---

## 🗂 Project Structure

```text
DSS5105_AUTOMATING_ESG_INSIGHTS/
├── api/                   # API route definitions and backend endpoints
├── app/                   # Application logic and model integration
├── Architecture_Flow/     # Diagrams and design documentation
├── frontend/              # HTML pages, JS charts, UI components
├── logs/                  # Auto-generated logs for both backend and frontend activity
├── notebooks/             # Jupyter notebooks for experimentation and EDA
├── sql/                   # SQL scripts for database setup or queries
├── tests/                 # Unit tests and integration tests
├── .gitignore             # Git ignore rules
├── DESCRIPTION.txt        # Short project description or metadata
├── README.md              # Main project documentation
```

## 📁 Key Directories Explained

### `frontend/` – ESG Dashboard Interface

The `frontend` directory contains the main interface of the ESG Dashboard. It includes HTML pages, JavaScript-powered visualizations, and embedded components for user interaction.

```text
frontend/
├── assets/                  # Frontend resources
│   ├── css/                # Global styling
│   ├── data/               # Data retrieve from various notebooks to be used to display on the frontend
│   └── js/                 # Core dashboard logic and interactivity
│       ├── esg-analyzer.js
│       ├── esg-filter.js
│       ├── esg-prediction-chart.js
│       ├── esg-reports.js
│       ├── esg-uploader.js
│       ├── finance-impact-chart.js
│       ├── load-esg-table.js
│       └── main.js
├── ai-forecast.html         # Page for AI-based ESG forecasting
├── esg-scores.html          # ESG scoring visualization by company
├── finance-impact.html      # ESG vs financial metrics view
├── trends.html              # Time series ESG trend plots
├── reports.html             # Upload & analyze ESG reports (PDF)
├── index.html               # Homepage and navigation entry point
├── header.html              # Shared navigation/header component
├── script.js                # Standalone JS script (non-modular) - for initial testing
├── method.png               # Image illustrating ESG rating methodology
├── esg_trend_plot.png       # ESG trend graphic used in reports or visual aids
├── sample_esg_plot.py       # Python script for quick ESG plot generation
├── open-webui-main/         # Embedded AI assistant (Open WebUI instance)
└── venv/                    # Local Python environment for frontend tooling
```

### 🔧 JavaScript Module Descriptions

| JS File                    | Purpose                                                                 |
|---------------------------|-------------------------------------------------------------------------|
| `esg-analyzer.js`         | Extracts and processes ESG data from uploaded PDF reports using GPT.    |
| `esg-filter.js`           | Enables dropdown filters and reset functionality for ESG company search.|
| `esg-prediction-chart.js` | Creates predictive ESG visualizations using model output.               |
| `esg-reports.js`          | Builds ESG summary tables and bar charts for the report view.           |
| `esg-uploader.js`         | Handles PDF uploads and triggers insight generation.                    |
| `finance-impact-chart.js` | Visualizes relationships between ESG and financial metrics.             |
| `load-esg-table.js`       | Injects ESG company data into score tables dynamically.                 |
| `main.js`                 | Initializes dashboard features and controls sidebar/UI toggling.        |


### 📡 API Folder Structure

```text
api/
├── node_modules/           # Node.js dependencies (for server.js)
├── .env                    # Environment variables for API keys and secrets
├── config.json             # Configuration file for model settings or routes
├── DESCRIPTION.txt         # API-side notes or overview
├── esg.db                  # Local SQLite database with ESG scores and financial metrics
├── model_api.py            # Python API for serving ML predictions and preprocessing logic
├── model_overall.pkl       # Pretrained ML model used for ESG forecasting
├── package.json            # Node project config, defines scripts and dependencies
├── package-lock.json       # Locked dependency versions for reproducibility
└── server.js               # Express.js server that serves static frontend files
```

### ⚙️ Key API Components

| File / Folder         | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `node_modules/`       | Node.js dependencies auto-installed via `npm install`                      |
| `.env`                | Stores environment variables like API keys, DB paths, or port settings     |
| `config.json`         | Configuration file for server routes, thresholds, or ML model parameters   |
| `DESCRIPTION.txt`     | Overview or notes about API functionality and architecture                 |
| `esg.db`              | SQLite database storing ESG scores, financial indicators, and metadata     |
| `model_api.py`        | Python Flask API serving ML predictions and handling PDF text processing   |
| `model_overall.pkl`   | Serialized machine learning model used for ESG scoring and forecasting     |
| `package.json`        | Node project setup – scripts, metadata, and required packages              |
| `package-lock.json`   | Lock file ensuring consistent dependency versions across installs          |
| `server.js`           | Node.js Express server to serve frontend assets and provide basic routing  |


## 🚀 Getting Started

This project uses **Node.js** to serve the frontend and **Python (Flask)** to run the ESG prediction API. You’ll need both running concurrently in separate terminals.

---

### 🧰 Prerequisites

- Python **3.11.10** with `venv`
- Node.js & npm
- `pip` (Python package installer)
- `sqlite3` (used for the ESG database)

### 🛠️ Installation

### 📦 Step 1: Clone the Repository
```bash
git clone https://github.com/junwneo/DSS5105_Automating_ESG_Insights.git
cd DSS5105_Automating_ESG_Insights
```

### 🧠 Step 2: Set Up Python Environment (3.11.10)

---

🐍 Python Environment Setup (Python 3.11.10)

This project requires **Python 3.11.10**. Follow the steps below to create a virtual environment with that specific version and install the required dependencies.

---

Check if Python 3.11.10 is available:
```bash
python3.11 --version
```

If it's not installed, download it from python.org
or install via a package manager:
* macOS (Homebrew):
```bash
brew install python@3.11
```

* Ubuntu/Debian:
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11
```

🛠 Create a Virtual Environment

From the /api/ folder:
```bash
python3.11 -m venv venv
```
This creates a virtual environment in the venv/ directory using Python 3.11.10

🔑 Activate the Virtual Environment
* macOS/Linux:
    ```bash
    source venv/bin/activate
    ```
* Windows (CMD):
    ```bash
    venv\Scripts\activate.bat
    ```
* Windows (PowerShell):
    ```bash
    venv\Scripts\Activate.ps1
    ```
You will now see (venv) in your terminal prompt.

📦 Install Dependencies

Once activated:
```bash
pip install -r requirements.txt
```
---

### 🧱 Step 3: Set Up Node.js Dependencies

From the `/api/` folder:
```bash
npm install
```
This installs all dependencies listed in `package.json`, including:
* `express` - to serve frontend assets
* `cors` - to handle cross-origin requests
* `sqlite3` - to connect to the local ESG database

---

🏃 **Running the Application**

You’ll need **three terminals** running at the same time:

### 🖥 Terminal 1 – Run the Node.js Frontend Server
From the `/api/` folder:
```bash
cd api
node server.js
# or for auto-reload during development:
npx nodemon server.js
```
You should see:
```arduino
✅ ESG API running at http://localhost:3001
```

### 🧠 Terminal 2 - Run the Python ML API
Still in the `/api/` folder (with your virtual environment activated):
```bash
python model_api.py
```
You should see something like:
```csharp
* Running on http://127.0.0.1:5000
```

### 🌐 Terminal 3 – Serve the Frontend Files
From the `/frontend/` folder:
```bash
cd frontend
python3 -m http.server
```

Visit
```csharp
http://localhost:8000
```
This will open your dashboard homepage
    
✅ Make sure both backend servers (Node + Python) are running before accessing the dashboard for full functionality.

---

### Step 4: 🧠 Setting Up AI Assistant (Open WebUI + Ollama)

This project integrates [Open WebUI](https://github.com/open-webui/open-webui) with [Ollama](https://ollama.com/) to provide a conversational AI assistant for ESG forecasting and report breakdowns.

🔄 Step 1: Install and Run Ollama

Download and install Ollama from [https://ollama.com](https://ollama.com).

Then launch your model (e.g. `llama3`):
```bash
ollama run llama3
```

🐳 Step 2: Start Open WebUI with Docker
Use the following command to launch the WebUI container and connect it to Ollama
```bash
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

This does the following:
* Maps port 3000 on your machine to WebUI's internal port 8080
* Links the Docker container to your local Ollama instance via `host.docker.internal`
* Automatically restarts the container if it stops
* Persists data in the named volume `open-webui`

Can now access the AI assistant at:
* http://localhost:3000

💡 Notes

* Ensure ollama is running with a model before launching the Docker container.

* If you're on Linux and host.docker.internal doesn’t work, replace it with 172.17.0.1 or your Docker bridge IP.

* This interface is integrated into your ESG dashboard via ai-forecast.html.*

---

### 🔌 Port Reference Table

| Port       | Component             | Purpose / Description                                         |
|------------|------------------------|----------------------------------------------------------------|
| `8000`     | Frontend (Python `http.server`) | Serves the ESG Dashboard UI via `index.html`              |
| `3001`     | Node.js Server (`server.js`)     | Serves static frontend files and handles proxy routing     |
| `5050`     | Python API (`model_api.py`)      | Flask server exposing ESG ML prediction endpoints          |
| `3000`     | Open WebUI (Docker)             | Accessible AI assistant interface connected to Ollama      |
| `8080`     | Internal WebUI port (Docker)    | Internal port inside the container, mapped to host `3000`  |
| `11434`    | Ollama                          | Local Ollama model API (used by Open WebUI)                |

### 📄 Example `.env` File

📝 To get started:
1. Copy `.env.example` to `.env`
2. Fill in your actual OpenAI API key

```bash
cp api/.env.example api/.env
```