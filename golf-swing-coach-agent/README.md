---
title: Golf Swing Coach Agent
emoji: ⛳
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: latest
app_file: app.py
pinned: false
---

# Golf Swing Coach Agent

An AI-powered golf swing analysis app using CrewAI agents and MediaPipe pose detection.

## Features

- **Vision Analysis**: AI-powered swing video analysis
- **Biomechanics Expert**: Identifies technical swing flaws
- **PGA Coach**: Provides actionable coaching advice
- **Streamlit UI**: Easy web interface for video upload

## Local Development

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set OpenAI API key:

   ```bash
   export OPENAI_API_KEY='your-key-here'
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deployment to Hugging Face Spaces

This app is configured for automatic deployment to Hugging Face Spaces.

### Setup

1. Create a new GitHub repository for this project
2. Push the code:

   ```bash
   git remote set-url origin https://github.com/clubjk/golf-swing-coach-agent.git
   git push -u origin main
   ```

3. Create a new Hugging Face Space:
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Docker" as the SDK
   - Connect it to your GitHub repository (`clubjk/golf-swing-coach-agent`)
   - Set the main branch to `main`

4. Add the OpenAI API key as a secret:
   - In your Space settings, go to "Secrets"
   - Add `OPENAI_API_KEY` with your API key value

### Auto-deployment

Any push to the `main` branch of your GitHub repository will automatically trigger a redeployment of the Space.

## Architecture

- **CrewAI**: Multi-agent framework for sequential AI workflows
- **MediaPipe**: Pose estimation for swing analysis
- **LangChain OpenAI**: LLM integration for feedback generation
- **Streamlit**: Web app framework

## Agents

1. **Vision Analyst**: Analyzes swing video and pose metrics
2. **Biomechanics Expert**: Identifies top swing flaws
3. **PGA Teaching Professional**: Provides coaching advice with drills</content>
   <parameter name="filePath">/Users/john/ai/play/golf-swing-coach-agent/README.md
