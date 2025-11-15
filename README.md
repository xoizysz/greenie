# 🌱 Greenie Chat – AI-Powered Climate Action Assistant

Greenie Chat is an AI-powered chatbot that helps people learn about **climate change** and discover **simple, practical actions** they can take in daily life to live more sustainably.

It’s built as part of the **1M1B Green Internship** as a prototype for making **climate education accessible, interactive, and personalized**.

---

## 🎯 Project Theme & SDG Alignment

**Theme:** Sustainable Living & Climate Education  
**Sub-Topic:** AI-powered climate action chatbot for individual-level awareness and guidance  

**Aligned UN SDGs:**

- **SDG 13 – Climate Action** 🌍  
  Educates and empowers individuals to take meaningful steps against climate change.

- **SDG 12 – Responsible Consumption and Production** ♻️  
  Encourages reduced waste, lower resource use, and more conscious lifestyle choices.

- **SDG 4 – Quality Education** 📚  
  Provides easy-to-understand information about climate issues to a wider audience.

---

## 💡 What is Greenie Chat?

Greenie Chat is a simple web-based assistant where users can ask questions like:

> “How can I reduce plastic waste?”  
> “What are easy steps to save energy at home?”  
> “How can I be more sustainable in college?”

The chatbot replies in **plain language**, with **actionable tips** instead of confusing jargon.  
Behind the scenes, it uses **AI + RAG (Retrieval-Augmented Generation)** to fetch relevant climate/sustainability content and generate helpful responses.

---

## ✨ Key Features

- 🌍 **Everyday Climate Help**  
  Ask questions about climate change, sustainability, waste reduction, energy saving, etc.

- 🧠 **AI-Powered Answers**  
  Uses a large language model (Llama 3 via Groq) to generate natural, conversational responses.

- 📚 **RAG-Based Knowledge Retrieval**  
  Uses a vector database (Pinecone) + LangChain to retrieve relevant climate content instead of “making things up”.

- 🔁 **Context-Aware Conversations**  
  Supports follow-up questions and tries to maintain context within a session.

- 🧪 **Prototype-Friendly Setup**  
  Simple Flask backend and minimal frontend – easy to run locally, demo, and extend.

---

## 🧰 Tech Stack

**Core:**

- 🐍 **Python**  
- 🧪 **Flask** – Web framework for the backend API + routes  
- 🧱 **HTML / CSS / JavaScript** – Frontend UI (templates + static assets)

**AI & Retrieval:**

- 🧠 **Llama 3 (via Groq API)** – Large language model for response generation  
- 🔍 **LangChain (RAG)** – Orchestration + retrieval-augmented generation  
- 📦 **Pinecone** – Vector database for semantic search and context retrieval  

> _Note: Exact libraries may be visible in `requirements.txt` and `app.py`._

---

## 📁 Project Structure

> This is a high-level overview based on the current repo layout.

```bash
Greenie-chat/
├─ app.py               # Main Flask application
├─ requirements.txt     # Python dependencies
├─ static/              # CSS, JS, images and other static assets
├─ templates/           # HTML templates (e.g., index.html, chat UI)
├─ LICENSE              # MIT License
└─ README.md            # You are here :)
