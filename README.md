# Chat with Any Docs using LlamaIndex and Streamlit

Upload your own set of documents and chat with them using a model of your choice.

It's a modified version of [StreamLit for LlamaIndex example](https://github.com/streamlit/llamaindex-chat-with-streamlit-docs) where you can actually upload your own files (PDF, TXT, DOCX, CSVs, etc.) and chat with them. The original version is using a hardcoded folder.

## Overview of the App

<img src="app.png" width="75%">

- Takes user queries via Streamlit's `st.chat_input` and displays both user queries and model responses with `st.chat_message`
- Uses LlamaIndex to load and index data and create a chat engine that will retrieve context from that data to respond to each user query

## Demo App

TBA

## Get an OpenAI API key

You can get your own OpenAI API key by following the following instructions:

1. Go to https://platform.openai.com/account/api-keys.
2. Click on the `+ Create new secret key` button.
3. Next, enter an identifier name (optional) and click on the `Create secret key` button.
4. Add your API key to your `.streamlit/secrets.toml` file (rename the sample file after you add your key).

> [!CAUTION]
> Don't commit your secrets file to your GitHub repository. The `.gitignore` file in this repo includes `.streamlit/secrets.toml` and `secrets.toml`.

## To launch the app

1. Clone the repo in your Terminal (or using a service like Koyeb)

```
git clone git@github.com:streamlit/llamaindex-chat-with-streamlit-docs.git
```

2. Change to `llamaindex-chat-with-streamlit-docs` directory and install dependencies:

```
pip install -r requirements.txt
```

Before that, you might want to change to a new Python environment not to mess up your current one. For instance, if you're using Conda:

```
conda create -n streamlit
conda activate streamlit
```

3. Once the dependencies are installed, run the app:

```
streamlit run streamlit_app.py
```

It will be available on

```
http://localhost:8501/
```

Once the app is loaded, use the folder on the left to load your own library, Reindex it, and after the indexing is done ask a question using the chat.

Note, that if you're using OpenAI's GPT4o model (the default one), it may be quite expensive for big folders. So either reduce the number of folders or use another model.
