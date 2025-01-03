{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# pip install streamlit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-01-03 13:49:25.761 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-03 13:49:26.783 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-03 13:49:26.783 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-03 13:49:26.783 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-03 13:49:26.783 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-03 13:49:26.783 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-03 13:49:26.796 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-03 13:49:26.796 Session state does not function when running a script without `streamlit run`\n",
      "2025-01-03 13:49:26.805 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-03 13:49:26.805 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-03 13:49:26.813 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-03 13:49:26.813 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-03 13:49:26.813 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-03 13:49:26.813 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-03 13:49:26.813 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-03 13:49:26.813 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-03 13:49:26.813 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import numpy as np\n",
    "\n",
    "# Add this at the beginning of your code\n",
    "st.set_page_config(\n",
    "    page_title=\"Hate Speech Detection\",\n",
    "    page_icon=\"🔍\",\n",
    "    layout=\"wide\"\n",
    ")\n",
    "\n",
    "# Load the saved model and vectorizer\n",
    "with open('random_forest_model.pkl', 'rb') as model_file:\n",
    "    model = pickle.load(model_file)\n",
    "\n",
    "with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:\n",
    "    vectorizer = pickle.load(vectorizer_file)\n",
    "\n",
    "# Title and input field\n",
    "st.title(\"Hate Speech Detection\")\n",
    "input_text = st.text_area(\"Enter a sentence to analyze:\", height=100)\n",
    "\n",
    "# Analyze sentence\n",
    "if st.button(\"Analyze Sentence\"):\n",
    "    if input_text.strip():\n",
    "        # Transform input using the vectorizer\n",
    "        input_vectorized = vectorizer.transform([input_text])\n",
    "        prediction = model.predict(input_vectorized)[0]\n",
    "        confidence = np.max(model.predict_proba(input_vectorized)) * 100\n",
    "        \n",
    "        # Display the result\n",
    "        prediction_class = 'hate-speech' if prediction == 1 else 'not-hate-speech'\n",
    "        st.markdown(f\"\"\"\n",
    "            <div class=\"prediction-box {prediction_class}\">\n",
    "                <h3>Prediction: {'Hate Speech' if prediction == 1 else 'Not Hate Speech'}</h3>\n",
    "                <p>Confidence: {confidence:.2f}%</p>\n",
    "            </div>\n",
    "        \"\"\", unsafe_allow_html=True)\n",
    "    else:\n",
    "        st.error(\"Please enter a valid sentence.\")\n",
    "\n",
    
    "# Footer\n",
    "st.markdown(\"\"\"\n",
    "<footer>\n",
    "    <p>Developed with ❤️ by [Your Name]</p>\n",
    "</footer>\n",
    "\"\"\", unsafe_allow_html=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "my_env",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
