from flask import Flask, render_template, request
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate

app = Flask(__name__)

# Initialize the llama3 model using ChatOllama
client = ChatOllama(model="llama3")

# Short and long prompt templates
short_prompt_template = """{text}\n\nWrite a short and precise summary for the above text in the form of bullet points properly arranged.\nSummary:"""
long_prompt_template = """{text}\n\nPlease provide a comprehensive summary of the above text using only bullet points properly arranged.  
Ensure that you include all numerical values, amounts, dates, and any other specific numerical information. Also, abbreviate important terms where applicable. 
At the end of the summary, include bullet points highlighting all the main points, ensuring all important timelines, amounts, and dates are included and avoid redundancy.\nSummary:"""

# Tone and sentiment prompt template
tone_prompt_template = """{text}\n\nAnalyze the tone and sentiment of the above text only using 3 of the following tags DO NOT EXPLAIN ANYTHING ELSE: \n[Friendly, Informative, Formal, Polite, Professional, Frustrated, Disappointed, Critical, Immediate action required, High priority, Time-sensitive, Informal, Apologetic, Regretful] \nTone and Sentiment Analysis:"""

# Predefined feedback options
feedback_options = [
    "Important details are missing from the summary.",
    "The summary contains inaccurate information.",
    "The summary is not relevant to the email content.",
    "The summary is too brief.",
    "The summary is too detailed.",
    "Other (Please specify)"
]

# Predefined feedback prompts for short summaries
short_feedback_prompts = [
    """{text}\n\nWrite a short and precise summary for the above text in the form of bullet points properly arranged. Add more details to the summary, add more points.\nSummary:""",
    """{text}\n\nWrite a short and precise summary for the above text in the form of bullet points properly arranged. Ensure the summary is more accurate.\nSummary:""",
    """{text}\n\nWrite a short and precise summary for the above text in the form of bullet points properly arranged. Only summarise from the given content, make it relevant.\nSummary:""",
    """{text}\n\nWrite a short and precise summary for the above text in the form of bullet points properly arranged. Don't make it too short.\nSummary:""",
    """{text}\n\nWrite a short and precise summary for the above text in the form of bullet points properly arranged. Make the summary more concise.\nSummary:""",
    None  # For "Other (Please specify)"
]

# Predefined feedback prompts for long summaries
long_feedback_prompts = [
    """{text}\n\nPlease provide a comprehensive summary of the above text using only bullet points properly arranged. Add as many details as possible to the summary, add more points.\nSummary:""",
    """{text}\n\nPlease provide a comprehensive summary of the above text using only bullet points properly arranged. Make the summary more accurate.\nSummary:""",
    """{text}\n\nPlease provide a comprehensive summary of the above text using only bullet points properly arranged. Only summarise from the given content, make it relevant.\nSummary:""",
    """{text}\n\nPlease provide a comprehensive summary of the above text using only bullet points properly arranged. Keep all the details.\nSummary:""",
    """{text}\n\nPlease provide a comprehensive summary of the above text using only bullet points properly arranged. Make the summary as concise as possible.\nSummary:""",
    None  # For "Other (Please specify)"
]

# Function to generate summaries
def generate_summary(text, is_long=True, feedback_index=None, custom_feedback=None):
    if is_long:
        if feedback_index is not None:
            if feedback_index < len(long_feedback_prompts) - 1:
                prompt_template = long_feedback_prompts[feedback_index]
            else:
                prompt_template = f"""{custom_feedback}\n\n{{text}}\n\nPlease provide a comprehensive summary of the above text using only bullet points properly arranged. Ensure that you include all numerical values, amounts, dates, and any other specific numerical information. Also, abbreviate important terms where applicable. At the end of the summary, include bullet points highlighting all the main points, ensuring all important timelines, amounts, and dates are included and avoid redundancy.\nSummary:"""
        else:
            prompt_template = long_prompt_template
    else:
        if feedback_index is not None:
            if feedback_index < len(short_feedback_prompts) - 1:
                prompt_template = short_feedback_prompts[feedback_index]
            else:
                prompt_template = f"""{custom_feedback}\n\n{{text}}\n\nWrite a short and precise summary for the above text in the form of bullet points properly arranged.\nSummary:"""
        else:
            prompt_template = short_prompt_template

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    summary = prompt | client
    return summary.invoke({"text": text}).content

# Function to analyze tone and sentiment
def analyze_tone(text):
    prompt = PromptTemplate(template=tone_prompt_template, input_variables=["text"])
    analysis = prompt | client
    return analysis.invoke({"text": text}).content

@app.route('/', methods=['GET', 'POST'])
def index():
    short_summary_output = ""
    long_summary_output = ""
    tone_output = ""
    if request.method == 'POST':
        text = request.form['text']
        if 'generate_short' in request.form:
            short_summary_output = generate_summary(text, is_long=False)
        elif 'generate_long' in request.form:
            long_summary_output = generate_summary(text, is_long=True)
        elif 'update_short_summary' in request.form:
            feedback_index = feedback_options.index(request.form['short_feedback'])
            custom_feedback = request.form.get('custom_short_feedback')
            short_summary_output = generate_summary(text, is_long=False, feedback_index=feedback_index, custom_feedback=custom_feedback)
        elif 'update_long_summary' in request.form:
            feedback_index = feedback_options.index(request.form['long_feedback'])
            custom_feedback = request.form.get('custom_long_feedback')
            long_summary_output = generate_summary(text, is_long=True, feedback_index=feedback_index, custom_feedback=custom_feedback)
        elif 'analyze_tone' in request.form:
            tone_output = analyze_tone(text)
    return render_template('index.html', short_summary_output=short_summary_output, long_summary_output=long_summary_output, tone_output=tone_output, feedback_options=feedback_options)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
