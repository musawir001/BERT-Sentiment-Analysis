import torch
import torch.nn.functional as F
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load local model
model_path = "sentiment_model"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)[0]

    negative_prob = probs[0].item()
    positive_prob = probs[1].item()

    difference = abs(positive_prob - negative_prob)
    confidence = max(positive_prob, negative_prob)

    if difference < 0.15:
        sentiment = "Neutral 😐"
    elif positive_prob > negative_prob:
        sentiment = "Positive 😊"
    else:
        sentiment = "Negative 😡"

    return f"""
Sentiment: {sentiment}

Confidence: {round(confidence*100,2)}%

Positive: {round(positive_prob*100,2)}%
Negative: {round(negative_prob*100,2)}%
"""

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Enter text here..."),
    outputs="text",
    title="BERT Sentiment Analyzer",
    description="AI-powered sentiment classification with confidence score."
)

demo.launch()
