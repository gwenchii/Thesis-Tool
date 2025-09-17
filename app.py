from flask import Flask, request, render_template
import joblib
import numpy as np
import re

tagalog_discourse_markers = r"\b(?:at|kung|hanggang|hangga’t|bagama’t|nang|o|kaya|pero|dahil\ sa|dahilan\ sa|gawa\ ng|sapagka’t|upang|sakali|noon|sa\ sandali|magbuhat|magmula|bagaman|maliban|bukod|dangan|dahil|yayamang|kapag|pagka|tuwing|matapos|pagkatapos|porke|maski|imbis|sa\ lugar|sa\ halip|miyentras|para|saka|haba|samantala|bago|kundi)\b"

english_discourse_markers = r"\b(?:and|but|or|so|because|although|however|nevertheless|nonetheless|yet|still|despite\ that|in\ spite\ of\ that|even\ so|on\ the\ contrary|on\ the\ other\ hand|otherwise|instead|alternatively|in\ contrast|as\ a\ result|therefore|thus|consequently|hence|so\ that|in\ order\ that|with\ the\ result\ that|because\ of\ this|due\ to\ this|then|next|after\ that|afterwards|since\ then|eventually|finally|in\ the\ end|at\ first|in\ the\ beginning|to\ begin\ with|first\ of\ all|for\ one\ thing|for\ another\ thing|secondly|thirdly|to\ start\ with|in\ conclusion|to\ conclude|to\ sum\ up|in\ short|in\ brief|overall|on\ the\ whole|all\ in\ all|to\ summarize|in\ a\ nutshell|moreover|furthermore|what\ is\ more|in\ addition|besides|also|too|as\ well|in\ the\ same\ way|similarly|likewise|in\ other\ words|that\ is\ to\ say|this\ means\ that|for\ example|for\ instance|such\ as|namely|in\ particular|especially|more\ precisely|to\ illustrate|as\ a\ matter\ of\ fact|actually|in\ fact|indeed|clearly|surely|certainly|obviously|of\ course|naturally|apparently|evidently|no\ doubt|undoubtedly|presumably|frankly|honestly|to\ be\ honest|luckily|fortunately|unfortunately|hopefully|interestingly|surprisingly|ironically)\b"

all_discourse_markers = tagalog_discourse_markers + "|" + english_discourse_markers

def split_into_clauses(text):
    if not isinstance(text, str):
        return []
    parts = re.split(f"(?:{all_discourse_markers}|,|;|:)", text, flags=re.IGNORECASE)
    merged_clauses, buffer = [], ""
    for part in parts:
        if not isinstance(part, str):
            continue
        part = part.strip()
        if not part:
            continue
        if buffer:
            merged_clauses.append(f"{buffer} {part}".strip())
            buffer = ""
        elif re.fullmatch(all_discourse_markers, part, flags=re.IGNORECASE):
            buffer = part
        else:
            merged_clauses.append(part)
    return merged_clauses

def extract_discourse_markers(text):
    return re.findall(all_discourse_markers, text, flags=re.IGNORECASE)

loaded = joblib.load(r"C:\Users\mynam\Downloads\Thesis Tool\Thesis Dataset Cleaning\Training and Testing\taglish_sentiment_model.pkl")

if isinstance(loaded, dict):
    vectorizer = loaded.get("vectorizer")
    clf = loaded.get("model")
else:
    vectorizer = None
    clf = loaded  


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    overall_sentiment = None

    if request.method == 'POST':
        user_input = request.form['user_input']
        sentences = [s.strip() for s in user_input.split('.') if s.strip()]
        sentiment_scores = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        for sentence in sentences:
            if vectorizer is not None:
                X = vectorizer.transform([sentence])
            else:
                X = [sentence]  

            pred = clf.predict(X)[0]
            prob = clf.predict_proba(X)[0]

            markers_found = extract_discourse_markers(sentence)
            clauses = split_into_clauses(sentence)

            clause_results = []
            for clause in clauses:
                if vectorizer is not None:
                    X_clause = vectorizer.transform([clause])
                else:
                    X_clause = [clause]
                pred_clause = clf.predict(X_clause)[0]
                clause_results.append({'clause': clause, 'sentiment': pred_clause})
                sentiment_scores[pred_clause] += 1

            results.append({
                'sentence': sentence,
                'overall_prediction': pred,
                'discourse_markers': markers_found,
                'clauses': clause_results
            })

        overall_sentiment = max(sentiment_scores, key=sentiment_scores.get)
    
    return render_template('index.html', results=results, overall=overall_sentiment)


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
