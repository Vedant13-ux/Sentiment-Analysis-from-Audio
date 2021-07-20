from logging import debug
from flask import Flask, views, request, render_template,send_file, send_from_directory, safe_join, abort
import lxml
from werkzeug.utils import secure_filename
import tensorflow as tf
import os 
from model import model

# Initializing the Flask Application 
app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = './audio'


@app.route('/upload', methods=["GET","POST"])
def upload_api():
    if(request.method=="GET"):
        return render_template('index.html')

    if(request.method=="POST"):
        audio=request.files['file']
        audio.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio.filename)))
        nlp_model=model.Model(secure_filename(audio.filename))
        transcript=nlp_model.audio_to_text()
        results=nlp_model.sentiment_analysis()
        filename=nlp_model.return_filename()
        return render_template('analysis.html', transcript=transcript, label=results['label'], score=results['score'], fileURL=f'/getfile/{filename}')

# @app.route('/prediction', methods=["GET","POST"])
# def results():
#     if(request.method=="GET"):
#         return render_template('analysis.html')


@app.route('/getfile/<filename>', methods=["GET"])
def file(filename): 
    print(filename)
    if(request.method=="GET"):
        return send_file(f'./audio/{filename}')


if __name__=="__main__":
    app.run(debug=True)
