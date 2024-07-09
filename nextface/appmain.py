from flask import Flask,render_template,request,redirect
from os.path import join, dirname, realpath
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import load_model,model_from_json
import tensorflow as tf
from scipy.stats import zscore
import pandas as pd
import librosa
import pickle
from  keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from optimizer import Optimizer
from config import Config


UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'static/')
ALLOWED_EXTENSIONS = {'jfif', 'png', 'jpg', 'jpeg'}

def mel_spectrogram(y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128, fmax=4000):
    mel_spect = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2
    mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels, fmax=fmax) 
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)  
    return mel_spect

def frame(x, win_step=128, win_size=64):
    nb_frames = 1 + int((x.shape[2] - win_size) / win_step)
    frames = np.zeros((x.shape[0], nb_frames, x.shape[1], win_size)).astype(np.float32)
    for t in range(nb_frames):
        frames[:,t,:,:] = np.copy(x[:,:,(t * win_step):(t * win_step + win_size)]).astype(np.float32)
    return frames

def get_key_speech(value):
    dictionary={'neutral':0,'calm':1,'happy':2,'sad':3,'angry':4,'fear':5,'disgust':6,'surprise':7}
    for key,val in dictionary.items():
          if (val==value):
            return key

def get_key_text(value):
    dictionary={'happy':0,'angry':1,'love':2,'sad':3,'fear':4,'surprise':5}
    for key,val in dictionary.items():
          if (val==value):
            return key

def recommend(value):
    if (value=='happy' or value=='Happy'):
        ans="Watch a funny movie or TV show to keep the positive emotions flowing.\n\
Meet up with friends and family to spend quality time together.\n\
Plan a fun activity, such as going to a theme park or playing a game outdoors.\n\
Listen to upbeat music and dance to your favorite songs.\n\
Practice gratitude by writing down three things you are thankful for each day."
    elif (value=='angry' or value=='Angry'):
        ans="Take a deep breath and count to 10 to calm down before reacting.\n\
Go for a run or engage in physical activity to release pent-up energy.\n\
Identify the source of your anger and express your feelings in a constructive manner.\n\
Practice relaxation techniques, such as meditation or deep breathing.\n\
Consider seeking help from a therapist or counselor if your anger is interfering with your daily life."     
    elif (value=='fear' or value=='Fear'):
        ans="Identify the source of your fear and challenge negative thoughts with positive ones.\n\
Practice relaxation techniques, such as deep breathing or visualization exercises.\n\
Seek support from a trusted friend or family member to talk about your fears.\n\
Engage in activities that make you feel safe and secure, such as spending time with a pet or in nature.\n\
Consider seeking help from a therapist or counselor if your fears are interfering with your daily life."
    elif (value=='disgust' or value=='Disgust'):
        ans="Identify any underlying beliefs or values that may be contributing to your disgust.\n\
Consider whether there may be cultural or societal influences shaping your reaction.\n\
If appropriate, try to reframe the situation or object in a more positive light to reduce your disgust.\n\
Experiment with different coping strategies, such as humor or cognitive reappraisal, to help regulate your emotions."
    elif (value=='surprise' or value=='Surprise'):
        ans="Negative:\n\
Reach out to a trusted friend or family member to talk about your feelings.\n\
Take a deep breath and try to stay calm.\n\
Seek professional help, such as therapy or counseling, if your surprise is causing distress or interfering with your daily life.\n\
Positive:\n\
Express gratitude for the surprise and the positive feelings it brings.\n\
Share your excitement with friends and family to spread the positive vibes.\n\
Reflect on the surprise and the positive feelings it brings to cultivate a sense of happiness and contentment."
    elif (value=="love" or value=="Love"):
        ans="Spend quality time with loved ones to nurture your relationships.\n\
Practice acts of kindness, such as writing a thoughtful note or doing something nice for someone.\n\
Do something you enjoy to boost your mood and show yourself some love.\n\
Give back to your community by volunteering or donating to a charitable cause.\n\
Practice self-care by taking care of your physical and emotional needs."      
    elif (value=='sad' or value=='Sad'):
        ans="Take a break and allow yourself to feel your emotions without judgment.\n\
Write down your thoughts and feelings in a journal to process them.\n\
Reach out to a trusted friend or family member to talk about your feelings.\n\
Engage in self-care activities, such as taking a relaxing bath or medication.\n\
Seek professional help, such as therapy or counseling, if your feelings persist."     
    elif (value=='calm' or value=='Calm'):
        ans="Take a break from technology and spend time in nature to connect with your surroundings and promote a sense of calm.\n\
Practice gratitude by focusing on the positive aspects of your life and expressing appreciation for the people and experiences that bring you joy.\n\
Use positive self-talk to reinforce feelings of calm and inner peace.\n\
Consider engaging in hobbies or activities that bring you peace and tranquility.\n\
Connect with loved ones and spend time in meaningful conversations or shared experiences."
    elif (value=='Neutral' or value=='neutral'):
        ans="Consider trying out a new hobby or activity that you've always been interested in but haven't had the chance to pursue.\n\
Take some time to reflect on your current goals and priorities, and make adjustments as needed.\n\
Connect with friends or loved ones and make plans to spend quality time together.\n\
Practice mindfulness or meditation to increase your awareness and presence in the moment.\n\
Engage in regular physical exercise or movement to boost your mood and energy levels."
    else:
        ans=" "
    return ans

def predictText(value):
    Textmodel = model_from_json(open("text_model.json", "r").read())
    Textmodel.load_weights('text_model.h5')
    with open('tokenizer.pickle','rb') as handle:
        tokenizer = pickle.load(handle)
    sentence_lst=[]
    sentence_lst.append(value)
    sentence_seq=tokenizer.texts_to_sequences(sentence_lst)
    sentence_padded=pad_sequences(sentence_seq,maxlen=80,padding='post')
    ans=Textmodel.predict(sentence_padded)
    ans1 = np.argmax(ans, axis=1)
    k=get_key_text(ans1)
    return k

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
app.secret_key = 'sameer'

 
@app.route('/')
def main():
    return render_template('index.html')


@app.route('/Imagepredict',  methods=['POST', 'GET'])
def uploadFile():
    if request.method == 'POST':
        if 'uploaded-file' not in request.files:
            return redirect(request.url)
        uploaded_img = request.files['uploaded-file']
        if uploaded_img.filename == '':
            return redirect(request.url)
        
        uploaded_img.save('static/file.jpg')
        img1 = cv2.imread('static/file.jpg')
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
        faces = cascade.detectMultiScale(gray, 1.1, 3)
        for x,y,w,h in faces:
            cv2.rectangle(img1, (x,y), (x+w, y+h), (0,255,0), 2)
            cropped = img1[y:y+h, x:x+w]
        cv2.imwrite('static/after.jpg', img1)
        try:
            cv2.imwrite('static/cropped.jpg', cropped)
        except:
            pass
        try:
            image = cv2.imread('static/cropped.jpg', 0)
        except:
            image = cv2.imread('static/file.jpg', 0)
        image = cv2.resize(image, (48,48))
        image = image/255.0
        image = np.reshape(image, (1,48,48,1))
        Imagemodel = model_from_json(open("emotion_model1.json", "r").read())
        Imagemodel.load_weights('model.h5')
        prediction = Imagemodel.predict(image)
        label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
        prediction = list(prediction[0])
        img_index = prediction.index(max(prediction))
        final_prediction=label_dict[img_index]
        rec=recommend(final_prediction)
        outimg=final_prediction+".gif"
        return render_template('Imagepredict.html', data=final_prediction,im=outimg,inf=rec)


@app.route('/contact')
def main2():
    return render_template('contact.html')


@app.route('/about')
def main3():
    return render_template('about.html')


@app.route('/run-code', methods=['POST'])
def run_code():
    Imagemodel = model_from_json(open("emotion_model1.json", "r").read())
    Imagemodel.load_weights('model.h5')
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            return render_template('index.html')
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w] 
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = Imagemodel.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return render_template('index.html')


@app.route('/TextEmo')
def textEmo():
    return render_template('TextEmo.html')


@app.route('/textPred',methods=['post'])
def textPred():
    sentence=request.form['input1']
    k=predictText(sentence)
    rec=recommend(k)
    out=k+".gif"
    return render_template('textPred.html',data=k,outimg=out,sent=sentence,inf=rec)


@app.route('/SpeechEmotion',methods=['POST', 'GET'])
def speechEmo():
     if request.method == 'POST':
        if 'uploaded-audio' not in request.files:
             return redirect(request.url)
        uploaded_aud = request.files['uploaded-audio']
        if uploaded_aud.filename == '':
             return redirect(request.url)

        uploaded_aud.save('static/audio.wav')
        Speechmodel = model_from_json(open("speech_model_json.json", "r").read())
        Speechmodel.load_weights('speech_emo_weights.h5')
        s = []
        sample_rate = 16000     
        max_pad_len = 49100
        win_ts = 128
        hop_ts = 64
        path_="static/audio.wav" 
        X, sample_rate = librosa.load(path_,duration=3,offset=0.5)
        sample_rate = np.array(sample_rate)
        y = zscore(X)
        if len(y) < max_pad_len:    
            y_padded = np.zeros(max_pad_len)
            y_padded[:len(y)] = y
            y = y_padded
        elif len(y) > max_pad_len:
            y = np.asarray(y[:max_pad_len])
        s.append(y)
        mel_spect = np.asarray(list(map(mel_spectrogram, s)))
        x = frame(mel_spect, hop_ts, win_ts)
        x = x.reshape(x.shape[0], x.shape[1] , x.shape[2], x.shape[3], 1)
        preds = Speechmodel.predict(x)
        preds=preds.argmax(axis=1)
        preds=get_key_speech(preds)
        rec=recommend(preds)
        outputimg=preds+".gif"
        return render_template('SpeechEmotion.html', data=preds,im=outputimg ,inf=rec)
 

@app.route('/aud1')
def aud1():
    return render_template('aud1.html')


@app.route('/predict',  methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        if 'upload-3dfile' not in request.files:
            return redirect(request.url)
        uploaded_img = request.files['upload-3dfile']
        if uploaded_img.filename == '':
            return redirect(request.url)
        uploaded_img.save('static/3dfile.jpg')
    config = Config()
    config.fillFromDicFile('C:/Users/user/Desktop/nextface/optimConfig.ini')
    config.device = 'cpu'
    config.path = 'C:/Users/user/Desktop/nextface/baselMorphableModel/'
    outputDir = 'C:/Users/user/Desktop/nextface/static/output/' 
    optimizer = Optimizer(outputDir ,config)
    imagePath = 'C:/Users/user/Desktop/nextface/static/3dfile.jpg'
    optimizer.run(imagePath)
    return render_template('predict.html')


@app.route('/upload-audio', methods=['POST', 'GET'])
def upload_audio():
    audio_file = request.files['audio']
    audio_file.save('static/audio1.wav')
    pass


@app.route('/RealTimeSpeech', methods=['POST', 'GET'])
def realtimespeech():
    Speechmodel1 = model_from_json(open("speech_model_json.json", "r").read())
    Speechmodel1.load_weights('speech_emo_weights.h5')
    s = []
    sample_rate = 16000     
    max_pad_len = 49100
    win_ts = 128
    hop_ts = 64
    path1="static/audio1.wav" 
    X, sample_rate = librosa.load(path1,duration=3,offset=0.5)
    sample_rate = np.array(sample_rate)
    y = zscore(X)
    if len(y) < max_pad_len:    
        y_padded = np.zeros(max_pad_len)
        y_padded[:len(y)] = y
        y = y_padded
    elif len(y) > max_pad_len:
        y = np.asarray(y[:max_pad_len])
    s.append(y)
    mel_spect = np.asarray(list(map(mel_spectrogram, s)))
    x = frame(mel_spect, hop_ts, win_ts)
    x = x.reshape(x.shape[0], x.shape[1] , x.shape[2], x.shape[3], 1)
    preds = Speechmodel1.predict(x)
    preds=preds.argmax(axis=1)
    preds=get_key_speech(preds)
    rec=recommend(preds)
    outputimg=preds+".gif"
    return render_template('RealTimeSpeech.html', data=preds,im=outputimg,inf=rec)
    

if __name__ == "__main__":
    app.run(debug=True)
