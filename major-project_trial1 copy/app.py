from flask import Flask, render_template, request
from models2 import ask_query, translate, listFiles
import os
app = Flask(__name__)

# Variable to store the chat input
chat_input_result = ""
chat_input =""

dict_pairs = {}
trans_output = ""
list_files = {}

@app.route('/', )
def home():
    # if request.method == 'POST' and request.form['submit'] == 'train_files':
    #     f = request.files['upload']
    #     f.save(secure_filename(f.filename))
    #     fname = f.filename
    #     print(f.filename)
    global list_files
    list_files = listFiles()
    return render_template('page1.html',dict_pairs = dict_pairs, trans_output = trans_output,list_files = list_files)


@app.route('/data', methods=['GET,POST'])
def train():
    if request.method == 'POST' and request.form['submit'] == 'train_files':
        f = request.files['upload']
        f.save(secure_filename(f.filename))
        fname = f.filename
        print(f.filename)
    global list_files
    list_files = listFiles()
    return render_template('page1.html',dict_pairs = dict_pairs, trans_output = trans_output,list_files = list_files)


@app.route('/process_input', methods=['POST'])
def process_input():
    # global chat_input_result
    # global chat_input
    global dict_pairs
    chat_input = request.form.get('chatInput')
    
    # Add your processing logic here
    # For demonstration purposes, let's update the variable with the processed input
    user_output = ask_query(chat_input)
    # user_output = {"dummy lang":["some ques","some ans"],"dummy lang2":["some ques","some ans"],"dummy lang 3":["some ques","some ans"],"dummy lang 4":["some ques","some ans"]}
    # user_output
    # chat_input_result = f'{user_output}'
    dict_pairs = user_output
    print(dict_pairs)
    global list_files
    list_files = listFiles()
    return render_template('page1.html', dict_pairs = dict_pairs, trans_output = trans_output, list_files = list_files)

@app.route('/translate_input', methods=['POST'])
def translate_input():
    lang = request.form.get('selectedOption1')
    # y = request.form.get('selectedOption2')
    trans_input = request.form.get('transInput')
    # Add your processing logic here
    # For demonstration purposes, let's update the variable with the processed input
    global trans_output 
    trans_output= translate(trans_input,lang)
    print(lang)
    print(trans_input)
    print(trans_output)
    global list_files
    list_files = listFiles()

    return render_template('page1.html', dict_pairs = dict_pairs, trans_output = trans_output, list_files = list_files)


@app.route('/train_model', methods=['POST'])
def train_model():
    file = request.files['file']
    lang = request.form.get('selectedOption2')
    print(lang)
    
    if file.filename == '':
         return 'No selected file'

    if file:
        # Specify the folder where you want to save the uploaded files
         upload_folder = '/Users/shreysharma/Documents/Shrey SNU/Seventh Sem/Cross-lingual Question Answering Model/cross-lingual-frontend/major-project_trial1/Documents/' + lang
         print(upload_folder)
        # Ensure the folder exists, create it if not
         os.makedirs(upload_folder, exist_ok=True)

        # Save the file to the specified folder
         file.save(os.path.join(upload_folder, file.filename))
         print("file is saved")
         print(file.name)
         global list_files
         list_files = listFiles()
    return render_template('page1.html', dict_pairs = dict_pairs, trans_output = trans_output, list_files = list_files)


if __name__ == '__main__':
    app.run(debug=True)