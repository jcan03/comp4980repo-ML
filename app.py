from flask import Flask, render_template, request, jsonify
import pandas as pd
import model  # import model.py
import os
import time

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        keystrokes = data.get('keystrokes')

        # convert the keystroke data to a DataFrame
        df = pd.DataFrame(keystrokes)

        # extract calculated metrics from the data (some are just placeholders to match the clean.csv)
        typing_speed = df['typingSpeed'].iloc[0]
        afTap = df['afTap'].iloc[0]
        sTap = df['sTap'].iloc[0]
        nqScore = df['nqScore'].iloc[0]
        flightTimeVariability = df['flightTimeVariability'].iloc[0]
        holdingLatencySD = df['holdingLatencySD'].iloc[0]
        releaseLatencySD = df['releaseLatencySD'].iloc[0]
        avgHoldDuration = df['avgHoldDuration'].iloc[0]
        backspaceRate = df['backspaceRate'].iloc[0]

        # generate a unique file name  
        timestamp = str(int(time.time()))
        csv_filename = f'typing_{timestamp}.csv'
        csv_path = os.path.join('static', csv_filename)

        # save the keystroke data to a CSV file
        df.to_csv(csv_path, index=False) 

        # create a summary DataFrame (once again, some are just placeholders to match the clean.csv)
        summary_df = pd.DataFrame([{
            'num': 0,
            'pID': df['pID'].iloc[0],
            'file_2': df['file_2'].iloc[0],
            'gt': df['gt'].iloc[0],
            'updrs108': df['updrs108'].iloc[0],
            'afTap': afTap,
            'sTap': sTap,
            'nqScore': nqScore,
            'typingSpeed': typing_speed, 
            'flightTimeVariability': flightTimeVariability, 
            'holdingLatencySD': holdingLatencySD,
            'releaseLatencySD': releaseLatencySD,
            'avgHoldDuration': avgHoldDuration,
            'backspaceRate': backspaceRate,
            'dataset': df['dataset'].iloc[0],
            'file_1': df['file_1'].iloc[0],
        }])

        # save the summary CSV file
        summary_path = f'static/summary_{timestamp}.csv'
        summary_df.to_csv(summary_path, index=False)

        # call the prediction function from model.py based on the saved csv
        result = model.predict_new_client(csv_path)

        return jsonify({'prediction': str(result)}) # return prediction result
    except Exception as e:
        return jsonify({'error': str(e)}), 500 # if theres an issue

if __name__ == '__main__':
    app.run(debug=True)
