from flask import Flask, send_file

app = Flask(__name__)

@app.route('/')
def download_csv():
    return send_file('C:\Users\nir67\Documents\GitHub\ASLetter---Generating-ASL-Images-From-Letters\sign_mnist_train.csv', as_attachment=True)

if __name__ == '__main__':
    app.run()