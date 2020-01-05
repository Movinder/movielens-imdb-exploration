from flask import Flask, request, render_template

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():

    if request.method == 'POST':
        count = int(request.form.get('people-select'))
        return(render_template('main.html', settings = {'showVote': True, 'people': count, 'buttonDisable': True}))
    elif request.method == 'GET':
       return(render_template('main.html', settings = {'showVote': False, 'people': 0, 'buttonDisable': False}))
    

if __name__ == '__main__':
    app.run()