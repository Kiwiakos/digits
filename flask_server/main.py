#!/home/kyriakos/.conda/envs/flask/bin/python -B

from flask import Flask, render_template
from PIL import Image
from io import BytesIO as IO
import base64
import numpy as np
from matplotlib import pyplot as plt

from keras.models import load_model, Sequential

app = Flask(__name__)

@app.get('/')
@app.get('/home')
def home():
    return render_template('home.html')

@app.route('/draw')
def draw():
    return render_template('draw.html')

@app.route('/show/<data>')
def show(data: str):
    d = data
    d = d.replace('_', '/')
    d = d.replace('-', '+')
    d = d.replace('#', '=')
    d += "=" * ((4 - len(d) % 4) % 4)
    i = Image.open(IO(base64.b64decode(d)))
    i.save('static/img.png')
    i28 = i.resize((28, 28)).convert('L')
    i28.save('static/img28.png')
    n28 = 1. - np.array(i28) / 255.
    n28 = np.reshape(n28, (1, 28, 28, 1))
    mdl: Sequential = load_model('mnist3.mdl')
    f = 100 * mdl.predict(n28).flatten()
    t = [k for k in range(10)]

    fg, ax = plt.subplots()
    ax.barh(t, f)
    ax.set_yticks(t)
    ax.set_xlim(0, 100)
    ax.grid()
    fg.savefig('static/bar.png', dpi=100)
    plt.close(fg)

    my_num = plt.imread("static/img28.png")
    my_num = 1.0 - my_num

    l = mdl.layers[0]
    w = l.weights[0].numpy().squeeze()
    b = l.weights[1].numpy().squeeze()
    N0, _, K0 = w.shape
    f0 = np.array([[[np.clip(np.sum(my_num[i:i+N0, j:j+N0] * w[:, :, k]) + b[k], -0, 999) for j in range(28-N0+1)] for i in range(28-N0+1)] for k in range(K0)])
    K1, N1, _ = f0.shape
    S1 = 2
    f1 = np.array([[[np.max(f0[k,i:i+S1-1,:][:,j:j+S1-1]) for j in range(0, N1-S1+1, S1)] for i in range(0, N1-S1+1, S1)]for k in range(K1)])
    K2, N2, _ = f1.shape

    mx = my_num.max()
    mn = my_num.min()
    fg, ax = plt.subplots()
    ax.pcolormesh(my_num[::-1, :], cmap='Reds', vmin=mn, vmax=mx)
    for i in range(29):
        ax.plot([0, 29], [i, i], 'w') 
        ax.plot([i, i], [0, 29], 'w') 
    ax.set_axis_off()
    ax.set_aspect('equal', 'box')
    fg.tight_layout()
    fg.savefig(f'static/lay{0}.png', dpi=70)
    plt.close(fg)

    mx = w.max()
    mn = w.min()
    for k in range(K0):
        fg, ax = plt.subplots()
        ax.pcolormesh(w[::-1, :, k], cmap='Reds', vmin=mn, vmax=mx)
        for i in range(N0+1):
            ax.plot([0, N0], [i, i], 'w') 
            ax.plot([i, i], [0, N0], 'w') 
        ax.set_axis_off()
        ax.set_aspect('equal', 'box')
        fg.tight_layout()
        fg.savefig(f'static/lay{0}_weight{k}.png', dpi=70)
        plt.close(fg)

    mx = f1.max()
    mn = f1.min()
    for k in range(K2):
        fg, ax = plt.subplots()
        ax.pcolormesh(f1[k, ::-1, :], cmap='Reds', vmin=mn, vmax=mx)
        for i in range(N2+1):
            ax.plot([0, N2], [i, i], 'w') 
            ax.plot([i, i], [0, N2], 'w') 
        ax.set_axis_off()
        ax.set_aspect('equal', 'box')
        fg.tight_layout()
        fg.savefig(f'static/lay{0}_out{k}.png', dpi=70)
        plt.close(fg)

    return render_template('show.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9999)