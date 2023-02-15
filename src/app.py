#必要なモジュールのインポート
import torch
from apple_tomato import transform, Net
from flask import Flask, request, render_template, redirect
import  io
from PIL import Image
import base64

#学習済みモデルをもとに推論を行う
def predict(img):
    #ネットワークの準備
    net = Net().cpu().eval()
    #学習済みモデルの重みを読み込み
    net.load_state_dict(torch.load('./src/apple.pt', map_location=torch.device('cpu')))
    #データ前処理
    img = transform(img)
    img = img.unsqueeze(0) #1次元増やす

    #推論
    y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    return y

#推論したラベルからリンゴかトマトかを返す
def getName(label):
    if label==0:
        return 'リンゴ'
    elif label==1:
        return 'トマト'
#Flaskインスタンス化
app = Flask(__name__)
#アップロードされる拡張子の制限
ALLOWED_EXTENTIONS = set(['png', 'jpg', 'gif', 'jpeg'])
#拡張子が適切かチェック
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENTIONS

#URLにアクセスがあった時の挙動を設定
@app.route('/', methods = [ 'GET', 'POST'])
def predicts():
    #POSTメソッドの場合(画像アップされて判定ボタンが押された)
    if request.method == 'POST':
        #ファイルがなかった場合
        if 'filename' not in request.files:
            return redirect(request.url)

        #ファイルがあったらデータ提出
        file = request.files['filename']
        #ファイルの拡張子チェック
        if file and allowed_file(file.filename):
            #画像ファイルに実行される処理
            #画像読み込みバッファ確保
            buf = io.BytesIO()
            image = Image.open(file).convert('RGB')        
            #画像データをバッファに書き込み
            image.save(buf, 'png')
            #バイナリデータをbase64でエンコードしてutf-8でデコード
            base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            base64_data = 'data:image/png;base64,{}'.format(base64_str)

            #入力された画像に対して推論
            pred = predict(image)
            apple_tomatoName_ = getName(pred)
            return render_template('result.html', apple_tomatoName=apple_tomatoName_, image=base64_data)
        return redirect(request.url)

    #Getメソッドの定義
    elif request.method == 'GET':
        return render_template('index.html')

#アプリケーションの実行定義
if __name__ == '__main__':
    app.run(debug=True)
