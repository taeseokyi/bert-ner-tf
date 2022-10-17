import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

from bert import NerHF_tag

app = Flask(__name__)
CORS(app)

model = NerHF_tag('./out_base_ner_kobart')

@app.route("/predict", methods=['POST'])
def predict():
  text = request.json["text"]
  try:
    out = model.predict(text)
    return jsonify({"result": out})
  except Exception as e: # pylint:disable=broad-except
    print(e)
    print(traceback.format_exc())
    return jsonify({"result": "Model Failed"})


if __name__ == "__main__":
  app.run('0.0.0.0', port=10000)
  # pylint:disable=pointless-string-statement
  '''
    # 사용법
    python api_ner_hf.py

    # 다음과 같이 호출한다.
    curl -X POST http://0.0.0.0:10000/predict -H 'Content-Type: application/json' -d '{ "text": "Steve went to Paris" }'
    curl -X POST http://0.0.0.0:10000/predict -H 'Content-Type: application/json' -d '{ "text": "신경세포(神經細胞) 또는 뉴런(neuron)은 신경계를 구성하는 세포이다." }'
    curl -X POST http://0.0.0.0:10000/predict -H 'Content-Type: application/json' -d '{ "text": "흡인기초는 구조물 내부의 물이나 공기와 같은 유체를 외부로 강제 배출하여 발생되는, 내부와 외부의 압력차를 이용하여 해저에 설치되는 구조물을 의미 한다(Sparrevik, 2002)." }'
    curl -X POST http://0.0.0.0:10000/predict -H 'Content-Type: application/json' -d '{ "text": "흡인 기초 는 구조물 내부 의 물 이나 공기 와 같 은 유체 를 외부 로 강제 배출 하 여 발생 되 는 , 내부 와 외부 의 압력 차 를 이용 하 여 해저 에 설치 되 는 구조물 을 의미 한다 ( Sparrevik , 2002 ) . " }'
    curl -X POST http://0.0.0.0:10000/predict -H 'Content-Type: application/json' -d '{ "text": "흡인기초는구조물내부의물이나공기와같은유체를외부로강제배출하여발생되는,내부와외부의압력차를이용하여해저에설치되는구조물을의미한다(Sparrevik,2002)." }'
    '''
