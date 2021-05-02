import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle
from hackfest_final import Multiple_regressor
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
cropsDict = {
				"rice": ["rice.jpg", "Rice", "Rice Description", "https://en.wikipedia.org/wiki/Rice"],
				"maize": ["maize.png", "Maize", "Maize Description", "https://en.wikipedia.org/wiki/Maize"],
				"chickpea": ["chickpea.jpg", "Chickpea", "Chickpea Description", "https://en.wikipedia.org/wiki/Chickpea"],
				"kidneybeans": ["kidneybeans.jpg", "Kidneybeans", "Kidneybeans Description",  "https://en.wikipedia.org/wiki/Kidney_bean"],
				"pigeonpeas": ["pigeonpeas.jpg", "Pigeonpeas", "Pigeonpeas Description", "https://en.wikipedia.org/wiki/Pigeon_pea"],
				"mothbeans": ["mothbeans.jpg", "Mothbeans", "Mothbeans Description", "https://en.wikipedia.org/wiki/Vigna_aconitifolia"],
				"mungbean": ["mungbean.jpg", "Mungbean", "Mungbean Description", "https://en.wikipedia.org/wiki/Mung_bean"],
				"blackgram": ["blackgram.jpg", "Blackgram", "Blackgram Description", "https://en.wikipedia.org/wiki/Vigna_mungo"],
				"lentil": ["lentil.jpg", "Lentil", "Lentil Description", "https://en.wikipedia.org/wiki/Lentil"],
				"pomegranate": ["pomegranate.jpg", "Pomegranate", "Pomegranate Description", "https://en.wikipedia.org/wiki/Pomegranate"],
				"banana": ["banana.jpg", "Banana", "Banana Description", "https://en.wikipedia.org/wiki/Banana"],
				"mango": ["mango.jpg", "Mango", "Mango Description", "https://en.wikipedia.org/wiki/Mango"],
				"grapes": ["grapes.png", "Grapes", "Grapes Description", "https://en.wikipedia.org/wiki/Grape"],
				"watermelon": ["watermelon.jpg", "Watermelon", "Watermelon Description", "https://en.wikipedia.org/wiki/Watermelon"],
				"muskmelon": ["muskmelon.jpg", "Muskmelon", "Muskmelon Description", "https://en.wikipedia.org/wiki/Cucumis_melo"],
				"apple": ["apple.jpg", "Apple", "Apple Description", "https://en.wikipedia.org/wiki/Apple"],
				"orange": ["orange.jpg", "Orange", "Orange Description", "https://en.wikipedia.org/wiki/Orange_(fruit)"],
				"papaya": ["papaya.jpeg", "Papaya", "Papaya Description", "https://en.wikipedia.org/wiki/Papaya"],
				"coconut": ["coconut.png", "Coconut", "Coconut Description", "https://en.wikipedia.org/wiki/Coconut"],
				"cotton": ["cotton.jpg", "Cotton", "Cotton Description", "https://en.wikipedia.org/wiki/Cotton"],
				"jute": ["jute.jpg", "Jute", "Jute Description", "https://en.wikipedia.org/wiki/Jute"],
				"coffee": ["coffee.jpg", "Coffee", "Coffee Description", "https://en.wikipedia.org/wiki/Coffea"]
			}
processedCropsToResults = [["Rice", "rice.jpg", "rice description", "Crop Info"], ["Maize", "maize.png", "maize description", "Crop Info"], ["Coffee", "coffee.jpg", "coffee description", "Crop Info"]]

def getStructuredInfo(output):
	structuredinfo = []
	for x in output:
		cropsDict[x[0]][2] = "Net return percentage is: "+str(x[1]*100)
		structuredinfo.append(cropsDict[x[0]])
	return structuredinfo

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/results',methods=['POST'])
def results():
  int_features=[float(x) for x in request.form.values()]
  final_features=[np.array(int_features)]
  prediction=model.NearestNeighbors(final_features,3)
  output=prediction
  print(prediction)
  return render_template('results.html',prediction_text=getStructuredInfo(output))
if __name__=="__main__":
  app.run(debug=True)
