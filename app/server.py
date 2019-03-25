from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
import json

from fastai import *
from fastai.vision import *

#PTH upload
export_file_url = 'https://www.dropbox.com/s/ubtyr33aa2tkvzk/DenseNet201_1_0322.pth?dl=1'
export_file_name = 'DenseNet201_1_0322'
classes = ['Venalfaxine 37.5mg', 'Venalfaxine ER 75mg', 'Venalfaxine ER 150mg', 'Levothyroxine 25mcg', 'Levothyroxine 50mcg', 'Levothyroxine 75mcg', 'Levothyroxine 100mcg', 'Levothyroxine 112mcg', 'Omeprazole 20mg', 'Lisinopril 5mg', 'Lisinopril 10mg', 'Lisinopril 20mg', 'Atorvastatin 10mg', 'Atorvastatin 20mg', 'Atorvastatin 40mg', 'Duloxetine 20mg', 'Duloxetine 30mg', 'Duloxetine 60mg', 'Levoxyl 25mcg', 'Levoxyl 50mcg', 'Levoxyl 88mcg', 'Levoxyl 112mcg', 'Gabapentin 100mg', 'Gabapentin 300mg', 'Sertraline 25mg', 'Sertraline 50mg', 'Sertraline 100mg', 'Gabapentin 600mg', 'Gabapentin 800mg', 'Omeprazole 40mg']

#AWS
#export_file_url = 'https://www.dropbox.com/s/xhoz8abah0tx7eo/resnet_one_2_0319_AWS.pkl?dl=1'
#export_file_name = 'resnet_one_2_0319_AWS.pkl'

#Resnet152
#export_file_url = 'https://www.dropbox.com/s/2nb6cc5y98lan1q/resnet152_0320.pkl?dl=1'
#export_file_name= 'resnet152_0320.pkl'

#densenet
#export_file_url = 'https://www.dropbox.com/s/8y8x1v4euesh747/densenet201_0322_AWS.pkl?dl=1'
#export_file_name = 'densenet201_0322_AWS.pkl'

#Resnet18
#export_file_url = 'https://www.dropbox.com/s/3u4v1yzyqm2pjek/resnet18_1_0322.pkl?dl=1'
#export_file_name= 'resnet18_1_0322'
#Working Names
#export_file_url = 'https://www.dropbox.com/s/yce5otqijrpfs8o/pill_3.pkl?dl=1'
#export_file_name = 'pill_3.pkl'

with open('app/static/Test30.json', 'r') as f:
    cat_to_name = json.load(f)

path = Path(__file__).parent


app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

def accuracy_thresh(y_pred:Tensor, y_true:Tensor, thresh:float=0.5, sigmoid:bool=True)->Rank0Tensor:
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: y_pred = y_pred.sigmoid()
    return ((y_pred>thresh)==y_true.byte()).float().mean()

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/'models'/f'{export_file_name}.pth')
    data_bunch = ImageDataBunch.single_from_classes(path, classes, size=128).normalize(imagenet_stats)
    learn = cnn_learner(data_bunch, models.densenet201, pretrained=False)
    learn.load(export_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/picture')
def picture(request):
    html = path/'view'/'picture.html'
    return HTMLResponse(html.open().read())

@app.route('/info')
def info(request):
    html = path/'view'/'info.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    name = export_file_name
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    #prediction = learn.predict(img)[0]
    prediction = sorted(zip(classes, map(float, losses)), key=lambda p: p[1], reverse=True)

    rs = '<p>Top 3 predictions:</p>\n'
    for clas,pr in predictions[:3]:
        rs+=f'<p> -{mv_dict[clas]}: {(pr*100):.2f}% </p>\n'
    if predictions[0][1] <= 0.70:
        rs+='<p>(Note: Model is not confident with this prediction)</p>\n'

    rs+=f'<p>Which part of the image the model considered for <b>{mv_dict[predictions[0][0]]}</b> prediction: </p>\n'

    class_names = learn.data.classes

    #answer = cat_to_name.get(class_names[prediction])

    for i in range(0,len(class_names)):
        class_names[i] = cat_to_name.get(class_names[i])

    pred_1_class, indice, preds = learn.predict(img)

    preds_sorted, idxs = preds.sort(descending=True)

    pred_1_class = learn.data.classes[idxs[0]]
    pred_2_class = learn.data.classes[idxs[1]]
    pred_3_class = learn.data.classes[idxs[2]]
    pred_4_class = learn.data.classes[idxs[3]]
    pred_5_class = learn.data.classes[idxs[4]]

    pred_1_prob = np.round(100*preds_sorted[0].item(),2)
    pred_2_prob = np.round(100*preds_sorted[1].item(),2)
    pred_3_prob = np.round(100*preds_sorted[2].item(),2)
    pred_4_prob = np.round(100*preds_sorted[3].item(),2)
    pred_5_prob = np.round(100*preds_sorted[4].item(),2)

    preds_best3 = [f'{pred_1_class} ({pred_1_prob}%)', f'{pred_2_class} ({pred_2_prob}%)', f'{pred_3_class} ({pred_3_prob}%)', f'{pred_4_class} ({pred_3_prob}%)', f'{pred_5_class} ({pred_5_prob}%)']

    if pred_1_prob < 80:
        result = (f' NOT Confident: \n {pred_1_class} ({pred_1_prob}%)')
    else:
        result = (f'Confident: \n {pred_1_class} ({pred_1_prob}%)')
        #output = (f'({result})\n {preds_best3[0]}\n {preds_best3[1]}\n {preds_best3[2]}')

    #output = ((preds_best3))
    return JSONResponse({'result': str(rs)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
