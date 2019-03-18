from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
import json

from fastai import *
from fastai.vision import *

#export_file_url = 'https://www.dropbox.com/s/v6cuuvddq73d1e0/export.pkl?raw=1'
#export_file_url = 'https://drive.google.com/open?id=1tpHsmC8kn_EJlozFe_L1BPpijPgRMn84'
#Working Squeezenet
#export_file_url = 'https://www.dropbox.com/s/jdm3feb2xt2cimj/squeezenet_v1.pkl?dl=1'
#Working ResNet
#export_file_url = 'https://www.dropbox.com/s/19xv4nd7f68z63o/resnet50_v1.pkl?dl=1'
export_file_url = 'https://www.dropbox.com/s/sauftmi8tp8axyh/json.pkl?dl=1'
export_file_name = 'json.pkl'

with open('app/static/Test30.json', 'r') as f:
    cat_to_name = json.load(f)

#class_names = data.classes

#for i in range(0,len(class_names)):
#    class_names[i] = cat_to_name.get(class_names[i])

#classes = ['beige', 'black', 'blue', 'brown', 'capsule', 'gold', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'tablet', 'tan', 'white', 'yellow']
classes = ['Venalfaxine 37.5mg', 'Venalfaxine ER 75mg', 'Venalfaxine ER 150mg', 'Levothyroxine 25mcg', 'Levothyroxine 50mcg', 'Levothyroxine 75mcg', 'Levothyroxine 100mcg', 'Levothyroxine 112mcg', 'Omeprazole 20mg', 'Lisinopril 5mg', 'Lisinopril 10mg', 'Lisinopril 20mg', 'Atorvastatin 10mg', 'Atorvastatin 20mg', 'Atorvastatin 40mg', 'Duloxetine 20mg', 'Duloxetine 30mg', 'Duloxetine 60mg', 'Levoxyl 25mcg', 'Levoxyl 50mcg', 'Levoxyl 88mcg', 'Levoxyl 112mcg', 'Gabapentin 100mg', 'Gabapentin 300mg', 'Sertraline 25mg', 'Sertraline 50mg', 'Sertraline 100mg', 'Gabapentin 600mg', 'Gabapentin 800mg', 'Omeprazole 40mg']
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
    await download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

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
    prediction = learn.predict(img)[0]
    name_two = class_names[prediction]
    print (name_two)
    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
