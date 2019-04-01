from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
import json

from fastai import *
from fastai.vision import *

#Densenet
#export_file_url = 'https://www.dropbox.com/s/ubtyr33aa2tkvzk/DenseNet201_1_0322.pth?dl=1'
#export_file_name = 'DenseNet201_1_0322'

export_file_url = 'https://www.dropbox.com/s/gpw07b7vn47q3k9/squeezenet1_0_0327_B64.pth?dl=1'
export_file_name = 'squeezenet1_0_0327_B64'

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

#async def download_file(url, dest):
#    if dest.exists(): return
#    async with aiohttp.ClientSession() as session:
#        async with session.get(url) as response:
#            data = await response.read()
#            with open(dest, 'wb') as f: f.write(data)

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

#async def setup_learner():
#    await download_file(export_file_url, path/'models'/f'{export_file_name}.pth')
#    data_bunch = ImageDataBunch.single_from_classes(path, classes, size=128).normalize(imagenet_stats)
#    learn = cnn_learner(data_bunch, models.squeezenet1_0, pretrained=False)
#    learn.load(export_file_name)
#    return learn

#loop = asyncio.get_event_loop()
#tasks = [asyncio.ensure_future(setup_learner())]
#learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
#loop.close()


async def setup_learner():
    await download_file(export_file_url, path/'models'/f'{export_file_name}.pth')
    try:
        learn = load_learner(export_file_url, export_file_name)
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
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    prediction, indice, losses = learn.predict(img)
    preds_sorted, idxs = losses.sort(descending=True)


    class_names = learn.data.classes

    for i in range(0,len(class_names)):
        class_names[i] = cat_to_name.get(class_names[i])

        pred_1_class = class_names[idxs[0]]
        pred_2_class = class_names[idxs[1]]

        pred_1_prob = np.round(100*preds_sorted[0].item(),2)
        pred_2_prob = np.round(100*preds_sorted[1].item(),2)

    #rs = '<p>PREDICTION:</p>\n'
    if pred_1_prob <= 80:
        #rs+='<p>(Note: Model is NOT confident with this prediction)</p>\n'
        result = (f' Model is NOT Confident: \n {pred_1_class} ({pred_1_prob}%)')

    else:
        #rs+=(f'<p>(Model IS confident: )</p>' + first_choice)
        #rs+=f'<p>Model IS confident <b>{first_choice}</b> prediction: </p>\n'
        result = (f'Model IS Confident: \n {pred_1_class} ({pred_1_prob}%)')


    return JSONResponse({'result': str(result)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
