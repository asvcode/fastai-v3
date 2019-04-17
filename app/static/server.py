from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

from fastai.vision.models import cadene_models
import pretrainedmodels
#Cardene Squeezenet
#export_file_url = 'https://www.dropbox.com/s/6ubzhbra6rc1zbd/cardene_sq.pkl?dl=1'
#export_file_name = 'cardene_sq.pkl'

export_file_url = 'https://www.dropbox.com/s/i6wd5fhoofkl0rc/squeeze_UNTRAINED_rerun_one_overfit_30_0415.pth?dl=1'
export_file_name = 'squeeze_UNTRAINED_rerun_one_overfit_30_0415'

#export_file_url = 'https://www.dropbox.com/s/1abrij8d4cinrts/squeeze_UNTRAINED_org_0415.pth?dl=1'
#export_file_name = 'squeeze_UNTRAINED_org_0415'

classes = ['Venalfaxine 37.5mg', 'Venalfaxine ER 75mg', 'Venalfaxine ER 150mg', 'Levothyroxine 25mcg', 'Levothyroxine 50mcg', 'Levothyroxine 75mcg', 'Levothyroxine 100mcg', 'Levothyroxine 112mcg', 'Omeprazole 20mg', 'Lisinopril 5mg', 'Lisinopril 10mg', 'Lisinopril 20mg', 'Atorvastatin 10mg', 'Atorvastatin 20mg', 'Atorvastatin 40mg', 'Duloxetine 20mg', 'Duloxetine 30mg', 'Duloxetine 60mg', 'Levoxyl 25mcg', 'Levoxyl 50mcg', 'Levoxyl 88mcg', 'Levoxyl 112mcg', 'Gabapentin 100mg', 'Gabapentin 300mg', 'Sertraline 25mg', 'Sertraline 50mg', 'Sertraline 100mg', 'Gabapentin 600mg', 'Gabapentin 800mg', 'Omeprazole 40mg']

with open('app/static/json_test.json', 'r') as f:
    cat_to_name = json.load(f)

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
     await download_file(export_file_url, path/'models'/f'{export_file_name}.pth')
     data_bunch = ImageDataBunch.single_from_classes(path, classes, size=296).normalize(imagenet_stats)
     learn = cnn_learner(data_bunch, models.squeezenet1_0, pretrained=False)
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

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    prediction, indice, losses = learn.predict(img)
    preds_sorted, idxs = losses.sort(descending=True)

    with open('app/static/json_test.json', 'r') as f:
        cat_to_name = json.load(f)

    class_to_idx = {sorted(cat_to_name)[i]: i for i in range(len(cat_to_name))}

    #idx_to_class = {val: key for key, val in class_to_idx.items()}

    pred_1_class = learn.data.classes[idxs[0]]
    pred_2_class = learn.data.classes[idxs[1]]

    pred_1_prob = np.round(100*preds_sorted[0].item(),2)
    pred_2_prob = np.round(100*preds_sorted[1].item(),2)

    #pred_1_class = learn.data.classes[idxs[0]]
    #pred_2_class = learn.data.classes[idxs[1]]
    pred_1_class_name = pred_1_class['name']
    pred_1_class_shape = pred_1_class['shape']
    pred_1_class_color = pred_1_class['color']
    pred_1_class_marking = pred_1_class['marking']

    info = learn.data.classes

    #result = (f' info: \n  str{prediction} {pred_1_class} ({pred_1_prob}%)')

    if pred_1_prob <= 80:
        #rs+='<p>(Note: Model is NOT confident with this prediction)</p>\n'
        result = (f' Model is NOT Confident: \n ({pred_1_prob}%) \n {pred_1_class_shape} \n {pred_1_class_color}')

    else:
        #rs+=(f'<p>(Model IS confident: )</p>' + first_choice)
        #rs+=f'<p>Model IS confident <b>{first_choice}</b> prediction: </p>\n'
        result = (f'Model IS Confident: \n {pred_1_class} ({pred_1_prob}%) \n {pred_1_class_shape} {pred_1_class_color} {pred_1_class_marking}')



    #result = (f' Model output: \n {prediction} {pred_1_class}\n {pred_1_prob} {pred_2_class} {pred_2_prob}')


    return JSONResponse({'result': str(result)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)