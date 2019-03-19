from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
import json

from fastai import *
from fastai.vision import *

export_file_url = 'https://www.dropbox.com/s/nio1dlb2nk73f82/Original0319.pkl?dl=1'
export_file_name = 'Original0319.pkl'

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

    #preds_sorted, idxs = prediction(descending=True)

    #pred_1_prob = np.round(100*prediction[0].item(),2)
    #pred_2_prob = np.round(100*preds_sorted[1].item(),2)
    #pred_3_prob = np.round(100*preds_sorted[2].item(),2)
    #preds_best3 = [f'{pred_1_class} ({pred_1_prob}%)', f'{pred_1_class} ({pred_2_prob}%)', f'{pred_1_class} ({pred_3_prob}%)']
    #preds_best3 = [f'({pred_1_prob}%)']

    #output = ((preds_best3), (prediction))
    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
