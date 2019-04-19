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

export_file_url = 'https://www.dropbox.com/s/uvyf78rjdwm8f7b/resnet50_json92_40epoch_0896471.pth?dl=1'
export_file_name = 'resnet50_json92_40epoch_0896471'

#export_file_url = 'https://www.dropbox.com/s/1abrij8d4cinrts/squeeze_UNTRAINED_org_0415.pth?dl=1'
#export_file_name = 'squeeze_UNTRAINED_org_0415'

#classes = ['Venalfaxine 37.5mg', 'Venalfaxine ER 75mg', 'Venalfaxine ER 150mg', 'Levothyroxine 25mcg', 'Levothyroxine 50mcg', 'Levothyroxine 75mcg', 'Levothyroxine 100mcg', 'Levothyroxine 112mcg', 'Omeprazole 20mg', 'Lisinopril 5mg', 'Lisinopril 10mg', 'Lisinopril 20mg', 'Atorvastatin 10mg', 'Atorvastatin 20mg', 'Atorvastatin 40mg', 'Duloxetine 20mg', 'Duloxetine 30mg', 'Duloxetine 60mg', 'Levoxyl 25mcg', 'Levoxyl 50mcg', 'Levoxyl 88mcg', 'Levoxyl 112mcg', 'Gabapentin 100mg', 'Gabapentin #300mg', 'Sertraline 25mg', 'Sertraline 50mg', 'Sertraline 100mg', 'Gabapentin 600mg', 'Gabapentin 800mg', 'Omeprazole 40mg']

#classes = ['000937384', '000937385', '000937386', '003781800', '003781803', '003781805', '003781809', '003781811', '007812790', '435470352', '435470353', '435470354', '605052578', '605052579', '605052580', '605052995', '605052996', '605052997', '607930850', '607930851', '607930853', '607930855', '658620198',
#'658620199', '681800351', '681800352', '681800353', '684620126', '684620127', '684620397']
classes = ['000937384', '000937385', '000937386', '003781800', '003781803', '003781805', '003781809', '003781811', '007812790',
           '435470352', '435470353', '435470354', '605052578', '605052579', '605052580', '605052995', '605052996', '605052997',
           '607930850', '607930851', '607930853', '607930855', '658620198', '658620199', '681800351', '681800352', '681800353',
           '684620126', '684620127', '684620397', '000024463', '000024464', '000544179', '000544181', '000544182', '000544183',
           '000544184', '000694220', '000711012', '000711013', '000711014', '000711015', '000711016', '000930753', '000932210',
           '000933123', '000933125', '001725728', '001725729', '001850674', '003642337', '003781160', '003784250', '003786410',
           '005910844', '005910900', '107020026', '107020027', '136680113', '136680114', '136680115', '433860356', '501110433',
           '501110434', '591480006', '591480011', '620370710', '620370999', '633040693', '651620076', '651620077', '651620627',
           '658620185', '658620448', '658620449', '659775036', '659775037', '669930060', '681800135', '681800136', '681800137',
           '681800302', '681800303', '681800396', '681800397', '681800513', '681800517', '681800590', '681800591', '681800980',
           '681800981', '683820022']

#with open('app/static/json_test.json', 'r') as f:
#    cat_to_name = json.load(f)

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
     #data = ImageDataBunch.from_folder(path, bs=64, size=296)
     #data.normalize(imagenet_stats)
     learn = cnn_learner(data_bunch, models.resnet50, pretrained=False)
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
    pred_1_class, indice, losses = learn.predict(img)
    preds_sorted, idxs = losses.sort(descending=True)

    with open('app/static/json_92_version1.json', 'r') as f:
        cat_to_name = json.load(f)

    #class_to_idx = {sorted(cat_to_name)[i]: i for i in range(len(cat_to_name))}

    #idx_to_class = {val: key for key, val in class_to_idx.items()}

    pred_1_class = learn.data.classes[idxs[0]]
    pred_2_class = learn.data.classes[idxs[1]]

    pred_1_prob = np.round(100*preds_sorted[0].item(),2)
    pred_2_prob = np.round(100*preds_sorted[1].item(),2)

    pred_name = cat_to_name[str(prediction)]
    pred_drug = pred_name['name']
    pred_shape = pred_name['shape']
    pred_color = pred_name['color']
    pred_marking = pred_name['marking']


    #pred_1_class = learn.data.classes[idxs[0]]
    #pred_2_class = learn.data.classes[idxs[1]]
    #pred_1_class_name = pred_1_class['name']
    #pred_1_class_shape = pred_1_class['shape']
    #pred_1_class_color = pred_1_class['color']
    #pred_1_class_marking = pred_1_class['marking']

    if pred_1_prob <= 80:
            #rs+='<p>(Note: Model is NOT confident with this prediction)</p>\n'
        result = (f' Model IS NOT Confident: \n ({pred_1_prob}%) \n Shape: {pred_shape} \n Color: {pred_color}')

    else:
            #rs+=(f'<p>(Model IS confident: )</p>' + first_choice)
            #rs+=f'<p>Model IS confident <b>{first_choice}</b> prediction: </p>\n'
        result = (f' Model IS confident: Drug Name: \n {pred_drug} \n ({pred_1_prob}% ) Shape: {pred_shape} \n Color: {pred_color} \n Marking: {pred_marking} \n NDC: {prediction}')

    info = learn.data.classes

    #result = (f' Drug Name: \n {pred_drug} \n ({pred_1_prob}% ) Shape: {pred_shape} \n Color: {pred_color} \n Marking: {pred_marking} \n NDC: {prediction}')




    #result = (f' Model output: \n {prediction} {pred_1_class}\n {pred_1_prob} {pred_2_class} {pred_2_prob}')


    #return JSONResponse({'result': str(result)})
    return JSONResponse({'result': str(result)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
