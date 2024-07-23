from aiohttp import web
import os
import onnx
import numpy as np
from onnx import numpy_helper
from artifacts import generate_artifatcs
# import pdb

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE,"model/inference.onnx")
print(MODEL_PATH)

model = onnx.load(MODEL_PATH)

async def update_model(request):
    
    try:
        
        data = await request.json()
        
        updated_weights = data.get("updated_weights")
                
        updated_weights_list = list(updated_weights["cpuData"].values())
        
        print(len(updated_weights_list))
        
        updated_weights_array = np.array(updated_weights_list, dtype=np.float32).reshape((1000,768))
        
        CLASSIFIER_WEIGHT = updated_weights_array.astype(np.float32)
        
        # pdb.set_trace()
        
        for initializer in model.graph.initializer:
            
            if initializer.name == "classifier.weight":
                # pdb.set_trace()
                print(f'Original dims: {initializer.dims}')
                new_weights_tensor = numpy_helper.from_array(CLASSIFIER_WEIGHT, initializer.name)
                initializer.CopyFrom(new_weights_tensor)
            
        onnx.save_model(model, MODEL_PATH)
        
        data = {"message": "MODEL AND TRAINING ARTIFACTS UPDATED"}
        
        generate_artifatcs()
        
        return web.json_response(data)
    
    except Exception as e:
        
        error_message = str(e)
        response_data = {"error": error_message}
        return web.json_response(response_data, status=500)


async def index(request):
    return web.FileResponse("./index.html")

async def style(request):
    return web.FileResponse("./style.css")