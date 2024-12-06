from fms import models
import inspect
model = models.get_model('llama', '7b', device_type='cuda', distributed_strategy='tp')
#print(inspect.getsource(models.get_model))