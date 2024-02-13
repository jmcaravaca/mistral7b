
import json
from llama_cpp import Llama

class Main(object):
   def __init__(self):
      self.n_ctx = 4096
      self.n_threads = 8
      self.n_gpu_layers = 0
      self.model_filepath = "mistral-7b-instruct-v0.1.Q5_0.gguf"
      self.reinitialize_model()
      self.max_tokens = 64

   def reinitialize_model(self):
      self.llm = Llama(
         model_path=self.model_filepath,  # Download the model file first
         n_ctx=self.n_ctx,  # The max sequence length to use - note that longer sequence lengths require much more resources
         n_threads=self.n_threads,            # The number of CPU threads to use, tailor to your system and the resulting performance
         n_gpu_layers=self.n_gpu_layers         # The number of layers to offload to GPU, if you have GPU acceleration available
      ) 
      
   def predict(self, skill_input):
      # Predict is the only publicly available endpoint so it works both for predictions and to set settings
      try:
         jsoninput = json.loads(skill_input)
         if jsoninput["action"] == "reinit":
            # Reinit model with new parameters
            if "n_ctx" in jsoninput.keys():
               self.n_ctx = int(jsoninput["n_ctx"])
            if "n_threads" in jsoninput.keys():               
               self.n_threads =  int(jsoninput["n_threads"])
            if "n_gpu_layers" in jsoninput.keys():
               self.n_gpu_layers =  int(jsoninput["n_gpu_layers"])
            if "max_tokens" in jsoninput.keys():
               self.max_tokens =  int(jsoninput["max_tokens"])
            self.reinitialize_model()
            return "Reinitialization successful"
         elif jsoninput["action"] == "predict":
            self.max_tokens = int(jsoninput["max_tokens"])
            predictiontext = str(jsoninput["text"])
      except Exception as e:
         #Not a json input, simply predict with existing defaults
         predictiontext = skill_input
      try:
         output = self.llm(
         predictiontext, # AQUÍ SE ESCRIBE EL PROMPT
         max_tokens=self.max_tokens,  # Generate up to 512 tokens
         stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
         echo=False        # Whether to echo the prompt
         )
         return json.dumps(output["choices"][0]["text"])
      except Exception as e:
         return "Error during prediction"
      
       
if __name__ == '__main__':
   a = Main()
   reinit_action = {
      "action" : "reinit",
      "n_ctx" : 256
   }
   a.predict(json.dumps(reinit_action))
   results = a.predict(skill_input="How hot is the surface of the Sun")
   print(results)